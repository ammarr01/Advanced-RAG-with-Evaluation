import glob
import os

import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from sentence_transformers import CrossEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma")

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".md", ".docx", ".pptx"}

CANDIDATE_K = 5
FINAL_K = 4
MULTI_QUERY_N = 3
MAX_SUBQUESTIONS = 4
RAPTOR_MIN_CHUNKS = 10
RAPTOR_MAX_CLUSTERS = 10


def load_all_docs(root_dir):
    # Tables are kept as HTML instead of being flattened to plain text.
    docs = []
    skipped = []
    for path in glob.glob(os.path.join(root_dir, "**", "*.*"), recursive=True):
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped.append((path, "unsupported extension"))
            continue
        doc_id = os.path.relpath(path, root_dir).replace(os.sep, "/")
        kwargs = {"mode": "elements", "infer_table_structure": True}
        if ext == ".pdf":
            kwargs["strategy"] = "hi_res"
        try:
            elements = UnstructuredLoader(path, **kwargs).load()
            if not elements:
                skipped.append((path, "loader returned no content"))
                continue
            parts = []
            for el in elements:
                if el.metadata.get("category") == "Table" and el.metadata.get("text_as_html"):
                    parts.append(el.metadata["text_as_html"])
                else:
                    parts.append(el.page_content)
            merged_text = "\n\n".join(parts)
            docs.append(Document(page_content=merged_text, metadata={"source": doc_id, "doc_id": doc_id}))
        except Exception as e:
            skipped.append((path, str(e)))
    if skipped:
        print(f"Skipped {len(skipped)} files")
    return docs


def split_documents_semantic(docs, embedding):
    # Splits on meaning (embedding similarity between sentences), not a fixed character count.
    chunker = SemanticChunker(embedding)
    splits = []
    for doc in docs:
        chunks = chunker.split_text(doc.page_content)
        cursor = 0
        for chunk in chunks:
            idx = doc.page_content.find(chunk, cursor)
            if idx == -1:
                idx = doc.page_content.find(chunk)
            char_start = idx if idx != -1 else None
            char_end = idx + len(chunk) if idx != -1 else None
            if idx != -1:
                cursor = idx + len(chunk)
            metadata = dict(doc.metadata)
            metadata["char_start"] = char_start
            metadata["char_end"] = char_end
            splits.append(Document(page_content=chunk, metadata=metadata))
    return splits


raptor_summary_prompt = PromptTemplate.from_template(
    """Write a concise summary of the following related passages. Capture the
shared topic and the key facts so the summary alone is useful for retrieving
this group of passages by topic.

Passages:
{text}"""
)


def build_raptor_summaries(splits, embedding, llm):
    # RAPTOR, single level: cluster leaf chunks by embedding similarity, then
    # add an LLM summary per cluster as an extra retrievable node.
    max_k = min(RAPTOR_MAX_CLUSTERS, len(splits) - 1)
    if len(splits) < RAPTOR_MIN_CHUNKS or max_k < 2:
        return []

    vectors = np.array(embedding.embed_documents([s.page_content for s in splits]))

    best_k, best_labels, best_score = 2, None, -1
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(vectors)
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score

    summarize = raptor_summary_prompt | llm | StrOutputParser()
    summaries = []
    for cluster_id in range(best_k):
        members = [s for s, label in zip(splits, best_labels) if label == cluster_id]
        cluster_text = "\n\n".join(m.page_content for m in members)
        summary = summarize.invoke({"text": cluster_text})
        cluster_key = f"raptor-cluster-{cluster_id}"
        summaries.append(Document(
            page_content=summary,
            metadata={
                "source": cluster_key,
                "doc_id": cluster_key,
                "char_start": None,
                "char_end": None,
                "raptor_level": 1,
            },
        ))
    return summaries


embedding = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

if os.path.exists(CHROMA_PATH):
    print("Loading vectorstore from disk...")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
else:
    print("Building vectorstore from scratch...")
    docs = load_all_docs(DATA_DIR)
    print(f"Loaded {len(docs)} documents")
    splits = split_documents_semantic(docs, embedding)
    print(f"Split into {len(splits)} chunks")
    raptor_summaries = build_raptor_summaries(splits, embedding, llm)
    print(f"Built {len(raptor_summaries)} RAPTOR cluster summaries")
    vectorstore = Chroma.from_documents(
        documents=splits + raptor_summaries, embedding=embedding, persist_directory=CHROMA_PATH
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": CANDIDATE_K})
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Query translation: multi-query ---
multi_query_prompt = PromptTemplate.from_template(
    """You are helping a retrieval system find relevant documents.
Generate {n} alternative phrasings of the question below. Vary vocabulary and
sentence structure but preserve the original meaning exactly.
Return one rephrasing per line, with no numbering, bullets, or commentary.

Question: {question}"""
)

# --- Query translation: task decomposition ---
decompose_prompt = PromptTemplate.from_template(
    """Break the question below into the simplest set of sub-questions that
would together be needed to answer it. If the question is already
single-part, return it unchanged as the only line. Return one sub-question
per line, with no numbering, bullets, or commentary. Produce at most
{max_subquestions} sub-questions.

Question: {question}"""
)


def parse_lines(text):
    return [line.strip() for line in text.strip().split("\n") if line.strip()]


multi_query_chain = multi_query_prompt | llm | StrOutputParser() | parse_lines
decompose_chain = decompose_prompt | llm | StrOutputParser() | parse_lines


def dedupe_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        key = (doc.metadata.get("doc_id"), doc.metadata.get("char_start"), doc.metadata.get("char_end"))
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def rerank(question, docs):
    if not docs:
        return docs
    pairs = [(question, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:FINAL_K]]


def retrieve(question):
    multi_queries = multi_query_chain.invoke({"question": question, "n": MULTI_QUERY_N})
    sub_questions = decompose_chain.invoke({"question": question, "max_subquestions": MAX_SUBQUESTIONS})

    candidates = []
    for q in [question] + multi_queries + sub_questions:
        candidates.extend(retriever.invoke(q))
    candidates = dedupe_docs(candidates)

    return rerank(question, candidates)


retrieve_runnable = RunnableLambda(retrieve)

template = """Answer the question based only on the following context.
If the context does not contain enough information to answer the question, say "I don't know" and nothing else.

Context:
{context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


rag_chain = (
    {"context": retrieve_runnable | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_sources = RunnableParallel(
    {"docs": retrieve_runnable, "question": RunnablePassthrough()}
) | RunnablePassthrough.assign(
    answer=(
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
)

if __name__ == "__main__":
    result = rag_chain.invoke("How to Activate UPI")
    print(result)
