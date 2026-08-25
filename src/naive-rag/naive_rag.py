import os
import glob
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredHTMLLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma")

loader_map = {
    ".pdf": (PyMuPDFLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
}


def load_all_docs(root_dir):
    docs = []
    skipped = []
    for path in glob.glob(os.path.join(root_dir, "**", "*.*"), recursive=True):
        ext = os.path.splitext(path)[1].lower()
        entry = loader_map.get(ext)
        if not entry:
            skipped.append((path, "unsupported extension"))
            continue
        loader_cls, kwargs = entry
        doc_id = os.path.relpath(path, root_dir).replace(os.sep, "/")
        try:
            loaded = loader_cls(path, **kwargs).load()
            if not loaded:
                skipped.append((path, "loader returned no content"))
                continue 
            merged_text = "".join(d.page_content for d in loaded)
            metadata = dict(loaded[0].metadata)
            metadata.pop("page", None)
            metadata["source"] = doc_id
            metadata["doc_id"] = doc_id
            docs.append(Document(page_content=merged_text, metadata=metadata))
        except Exception as e:
            skipped.append((path, str(e)))
    if skipped:
        print(f"Skipped {len(skipped)} files")
    return docs


embedding = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(CHROMA_PATH):
    print("Loading vectorstore from disk...")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
else:
    print("Building vectorstore from scratch...")
    docs = load_all_docs(DATA_DIR)
    print(f"Loaded {len(docs)} documents")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    splits = splitter.split_documents(docs)
    for split in splits:
        char_start = split.metadata.pop("start_index")
        split.metadata["char_start"] = char_start
        split.metadata["char_end"] = char_start + len(split.page_content)
    print(f"Split into {len(splits)} chunks")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=CHROMA_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

template = """Answer the question based only on the following context.
If the context does not contain enough information to answer the question, say "I don't know" and nothing else.

Context:
{context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def format_docs(docs):
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_sources = RunnableParallel(
    {"docs": retriever, "question": RunnablePassthrough()}
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
