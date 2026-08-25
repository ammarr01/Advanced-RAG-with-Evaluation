import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_SET_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "golden_set.jsonl")
RUNS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "runs")


def load_pipeline(pipeline):
    """Returns (retrieve_fn, rag_chain_with_sources, retrieval_meta) for the given pipeline.

    retrieve_fn must implement .invoke(question) -> list[Document] and must
    reflect the pipeline's full retrieval path (query translation + reranking
    for advanced), not just its base vector search.
    """
    if pipeline == "naive":
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "naive-rag"))
        from naive_rag import retriever, rag_chain_with_sources
        return retriever, rag_chain_with_sources, {"retriever_k": retriever.search_kwargs.get("k")}
    if pipeline == "advanced":
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "advanced-rag"))
        import advanced_rag
        retrieval_meta = {
            "retriever_k": advanced_rag.FINAL_K,
            "candidate_k": advanced_rag.CANDIDATE_K,
            "multi_query_n": advanced_rag.MULTI_QUERY_N,
            "max_subquestions": advanced_rag.MAX_SUBQUESTIONS,
        }
        return advanced_rag.retrieve_runnable, advanced_rag.rag_chain_with_sources, retrieval_meta
    raise ValueError(f"Unknown pipeline: {pipeline!r}")


def load_golden_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_sha():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
            ).strip()
        )
        return sha, dirty
    except subprocess.CalledProcessError:
        return None, None


def docs_to_retrieved(docs):
    return [
        {
            "rank": rank,
            "doc_id": doc.metadata.get("doc_id"),
            "char_start": doc.metadata.get("char_start"),
            "char_end": doc.metadata.get("char_end"),
            "text": doc.page_content,
        }
        for rank, doc in enumerate(docs)
    ]


def run_retrieval(pipeline, retriever, golden_set, retrieval_meta):
    golden_set = [item for item in golden_set if item.get("gold_evidence")]
    print(f"Loaded {len(golden_set)} questions with gold_evidence (out_of_scope excluded)")

    results = []
    for i, item in enumerate(golden_set):
        print(f"  [{i + 1}/{len(golden_set)}] ({item['category']}) {item['question']}")
        docs = retriever.invoke(item["question"])
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "retrieved": docs_to_retrieved(docs),
            }
        )
    return results, retrieval_meta


def run_generation(pipeline, chain, golden_set):
    print(f"Loaded {len(golden_set)} questions (all categories, including out_of_scope)")

    results = []
    for i, item in enumerate(golden_set):
        print(f"  [{i + 1}/{len(golden_set)}] ({item['category']}) {item['question']}")
        output = chain.invoke(item["question"])
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": output["answer"],
                "retrieved": docs_to_retrieved(output["docs"]),
            }
        )
    return results, {}


def run(pipeline, run_type, run_index=1):
    retriever, chain, retrieval_meta = load_pipeline(pipeline)
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    sha, dirty = git_sha()

    if run_type == "retrieval":
        results, extra = run_retrieval(pipeline, retriever, golden_set, retrieval_meta)
    elif run_type == "generation":
        results, extra = run_generation(pipeline, chain, golden_set)
    else:
        raise ValueError(f"Unknown run type: {run_type!r}")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_record = {
        "pipeline": pipeline,
        "run_type": run_type,
        "run_index": run_index,
        "timestamp_utc": timestamp,
        "git_sha": sha,
        "git_dirty": dirty,
        "golden_set_sha256": file_sha256(GOLDEN_SET_PATH),
        "golden_set_count": len(results),
        **extra,
        "results": results,
    }

    out_dir = os.path.join(RUNS_DIR, pipeline, run_type)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, ensure_ascii=False, indent=2)

    print(f"\nWrote raw {run_type} run to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=["naive", "advanced"], default="naive")
    parser.add_argument("--type", choices=["retrieval", "generation"], required=True)
    parser.add_argument(
        "--run-index",
        type=int,
        default=1,
        help="Which repeat this is (generation metrics need >=3 repeats per your eval rules)",
    )
    args = parser.parse_args()
    run(args.pipeline, args.type, run_index=args.run_index)