"""
Score run files against the golden set: retrieval (recall@k, nDCG), generation
(faithfulness, relevancy, correctness), and abstention (false-answer rate,
over-abstention rate). Writes a headline table to docs/results.md.

Retrieval scoring is deterministic, no judge, cheap to re-run.
Generation and abstention scoring use gpt-5.4 Nano as an LLM judge -- note this
is the same model naive's own generation uses, a known limitation (the judge
isn't fully independent of the pipeline it's grading), documented rather than
hidden.

Usage:
    python src/evaluate.py --retrieval-run benchmarks/runs/naive/retrieval/<ts>.json
    python src/evaluate.py --generation-run benchmarks/runs/naive/generation/<ts>.json
    python src/evaluate.py --retrieval-run ... --generation-run ...   # both, one results.md entry
"""
import argparse
import json
import math
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_SET_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "golden_set.jsonl")
RESULTS_MD_PATH = os.path.join(PROJECT_ROOT, "docs", "results.md")
JUDGE_MODEL = "gpt-5.4-nano"


def load_golden_set(path):
    with open(path, "r", encoding="utf-8") as f:
        golden = {}
        for line in f:
            if line.strip():
                item = json.loads(line)
                golden[item["id"]] = item
        return golden


def load_run(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- retrieval

def get_extracted_texts(pipeline):
    """{doc_id: full_extracted_text} for this pipeline, via its own loader."""
    if pipeline == "naive":
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "naive-rag"))
        from naive_rag import load_all_docs, DATA_DIR
        docs = load_all_docs(DATA_DIR)
        return {d.metadata["doc_id"]: d.page_content for d in docs}
    if pipeline == "advanced":
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "advanced-rag"))
        from advanced_rag import load_all_docs, DATA_DIR
        docs = load_all_docs(DATA_DIR)
        return {d.metadata["doc_id"]: d.page_content for d in docs}
    raise ValueError(f"Unknown pipeline: {pipeline!r}")


def resolve_offsets(quote, doc_id, extracted_texts):
    text = extracted_texts.get(doc_id)
    if text is None:
        return None, "doc_id not found in extraction"
    idx = text.find(quote)
    if idx == -1:
        return None, "quote not found verbatim in extracted text"
    if text.count(quote) > 1:
        return (idx, idx + len(quote)), "warning: quote appears more than once, used first match"
    return (idx, idx + len(quote)), None


def spans_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def score_retrieval_question(gold_spans, retrieved):
    n_gold = len(gold_spans)
    if n_gold == 0:
        return None

    relevance = []
    matched_gold = set()
    for r in retrieved:
        is_hit = False
        for gi, (gs, ge, gdoc) in enumerate(gold_spans):
            if r["doc_id"] == gdoc and r.get("char_start") is not None:
                if spans_overlap(r["char_start"], r["char_end"], gs, ge):
                    is_hit = True
                    matched_gold.add(gi)
        relevance.append(1 if is_hit else 0)

    recall = len(matched_gold) / n_gold

    def dcg(rels):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    ideal_rels = [1] * min(n_gold, len(relevance)) + [0] * max(0, len(relevance) - n_gold)
    idcg = dcg(ideal_rels)
    ndcg = dcg(relevance) / idcg if idcg > 0 else 0.0

    return {"recall": recall, "ndcg": ndcg, "n_gold": n_gold}


def score_retrieval_run(run_path):
    run = load_run(run_path)
    golden = load_golden_set(GOLDEN_SET_PATH)
    extracted_texts = get_extracted_texts(run["pipeline"])

    per_question, unresolved = [], []
    for item in run["results"]:
        golden_item = golden[item["id"]]
        gold_spans = []
        for ev in golden_item["gold_evidence"]:
            span, note = resolve_offsets(ev["quote"], ev["doc_id"], extracted_texts)
            if span is None:
                unresolved.append({"id": item["id"], "doc_id": ev["doc_id"], "reason": note})
                continue
            if note:
                print(f"  [{item['id']}] {note} ({ev['doc_id']})")
            gold_spans.append((span[0], span[1], ev["doc_id"]))

        if not gold_spans:
            unresolved.append({"id": item["id"], "doc_id": "ALL", "reason": "no gold spans resolved"})
            continue

        scores = score_retrieval_question(gold_spans, item["retrieved"])
        per_question.append({"id": item["id"], "category": item["category"], **scores})

    return per_question, unresolved


def summarize_retrieval(per_question):
    by_category = {}
    for q in per_question:
        by_category.setdefault(q["category"], []).append(q)
    summary = {}
    for cat, qs in by_category.items():
        summary[cat] = {
            "n": len(qs),
            "recall_mean": sum(q["recall"] for q in qs) / len(qs),
            "ndcg_mean": sum(q["ndcg"] for q in qs) / len(qs),
        }
    summary["overall"] = {
        "n": len(per_question),
        "recall_mean": sum(q["recall"] for q in per_question) / len(per_question),
        "ndcg_mean": sum(q["ndcg"] for q in per_question) / len(per_question),
    }
    return summary


# ---------------------------------------------------------------- LLM judge

def get_judge():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=JUDGE_MODEL, temperature=0)


def _ask_judge_json(judge, prompt):
    """Send prompt, parse JSON out of the response. Returns dict or None on failure."""
    response = judge.invoke(prompt)
    text = response.content.strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  [judge] failed to parse JSON: {text[:200]}")
        return None


FAITHFULNESS_PROMPT = """You are scoring the faithfulness of a generated answer against retrieved context.

Question: {question}

Retrieved context:
{context}

Generated answer: {answer}

Step 1: break the generated answer into a list of individual factual claims.
Step 2: for each claim, decide if it is directly supported by the retrieved context.

Respond ONLY with JSON, no other text:
{{"claims": [{{"claim": "...", "supported": true/false}}, ...]}}
"""

RELEVANCY_PROMPT = """You are scoring how well a generated answer addresses the question asked.

Question: {question}
Generated answer: {answer}

Score 0.0 to 1.0: does the answer address what was actually asked, or does it drift
off-topic / answer a different question?

Respond ONLY with JSON, no other text:
{{"relevancy_score": 0.0, "reasoning": "..."}}
"""

CORRECTNESS_PROMPT = """You are scoring a generated answer against a ground-truth answer.

Question: {question}
Ground truth answer: {ground_truth}
Generated answer: {answer}

Score 0.0 to 1.0: does the generated answer match the ground truth, including any
important facts the ground truth contains? Partial credit for missing details.

Respond ONLY with JSON, no other text:
{{"correctness_score": 0.0, "reasoning": "..."}}
"""

ABSTENTION_PROMPT = """Does the following answer abstain (say it doesn't know / the information
isn't available / similar), or does it give a confident, substantive answer?

Answer: {answer}

Respond ONLY with JSON, no other text:
{{"abstained": true/false}}
"""


def score_faithfulness(judge, question, answer, retrieved):
    context = "\n\n".join(f"[{r['doc_id']}]\n{r['text']}" for r in retrieved)
    result = _ask_judge_json(judge, FAITHFULNESS_PROMPT.format(
        question=question, context=context, answer=answer
    ))
    if not result or not result.get("claims"):
        return None
    claims = result["claims"]
    supported = sum(1 for c in claims if c.get("supported"))
    return {"score": supported / len(claims), "claims": claims}


def score_relevancy(judge, question, answer):
    result = _ask_judge_json(judge, RELEVANCY_PROMPT.format(question=question, answer=answer))
    if not result:
        return None
    return {"score": float(result.get("relevancy_score", 0.0)), "reasoning": result.get("reasoning", "")}


def score_correctness(judge, question, ground_truth, answer):
    result = _ask_judge_json(judge, CORRECTNESS_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer
    ))
    if not result:
        return None
    return {"score": float(result.get("correctness_score", 0.0)), "reasoning": result.get("reasoning", "")}


def score_abstention(judge, answer):
    result = _ask_judge_json(judge, ABSTENTION_PROMPT.format(answer=answer))
    if not result:
        return None
    return bool(result.get("abstained", False))


def score_generation_run(run_path):
    run = load_run(run_path)
    golden = load_golden_set(GOLDEN_SET_PATH)
    judge = get_judge()

    per_question = []
    for i, item in enumerate(run["results"]):
        golden_item = golden[item["id"]]
        print(f"  [{i + 1}/{len(run['results'])}] judging ({item['category']}) {item['id']}")

        faithfulness = score_faithfulness(judge, item["question"], item["answer"], item["retrieved"])
        relevancy = score_relevancy(judge, item["question"], item["answer"])
        correctness = score_correctness(judge, item["question"], golden_item["ground_truth_answer"], item["answer"])
        abstained = score_abstention(judge, item["answer"])

        per_question.append({
            "id": item["id"],
            "category": item["category"],
            "faithfulness": faithfulness["score"] if faithfulness else None,
            "relevancy": relevancy["score"] if relevancy else None,
            "correctness": correctness["score"] if correctness else None,
            "abstained": abstained,
        })

    return per_question


def summarize_generation(per_question):
    def mean(key, qs):
        vals = [q[key] for q in qs if q[key] is not None]
        return sum(vals) / len(vals) if vals else None

    by_category = {}
    for q in per_question:
        by_category.setdefault(q["category"], []).append(q)

    summary = {}
    for cat, qs in by_category.items():
        summary[cat] = {
            "n": len(qs),
            "faithfulness_mean": mean("faithfulness", qs),
            "relevancy_mean": mean("relevancy", qs),
            "correctness_mean": mean("correctness", qs),
        }

    non_oos = [q for q in per_question if q["category"] != "out_of_scope"]
    oos = [q for q in per_question if q["category"] == "out_of_scope"]

    false_answer_rate = None
    if oos:
        false_answers = sum(1 for q in oos if q["abstained"] is False)
        false_answer_rate = false_answers / len(oos)

    over_abstention_rate = None
    if non_oos:
        over_abstentions = sum(1 for q in non_oos if q["abstained"] is True)
        over_abstention_rate = over_abstentions / len(non_oos)

    summary["overall"] = {
        "n": len(per_question),
        "faithfulness_mean": mean("faithfulness", per_question),
        "relevancy_mean": mean("relevancy", per_question),
        "correctness_mean": mean("correctness", per_question),
        "false_answer_rate": false_answer_rate,
        "over_abstention_rate": over_abstention_rate,
    }
    return summary


# ---------------------------------------------------------------- reporting

def fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def append_results_md(pipeline, retrieval_summary, generation_summary):
    import datetime
    os.makedirs(os.path.dirname(RESULTS_MD_PATH), exist_ok=True)
    lines = []
    lines.append(f"\n## {pipeline} -- {datetime.datetime.utcnow().isoformat()}Z\n")

    if retrieval_summary:
        lines.append("### Retrieval\n")
        lines.append("| Category | n | Recall | nDCG |")
        lines.append("|---|---|---|---|")
        for cat, s in retrieval_summary.items():
            label = "**Overall**" if cat == "overall" else cat
            lines.append(f"| {label} | {s['n']} | {fmt(s['recall_mean'])} | {fmt(s['ndcg_mean'])} |")
        lines.append("")

    if generation_summary:
        lines.append("### Generation\n")
        lines.append("| Category | n | Faithfulness | Relevancy | Correctness |")
        lines.append("|---|---|---|---|---|")
        for cat, s in generation_summary.items():
            if cat == "overall":
                continue
            label = cat
            lines.append(f"| {label} | {s['n']} | {fmt(s['faithfulness_mean'])} | {fmt(s['relevancy_mean'])} | {fmt(s['correctness_mean'])} |")
        overall = generation_summary["overall"]
        lines.append(f"| **Overall** | {overall['n']} | {fmt(overall['faithfulness_mean'])} | {fmt(overall['relevancy_mean'])} | {fmt(overall['correctness_mean'])} |")
        lines.append("")
        lines.append("### Abstention\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| False-answer rate (out_of_scope) | {fmt(overall['false_answer_rate'], pct=True)} |")
        lines.append(f"| Over-abstention rate (answerable) | {fmt(overall['over_abstention_rate'], pct=True)} |")
        lines.append("")

    with open(RESULTS_MD_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nAppended results to {RESULTS_MD_PATH}")


def score_abstention_run(run_path):
    """Score abstention on out_of_scope questions only (false-answer rate)."""
    run = load_run(run_path)
    judge = get_judge()

    oos_items = [item for item in run["results"] if item["category"] == "out_of_scope"]
    print(f"Scoring abstention on {len(oos_items)} out_of_scope questions")

    per_question = []
    for i, item in enumerate(oos_items):
        print(f"  [{i + 1}/{len(oos_items)}] {item['id']}")
        abstained = score_abstention(judge, item["answer"])
        per_question.append({"id": item["id"], "abstained": abstained})

    resolved = [q for q in per_question if q["abstained"] is not None]
    false_answers = sum(1 for q in resolved if q["abstained"] is False)
    false_answer_rate = false_answers / len(resolved) if resolved else None

    return per_question, false_answer_rate


def append_abstention_results_md(pipeline, per_question, false_answer_rate):
    import datetime
    os.makedirs(os.path.dirname(RESULTS_MD_PATH), exist_ok=True)
    lines = [f"\n## {pipeline} -- abstention -- {datetime.datetime.utcnow().isoformat()}Z\n"]
    lines.append(f"Judge model: {JUDGE_MODEL}\n")
    lines.append("| Question ID | Abstained |")
    lines.append("|---|---|")
    for q in per_question:
        lines.append(f"| {q['id']} | {q['abstained']} |")
    lines.append("")
    lines.append(f"**False-answer rate: {fmt(false_answer_rate, pct=True)}** ({sum(1 for q in per_question if q['abstained'] is False)}/{len(per_question)} answered instead of abstaining)")
    lines.append("")
    with open(RESULTS_MD_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAppended abstention results to {RESULTS_MD_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", help="Path to a retrieval run JSON file")
    parser.add_argument("--generation-run", help="Path to a generation run JSON file")
    parser.add_argument(
        "--abstention-only", action="store_true",
        help="With --generation-run, score ONLY abstention on out_of_scope questions"
    )
    args = parser.parse_args()

    if not args.retrieval_run and not args.generation_run:
        parser.error("Provide --retrieval-run and/or --generation-run")

    if args.abstention_only:
        if not args.generation_run:
            parser.error("--abstention-only requires --generation-run")
        pipeline = load_run(args.generation_run)["pipeline"]
        per_question, false_answer_rate = score_abstention_run(args.generation_run)
        print(f"\nFalse-answer rate: {fmt(false_answer_rate, pct=True)}")
        append_abstention_results_md(pipeline, per_question, false_answer_rate)
        sys.exit(0)

    pipeline = None
    retrieval_summary = None
    generation_summary = None

    if args.retrieval_run:
        per_question, unresolved = score_retrieval_run(args.retrieval_run)
        pipeline = load_run(args.retrieval_run)["pipeline"]
        if unresolved:
            print(f"\n{len(unresolved)} gold spans could not be resolved:")
            for u in unresolved:
                print(f"  {u['id']} / {u['doc_id']}: {u['reason']}")
        retrieval_summary = summarize_retrieval(per_question)
        print("\n--- Retrieval scores ---")
        for cat, s in retrieval_summary.items():
            print(f"{cat:28s} n={s['n']:2d}  recall={fmt(s['recall_mean'])}  ndcg={fmt(s['ndcg_mean'])}")

    if args.generation_run:
        per_question = score_generation_run(args.generation_run)
        pipeline = load_run(args.generation_run)["pipeline"]
        generation_summary = summarize_generation(per_question)
        print("\n--- Generation scores ---")
        for cat, s in generation_summary.items():
            if cat == "overall":
                continue
            print(f"{cat:28s} n={s['n']:2d}  faithfulness={fmt(s['faithfulness_mean'])}  relevancy={fmt(s['relevancy_mean'])}  correctness={fmt(s['correctness_mean'])}")
        overall = generation_summary["overall"]
        print(f"{'overall':28s} n={overall['n']:2d}  faithfulness={fmt(overall['faithfulness_mean'])}  relevancy={fmt(overall['relevancy_mean'])}  correctness={fmt(overall['correctness_mean'])}")
        print(f"false_answer_rate={fmt(overall['false_answer_rate'], pct=True)}  over_abstention_rate={fmt(overall['over_abstention_rate'], pct=True)}")

    append_results_md(pipeline, retrieval_summary, generation_summary)