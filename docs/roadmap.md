## Roadmap (v2)

### Phase 0 — Repair and freeze the baseline
- Pin `text-embedding-3-small` (the unspecified default is the stale `ada-002`)
- Persist Chroma to disk so eval runs stop re-embedding
- Add an abstention instruction to the prompt ("if the context does not contain the answer,
  say you don't know")
- Carry `doc.metadata` through so answers cite their sources — this metadata is also what
  makes span-based gold evidence scoreable later (Phase 1)
- Tag the commit. `src/naive.py` is frozen from here.

### Phase 1 — Golden question set — **THE GATE**
- 50 questions in `benchmarks/golden_set.jsonl`, each with: `question`,
  `ground_truth_answer`, `category`, `gold_evidence`
- `gold_evidence` is a list of **source-document spans**, not chunk IDs:
  `{"doc_id": "...", "char_start": N, "char_end": N}`. Spans are anchored to the *original
  document text*, so they survive any future re-chunking (Phase 5 changes chunking
  strategy — chunk-ID-based gold labels would break and force full re-annotation).
- Scoring later uses **span overlap**, not ID equality — deterministic, no LLM judge needed.
- Category breakdown (weighted by what each metric needs to be statistically meaningful,
  not evenly split):

| Category | Count | Why this weight |
|---|---|---|
| out_of_scope | 10 | Abstention rate is a proportion — needs the largest n to be stable; #1 client-facing metric |
| multi_hop | 9 | Needs volume — this is where nDCG vs. recall vs. MRR actually diverge |
| table_numeric | 7 | Known bug target (PyMuPDF flattens tables) |
| acronym_exact_term | 7 | Known bug target (dense retrieval weak on rare exact tokens) |
| comparison | 6 | Distinct failure mode from multi-hop (near-duplicate retrieval, not missing docs) |
| ambiguous_underspecified | 6 | Tests query rewriting (a Phase 5 candidate upgrade) |
| simple_factoid | 5 | Sanity floor — low variance expected |

- Generation method: structured inventory pass over the corpus first (docs flagged by
  table/acronym/cross-doc-topic presence), then per-category candidate generation targeted
  at flagged docs, each candidate carrying an exact quoted span (not paraphrase) so
  `char_start/char_end` can be located programmatically.
- LLM drafts candidates, **hand-verify every one** — non-negotiable. An unverified golden
  set measures nothing.
- Nothing downstream is meaningful until this is frozen.

### Phase 2 — Eval harness (`src/evaluate.py`)
- Retrieval: **recall@k, nDCG** — deterministic, span-overlap based, no LLM judge.
  (MRR optional/secondary: it's blind on multi-hop questions since it only credits the
  first correct hit — nDCG subsumes what it measures and adds ranking-depth signal.)
- Generation: **faithfulness, answer relevancy, answer correctness** (RAGAS) — LLM-judged.
- Abstention: false-answer rate on `out_of_scope` (deterministic, classifier-based
  abstain/no-abstain check), plus over-abstention rate on answerable questions.
- Judge runs: fix judge model + seed, run each config ≥3×, report mean and spread.
  Single-run LLM-judge numbers are not evidence.
- Each run writes a timestamped JSON to `benchmarks/runs/`: git SHA, pipeline, config,
  golden set version, judge model, all metrics, per-question results. Committed.

### Phase 3 — Baseline run + failure taxonomy
- Run naive over the full golden set; record numbers per category
- For each failure, capture a **real transcript**: question → retrieved chunks → answer →
  diagnosis. Write to `docs/failure-modes.md`.
- The failures found here dictate which advanced techniques get built. Do not pick
  techniques first.

### Phase 4 — Chat UI (`app.py`)
- One UI, `--pipeline naive|advanced`
- Show retrieved chunks and citations, not just the answer
- Built now, not earlier: Phase 3 determines what needs to be visible

### Phase 5 — Advanced pipeline (`src/advanced.py`)
- Build upgrades that target observed failures. Likely candidates: table-aware parsing,
  structure-aware chunking with heading metadata, hybrid BM25 + dense, reranking, query
  rewriting, MMR for diversity.
- Same corpus, same vector store (Chroma stays constant — swapping to Pinecone here would
  confound retrieval-technique gains with infra changes), same golden set, same judge.
- Chunking strategy is allowed to differ between naive and advanced (it's an ablation
  variable) — this is exactly why gold evidence had to be span-based, not chunk-ID-based.
- **Re-run the harness after each upgrade** and log the delta in `docs/results.md`,
  including upgrades that made things worse. Only two pipelines ship, but the per-upgrade
  attribution is what lets a client ask "which of these would help my corpus?"

### Phase 6 — Packaging
- Naive vs. advanced architecture diagrams, side by side
- Headline benchmark table plus an honest latency/cost regression table — deferred from
  active build-out for now, but logged (token counts + timestamps) alongside eval runs
  from Phase 2 onward so it isn't a scramble to backfill later
- Video: same questions, both pipelines, retrieval visible
- README written for a client: problem, approach, results, trade-offs

### Separate chapter — Productionizing (post-Phase 6)
- Chroma vs. Pinecone comparison, evaluated on latency/cost/ops — not on answer quality.
  Kept out of the naive-vs-advanced comparison to avoid confounding retrieval technique
  with infrastructure choice.