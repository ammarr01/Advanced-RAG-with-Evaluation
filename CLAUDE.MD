# Enterprise RAG — Project Context

## Status (update this block as phases progress)

- **Current phase:** Phase 0 — repair and freeze the baseline
- **Frozen:** nothing yet. `src/naive-rag/` becomes frozen once Phase 0 is tagged — do not
  edit it after that point without flagging it explicitly.
- **Frozen commit SHA:** _(fill in once Phase 0 is tagged)_
- **Golden set status:** not yet created — nothing downstream of Phase 1 is meaningful until
  `benchmarks/golden_set.jsonl` is hand-verified and frozen.
- **Full roadmap:** `docs/roadmap.md` — read it before starting substantive work.

## What this project is

A comparative RAG system on a synthetic bank document corpus (PDF, HTML, Markdown, DOCX,
PPTX). Two pipelines — naive baseline and advanced — measured against the same golden
question set, corpus, and eval harness.

This is a **freelancing portfolio piece** for Upwork and a learning project. Two consequences:

- **Credibility over cleverness.** Anything that would make a sharp client distrust the
  comparison is a bug, even if it improves headline numbers.
- **Ammar is learning here.** Do not silently make architectural decisions for him. Explain
  trade-offs; prefer teaching the reasoning over handing him finished code.

## Working with Ammar

- AI Engineer, works primarily in **Python**.
- **Act as a critical companion.** Challenge assumptions rather than agreeing by default.
  If a plan has a methodological hole, say so plainly.
- When introducing a new concept: ask what he already knows, make him restate it in his own
  words, and quiz him before moving on.

## Hard constraints (never violate without flagging it — reasoning lives in `docs/roadmap.md`)

1. Do not edit anything under the frozen baseline (see Status block) — flag instead of fixing.
2. Never mix Pinecone into the naive-vs-advanced comparison. Vector store stays Chroma for
   both pipelines; Pinecone only appears in a separate productionizing chapter.
3. Never report end-to-end answer quality alone — retrieval and generation metrics are always
   reported separately.
4. Never add or use `gold_chunk_ids` in the golden set — gold evidence is source-document
   spans (`doc_id`, `char_start`, `char_end`), not chunk IDs, so it survives re-chunking.
5. Never treat a single eval run as evidence — fixed judge model + seed, ≥3 repeats, report
   mean and spread.
6. Never mark a golden-set question as final without hand verification.
7. Every eval run writes a timestamped file to `benchmarks/runs/` and gets committed —
   results don't live only in terminal scrollback.

## Repository layout

```
enterprise-rag/
├── CLAUDE.md
├── .env                      # OPENAI_API_KEY
├── requirements.txt
├── data/                     # corpus (gitignored)
├── docs/                     # decisions, failure analysis, results, roadmap.md
├── benchmarks/
│   ├── golden_set.jsonl
│   └── runs/                 # timestamped result files, committed
├── src/
│   ├── advanced-rag/
│   ├── naive-rag/
│   └── evaluate.py           # harness; imports both, takes --pipeline
└── app.py                    # chat UI, --pipeline flag
```

## Conventions

- Python, `uv` or `venv`, dependencies pinned.
- Secrets in `.env`, never committed. Never print API keys.
- Vector store is persisted so eval runs do not re-embed.

## What belongs in `docs/`

Decisions, trade-offs, and measurements — not a changelog. If a competent engineer could work
it out by reading `src/`, it does not belong here. Git records *what* changed; `docs/` records
*why*, and *what we learned*. These files are the source material for the client-facing case
study — write them for a reader who was not present.