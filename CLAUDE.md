# CLAUDE.md — Corrective RAG Project

## What this repo is

Notebook-based CRAG implementation. Each notebook adds one feature on top of the last.
No `src/` package — all logic lives in notebooks + one evaluation script.

## Notebooks (do not modify)

| Notebook | What it adds |
|----------|-------------|
| `1_basic_rag.ipynb` | Retrieve top-4 chunks → answer directly |
| `2_retrieval_refinement.ipynb` | Sentence-level filter on retrieved chunks |
| `3_retrieval_evaluator.ipynb` | LLM scores each chunk, routes CORRECT/INCORRECT/AMBIGUOUS |
| `4_web_search_refinement.ipynb` | INCORRECT → Tavily web search → refine → answer |
| `5_query_rewrite.ipynb` | INCORRECT → rewrite query first, then web search |
| `6_ambiguous.ipynb` | INCORRECT/AMBIGUOUS both go to web; AMBIGUOUS uses internal + web combined |

Shared config across notebooks: `UPPER_TH=0.7`, `LOWER_TH=0.3`, `k=4`, `chunk_size=900`, `chunk_overlap=150`.
Embeddings: `text-embedding-3-large`. LLM: `gpt-4o-mini`.

## What was just built — RAGAS Evaluation (evaluate_crag.py)

Added `evaluate_crag.py` in the repo root. It evaluates 3 CRAG variants against a 10-question golden dataset
(questions about "Attention Is All You Need").

### Three variants and their logic

- **basic** — notebook 1 logic: retrieve top-4, answer directly, no scoring
- **correct** — notebook 5 logic: score chunks → CORRECT→refine(internal), INCORRECT→rewrite→web→refine, AMBIGUOUS→returns "Ambiguous: ..." message (faithful to notebooks 3-5)
- **full** — notebook 6 logic: CORRECT→internal, INCORRECT/AMBIGUOUS both go to rewrite→web→refine; AMBIGUOUS uses internal+web combined

### Key decisions

1. **Index `eval_data/attention_is_all_you_need.pdf` not book1/2/3.** The golden dataset is about that paper. Using book1/2/3 would make every question route INCORRECT and collapse variant comparison.

2. **Contexts for RAGAS = source Documents, not filtered sentences.** The task says "full chunk strings, not truncated". The `refined_context` (post sentence-filter) would be fragments — bad for RAGAS context metrics.

3. **`correct` variant keeps AMBIGUOUS→message behavior** (faithful to notebooks 3-5). This tanks faithfulness slightly for ambiguous questions but accurately reflects what notebook 5 does.

4. **RAGAS dual API support.** Handles ragas 0.1.x (`datasets.Dataset`) and 0.2.x (`EvaluationDataset`) — same pattern as the reference LangGraph repo.

5. **Tavily is optional.** If `TAVILY_API_KEY` not set: web search returns `[]`, INCORRECT returns a fallback message, AMBIGUOUS in `full` uses internal docs only.

### Results (run 2026-03-31)

```
                    basic    correct    full
faithfulness        0.95     0.83       0.90
answer_relevancy    0.99     0.89       0.99
context_precision   0.77     0.90       0.90
context_recall      0.90     0.90       0.85
```

Key takeaway: `basic` wins on generation (no filtering risk), `correct`/`full` win on precision (noisy chunks removed). `correct` faithfulness dip is from AMBIGUOUS→message returning non-answers. `full` is most balanced.

CRAG correct vs LangGraph none: CRAG wins precision (+0.23) and recall (+0.10), LangGraph wins faithfulness (0.91 vs 0.83 — mostly the AMBIGUOUS penalty).

### Output files

- `eval_results_crag_basic.json`
- `eval_results_crag_correct.json`
- `eval_results_crag_full.json`

Each has `variant`, `aggregate_scores`, `per_question` keys. Per-question includes `verdict` field.

## File layout

```
/
├── 1_basic_rag.ipynb … 6_ambiguous.ipynb   ← do not modify
├── evaluate_crag.py                          ← RAGAS eval script
├── eval_data/
│   ├── attention_is_all_you_need.pdf         ← indexed by evaluate_crag.py
│   └── golden_dataset.json                   ← 10 Q&A pairs
├── documents/
│   ├── book1.pdf, book2.pdf, book3.pdf       ← used by notebooks only
│   └── readme.md
├── requirements.txt                          ← includes ragas
├── ragas_evaluation_notes.md                 ← user's notes on RAGAS concepts
└── indexing_notes.md
```

## Env vars needed

```env
OPENAI_API_KEY=required
TAVILY_API_KEY=optional (web search variants degrade gracefully without it)
```

## Current state

- [x] All 6 notebooks implemented
- [x] RAGAS evaluation script (`evaluate_crag.py`)
- [x] Golden dataset + attention paper in `eval_data/`
- [ ] No `src/` package structure
- [ ] No CI / automated tests

## Gotchas

- `correct` variant AMBIGUOUS returns a non-answer by design (matching notebook 3-5 behavior). If you want a real answer for AMBIGUOUS, use `full` logic instead.
- The notebooks index book1/2/3 but `evaluate_crag.py` indexes the attention paper — intentional, not a bug.
- Notebook 6's `generate` prompt uses `{context}` as template variable (not `{refined_context}` like notebooks 3-5). The script matches this exactly.
- `TavilySearchResults` from `langchain_community` triggers a deprecation warning in newer LangChain versions — matches notebook behavior, not a breakage.
