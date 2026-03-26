# How This Code Works

This repo is a step-by-step build of a corrective RAG pipeline using LangGraph, FAISS, OpenAI embeddings, and OpenAI chat models.

The notebooks evolve like this:

- `1_basic_rag.ipynb`: simple retrieval + answer generation
- `2_retrieval_refinement.ipynb`: adds sentence-level refinement
- `3_retrieval_evaluator.ipynb`: adds scoring of retrieved chunks
- `4_web_search_refinement.ipynb`: adds web fallback for incorrect retrieval
- `5_query_rewrite.ipynb`: improves web fallback with query rewriting
- `6_ambiguous.ipynb`: handles ambiguous retrieval by combining internal + web knowledge

## Core Flow

Across the notebooks, the pipeline has these main stages:

1. Load PDFs
2. Split them into chunks
3. Convert chunks into embeddings
4. Store them in FAISS
5. Retrieve top matching chunks for a question
6. Evaluate whether those chunks are good enough
7. Refine the useful knowledge
8. Generate the final answer

## Data Loading And Indexing

The notebooks load three PDF books from `documents/` using `PyPDFLoader`.

Then they split the text into overlapping chunks:

- `chunk_size = 900`
- `chunk_overlap = 150`

These chunks are embedded using:

- `OpenAIEmbeddings(model="text-embedding-3-large")`

The embeddings are stored in FAISS, and retrieval is done with:

- similarity search
- top `k = 4` chunks

So for each user question, the system first retrieves 4 chunks from the vector store.

## Notebook 2: Retrieval Refinement

This notebook improves basic RAG by cleaning retrieved knowledge before answering.

Flow:

1. Retrieve top 4 chunks
2. Join them into one context
3. Split that context into sentences
4. Ask an LLM judge whether each sentence directly helps answer the question
5. Keep only the useful sentences
6. Join those kept sentences into `refined_context`
7. Generate the answer only from `refined_context`

Key idea:

- retrieval may bring noisy chunks
- instead of trusting the whole chunk, refine at sentence level

LangGraph flow:

- `retrieve -> refine -> generate`

## Notebook 3: Retrieval Evaluator

This notebook adds a scoring step before refinement.

Each retrieved chunk is scored by the LLM with:

- `score` in `[0.0, 1.0]`
- `reason`

Two thresholds are used:

- `UPPER_TH = 0.7`
- `LOWER_TH = 0.3`

### Meaning of the thresholds

- If at least one chunk has `score > 0.7`, retrieval is `CORRECT`
- If all chunks have `score < 0.3`, retrieval is `INCORRECT`
- Otherwise retrieval is `AMBIGUOUS`

### What are `good_docs`?

`good_docs` is the list of retrieved chunks with:

- `score > 0.3`

So `good_docs` means:

- not necessarily perfect chunks
- but useful enough to keep

Important distinction:

- `good_docs` criterion: `score > 0.3`
- `CORRECT` criterion: at least one chunk `> 0.7`

So a chunk can be in `good_docs` without being strong enough to make the verdict `CORRECT`.

### Example

If chunk scores are:

- `0.7`
- `0.6`
- `0.5`
- `0.3`

Then:

- verdict is `AMBIGUOUS`
- `good_docs` contains the chunks with `0.7`, `0.6`, `0.5`
- the chunk with `0.3` is excluded because the code uses `>` not `>=`

## Notebook 4: Web Search Refinement

This notebook adds a real fallback for the `INCORRECT` case.

If internal retrieval fails:

1. Perform Tavily web search using the original question
2. Convert search results into `Document` objects
3. Refine those web docs at sentence level
4. Generate the answer from refined web context

Routing:

- `CORRECT -> refine internal docs`
- `INCORRECT -> web_search -> refine web docs -> generate`
- `AMBIGUOUS -> stop with ambiguity message`

Key idea:

- if local retrieval is clearly bad, fall back to the web

## Notebook 5: Query Rewrite

This notebook improves web search quality.

Instead of sending the raw question to Tavily, it first rewrites the question into a short search-style query.

Example behavior:

- compress the question into keywords
- add recency hints like `last 30 days` when needed

Routing:

- `CORRECT -> refine internal docs`
- `INCORRECT -> rewrite_query -> web_search -> refine -> generate`
- `AMBIGUOUS -> stop`

Key idea:

- web search works better with optimized search queries than with raw natural-language questions

## Notebook 6: Ambiguous Handling

This is the most complete notebook in the repo.

Now `AMBIGUOUS` does not stop the flow. Instead:

1. Rewrite the question for web search
2. Fetch web results
3. Combine internal `good_docs` and `web_docs`
4. Refine that combined context
5. Generate the final answer

Routing:

- `CORRECT -> refine using internal good_docs only`
- `INCORRECT -> rewrite_query -> web_search -> refine using web_docs only`
- `AMBIGUOUS -> rewrite_query -> web_search -> refine using good_docs + web_docs`

This matches the main corrective idea:

- trust internal docs when retrieval is good
- use web docs when retrieval is poor
- combine both when retrieval is mixed

## Important Clarifications

### Are web docs scored again?

No. In the current notebooks, the web results are not re-scored using the same `DocEvalScore` evaluator.

What happens instead:

- Tavily returns up to 5 results
- all returned results are stored in `web_docs`
- those web docs go directly into sentence-level refinement

So weak web documents are not filtered by chunk score first. They are filtered later at the sentence level.

### Is there a loop?

No. The graph is single-pass.

It does not:

- score web docs again
- repeat web search
- keep looping until confidence improves

It always ends after:

- retrieve
- evaluate
- maybe rewrite query
- maybe web search
- refine
- generate

### Why this is CRAG-inspired, not exact CRAG reproduction

The paper uses a trained `T5-large` evaluator.

This repo uses OpenAI models as judges for:

- chunk scoring
- sentence filtering
- query rewriting

So the pipeline captures the paper's idea, but not the exact paper model setup.

## One-Line Summary

This code starts from basic RAG, then gradually adds retrieval scoring, sentence-level refinement, web fallback, query rewriting, and ambiguous-case correction to build a CRAG-style pipeline.
