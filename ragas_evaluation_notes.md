# RAGAS And Evaluation Notes

This file is a one-time-read guide to remember how RAG evaluation works, what RAGAS does, and how human references and LLM judges fit together.

## What RAGAS Is

RAGAS is an evaluation framework for RAG systems.

It does not improve your model by itself.

It helps you answer:

- Is retrieval bringing the right context?
- Is retrieval missing important context?
- Is the answer grounded in the retrieved context?
- Is the answer actually answering the question?

So RAGAS is mainly for measuring where your pipeline is weak.

## The Four Main Ideas

Think of RAG as two jobs:

- retrieval job
- generation job

The four common evaluation ideas are:

- `Context Precision`
- `Context Recall`
- `Faithfulness`
- `Answer Relevancy`

## 1. Context Precision

Question:

- Among the chunks I retrieved, how much of it was actually useful?

If you retrieve:

- 10 chunks
- only 2 are relevant
- 8 are noise

then precision is low.

So precision is about retrieval cleanliness.

### Easy way to remember

- high precision = mostly useful chunks
- low precision = lots of irrelevant chunks

### One-line memory

`Context Precision = Did I retrieve useful context, or a lot of noise?`

## 2. Context Recall

Question:

- Did I retrieve enough information to answer the question?

If the answer needs 5 important facts, but retrieved chunks only contain 3, then recall is low.

So recall is about missing useful information.

### Easy way to remember

- high recall = the needed facts are present
- low recall = some needed facts were never retrieved

### One-line memory

`Context Recall = Did I miss anything important?`

## Precision vs Recall

This is the most important distinction:

- precision = too much irrelevant stuff
- recall = missing useful stuff

You do not think of them as a strict sequence.

They diagnose two different failure modes.

Examples:

- high recall, low precision: you brought the answer, but also lots of junk
- high precision, low recall: what you brought is relevant, but incomplete

### One-line memory

`Precision checks noise, recall checks missing information.`

## 3. Faithfulness

Question:

- Is the generated answer supported by the retrieved context?

If the retrieved chunk says:

- "Paris is the capital of France"

and the model answers:

- "Paris is the capital of France and has 10 million people"

then the second part is unfaithful if the context did not mention it.

Faithfulness is about hallucination relative to the context.

### Easy way to remember

- high faithfulness = answer stays inside the evidence
- low faithfulness = answer adds unsupported claims

### One-line memory

`Faithfulness = Did the answer stay loyal to the context?`

## 4. Answer Relevancy

Question:

- Does the answer actually answer the user's question?

An answer can be true but still not answer the question properly.

Example:

- Question: "Why is batch normalization useful?"
- Answer: "Batch normalization was introduced in 2015."

That answer may be factual, but it is not relevant to what was asked.

### One-line memory

`Answer Relevancy = Did the answer actually answer the question?`

## How Human References Fit In

If you created a golden dataset with:

- question
- reference answer

then you are using human-curated evaluation data.

This is the benchmark.

It acts like the exam answer key.

Your system outputs are compared against this benchmark during evaluation.

## How LLM-As-A-Judge Fits In

LLM-as-a-judge is another evaluation mechanism.

Instead of exact string matching, an LLM reads:

- the question
- retrieved context
- model answer
- and sometimes the human reference

Then it judges whether the answer is relevant, faithful, correct, or complete.

So:

- human reference = the target or benchmark
- LLM judge = the scorer for more nuanced evaluation

## If You Have A Golden Dataset

This is your current setup.

That means:

- you already have a strong evaluation foundation
- RAGAS can use your dataset to benchmark system behavior
- some metrics may use the reference answer directly
- some metrics may compare answer vs context using an LLM judge

Simple view:

- humans create the exam
- RAGAS defines what to grade
- the LLM helps grade it

## If You Do Not Have A Golden Dataset

Then evaluation is weaker and usually more judge-based.

In that setup, you often store:

- question
- retrieved context
- generated answer

and ask an LLM to score:

- correctness
- faithfulness
- relevancy
- completeness

This is still useful, especially early on, but it is less reliable than having a curated benchmark.

## How Your Golden Dataset Is Used

If you have:

- `question`
- `reference answer`

then RAGAS can use that reference answer for retrieval-side evaluation.

Simple idea:

- `Context Recall`: do retrieved chunks contain what is needed for the reference answer?
- `Context Precision`: are the retrieved chunks actually useful for answering toward the reference answer?
- `Faithfulness`: is the system answer supported by the retrieved chunks?
- `Answer Relevancy`: does the system answer address the question?

So the golden dataset is not there to generate answers.

It is there to evaluate your RAG system against a stable benchmark.

## Which Metrics Use Your Curated Dataset

If your dataset contains:

- `question`
- `reference answer`

then the `reference answer` is used differently across metrics.

### Quick Mapping

| Metric | Uses your curated `reference` answer? | Main comparison |
| --- | --- | --- |
| Context Recall | Yes | retrieved context vs reference answer |
| Context Precision | Yes in the reference-based version | retrieved context vs reference answer |
| Faithfulness | No, not mainly | model response vs retrieved context |
| Answer Relevancy | No | model response vs question |

### Simple interpretation

- `Context Recall`:
  asks whether the retrieved chunks contain the information needed for the reference answer

- `Context Precision`:
  asks whether the retrieved chunks are actually useful and relevant for answering toward the reference answer

- `Faithfulness`:
  asks whether the generated answer is supported by the retrieved context, not whether it matches your gold answer

- `Answer Relevancy`:
  asks whether the generated answer addresses the question, not whether it matches your gold answer

### One important note

`Context Precision` can exist in different variants in RAGAS.

So when you say "context precision does not need my answer", that is only true for no-reference variants.

In the reference-based version, your curated answer is used.

## Should You Use Only LLM Judge?

Only-LMM-judge evaluation is okay for:

- early prototypes
- quick comparisons
- rough ranking between system versions

But it is weaker for:

- niche domains
- high-stakes domains
- stable benchmarking over time

Best practice:

- start with LLM judge if you must
- build a small golden dataset as soon as possible
- use both together

## Why CRAG-Like Evaluation Still Makes Sense

A retrieval evaluator is useful because it tells you whether retrieved chunks are good enough to trust.

But the fallback action should depend on your domain.

For broad or recent topics:

- web search can help

For niche or closed domains:

- open web fallback may be the wrong choice
- better fallback is usually a trusted internal corpus, domain KB, or abstain path

So the core useful idea is:

- keep the evaluator
- choose a domain-safe fallback

## Final Memory Sheet

- `Context Precision` -> too much noise?
- `Context Recall` -> missing needed facts?
- `Faithfulness` -> answer supported by context?
- `Answer Relevancy` -> answer actually addresses the question?

- human reference -> benchmark
- LLM judge -> scorer
- RAGAS -> evaluation framework

## One-Line Summary

RAGAS helps you measure whether retrieval is clean and complete, and whether generation is grounded and relevant, using a mix of human references and LLM-based judging.
