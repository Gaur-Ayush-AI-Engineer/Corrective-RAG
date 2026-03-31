"""
RAGAS evaluation for Corrective-RAG (CRAG) variants.

Variants:
  basic   – notebook 1 logic: retrieve top-4, no scoring, answer directly
  correct – notebook 5 logic: score chunks, route CORRECT/INCORRECT/AMBIGUOUS,
             CORRECT→refine(internal), INCORRECT→rewrite→web→refine, AMBIGUOUS→message
  full    – notebook 6 logic: CORRECT→refine(internal),
             INCORRECT/AMBIGUOUS→rewrite→web→refine (AMBIGUOUS uses internal+web combined)

Run: python evaluate_crag.py
Outputs:
  eval_results_crag_basic.json
  eval_results_crag_correct.json
  eval_results_crag_full.json
"""

import os
import json
import re
import sys
from typing import List, Tuple

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
UPPER_TH = 0.7
LOWER_TH = 0.3
TOP_K = 4
TAVILY_SEARCH_RESULTS = 3

PDF_PATH            = "./eval_data/attention_is_all_you_need.pdf"
GOLDEN_DATASET_PATH = "./eval_data/golden_dataset.json"

# ── Validate paths ────────────────────────────────────────────────────────────
if not os.path.exists(PDF_PATH):
    print(f"❌ PDF not found at {PDF_PATH}")
    sys.exit(1)

if not os.path.exists(GOLDEN_DATASET_PATH):
    print(f"❌ Golden dataset not found at {GOLDEN_DATASET_PATH}")
    sys.exit(1)

# ── Check optional Tavily ─────────────────────────────────────────────────────
TAVILY_AVAILABLE = bool(os.getenv("TAVILY_API_KEY"))
if not TAVILY_AVAILABLE:
    print("⚠️  TAVILY_API_KEY not set — web search steps will be skipped.")
    print("   INCORRECT verdicts will return a fallback message.")
    print("   AMBIGUOUS in 'full' variant will use internal docs only.\n")

# ── Imports ───────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ── Load golden dataset ───────────────────────────────────────────────────────
with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as _f:
    GOLDEN_DATASET = json.load(_f)

# ── Build shared FAISS index ──────────────────────────────────────────────────
print("📄 Loading and indexing attention_is_all_you_need.pdf...")
_raw_docs = PyPDFLoader(PDF_PATH).load()
_chunks = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=150
).split_documents(_raw_docs)
for _d in _chunks:
    _d.page_content = _d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
_vector_store = FAISS.from_documents(_chunks, _embeddings)
retriever = _vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": TOP_K}
)
print(f"✅ Indexed {len(_chunks)} chunks.\n")

# ── Shared LLM ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Shared chains ─────────────────────────────────────────────────────────────

# Doc evaluator — from notebooks 3-6
class DocEvalScore(BaseModel):
    score: float
    reason: str

_doc_eval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict retrieval evaluator for RAG.\n"
        "You will be given ONE retrieved chunk and a question.\n"
        "Return a relevance score in [0.0, 1.0].\n"
        "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
        "- 0.0: chunk is irrelevant\n"
        "Be conservative with high scores.\n"
        "Also return a short reason.\n"
        "Output JSON only.",
    ),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])
_doc_eval_chain = _doc_eval_prompt | llm.with_structured_output(DocEvalScore)

# Sentence filter — from notebooks 3-6
class KeepOrDrop(BaseModel):
    keep: bool

_filter_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict relevance filter.\n"
        "Return keep=true only if the sentence directly helps answer the question.\n"
        "Use ONLY the sentence. Output JSON only.",
    ),
    ("human", "Question: {question}\n\nSentence:\n{sentence}"),
])
_filter_chain = _filter_prompt | llm.with_structured_output(KeepOrDrop)

# Query rewriter — from notebooks 5-6
class WebQuery(BaseModel):
    query: str

_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user question into a web search query composed of keywords.\n"
        "Rules:\n"
        "- Keep it short (6–14 words).\n"
        "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
        "- Do NOT answer the question.\n"
        "- Return JSON with a single key: query",
    ),
    ("human", "Question: {question}"),
])
_rewrite_chain = _rewrite_prompt | llm.with_structured_output(WebQuery)

# ── Answer prompts (exact templates from notebooks) ───────────────────────────

# Notebook 1 — basic
_basic_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the context. If not in context, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])

# Notebooks 3-5 — correct variant
_refined_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
        "If the context is empty or insufficient, say: 'I don't know.'",
    ),
    ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
])

# Notebook 6 — full variant (uses {context} variable name, value is refined_context)
_full_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
        "If the context is empty or insufficient, say: 'I don't know.'",
    ),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])

# ── Internal helpers ──────────────────────────────────────────────────────────

def _decompose_to_sentences(text: str) -> List[str]:
    """From notebooks 3-6."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _run_refine(question: str, docs: List[Document]) -> str:
    """Decompose docs → filter sentences → recompose. From notebooks 3-6."""
    context = "\n\n".join(d.page_content for d in docs).strip()
    strips = _decompose_to_sentences(context)
    kept = [s for s in strips if _filter_chain.invoke({"question": question, "sentence": s}).keep]
    return "\n".join(kept).strip()


def _eval_docs(
    question: str, docs: List[Document]
) -> Tuple[List[float], List[Document], str, str]:
    """Score each doc, return (scores, good_docs, verdict, reason). From notebooks 3-6."""
    scores: List[float] = []
    good_docs: List[Document] = []

    for d in docs:
        out = _doc_eval_chain.invoke({"question": question, "chunk": d.page_content})
        scores.append(out.score)
        if out.score > LOWER_TH:
            good_docs.append(d)

    if any(s > UPPER_TH for s in scores):
        return scores, good_docs, "CORRECT", f"At least one retrieved chunk scored > {UPPER_TH}."

    if scores and all(s < LOWER_TH for s in scores):
        return scores, [], "INCORRECT", f"All retrieved chunks scored < {LOWER_TH}. No chunk was sufficient."

    return (
        scores,
        good_docs,
        "AMBIGUOUS",
        f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. Mixed relevance signals.",
    )


def _web_search(question: str) -> List[Document]:
    """Query rewrite + Tavily search. Returns [] if Tavily unavailable. From notebooks 5-6."""
    if not TAVILY_AVAILABLE:
        return []

    from langchain_community.tools.tavily_search import TavilySearchResults

    web_query = _rewrite_chain.invoke({"question": question}).query
    tavily = TavilySearchResults(max_results=TAVILY_SEARCH_RESULTS)
    results = tavily.invoke({"query": web_query}) or []

    web_docs: List[Document] = []
    for r in results:
        title   = r.get("title", "")
        url     = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text    = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return web_docs


# ── Variant runners ───────────────────────────────────────────────────────────

def run_basic(question: str) -> dict:
    """
    Notebook 1 logic: retrieve top-4, no scoring, answer directly.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    out = (_basic_prompt | llm).invoke({"question": question, "context": context})
    return {
        "answer":   out.content,
        "contexts": [d.page_content for d in docs],
        "verdict":  "N/A",
    }


def run_correct(question: str) -> dict:
    """
    Notebook 3-5 logic (most complete = notebook 5):
      CORRECT   → refine(good_docs) → generate
      INCORRECT → rewrite → web_search → refine(web_docs) → generate
      AMBIGUOUS → return verdict message (notebook 3/4/5 behavior)
    """
    docs = retriever.invoke(question)
    _, good_docs, verdict, reason = _eval_docs(question, docs)

    if verdict == "CORRECT":
        refined = _run_refine(question, good_docs)
        out = (_refined_prompt | llm).invoke(
            {"question": question, "refined_context": refined}
        )
        return {
            "answer":   out.content,
            "contexts": [d.page_content for d in good_docs],
            "verdict":  verdict,
        }

    if verdict == "INCORRECT":
        web_docs = _web_search(question)
        if web_docs:
            refined = _run_refine(question, web_docs)
            out = (_refined_prompt | llm).invoke(
                {"question": question, "refined_context": refined}
            )
            return {
                "answer":   out.content,
                "contexts": [d.page_content for d in web_docs],
                "verdict":  verdict,
            }
        # Tavily unavailable
        return {
            "answer":   f"INCORRECT: {reason} (web search unavailable — TAVILY_API_KEY not set)",
            "contexts": [],
            "verdict":  verdict,
        }

    # AMBIGUOUS — notebooks 3/4/5 return a message, no answer generated
    return {
        "answer":   f"Ambiguous: {reason}",
        "contexts": [d.page_content for d in good_docs],
        "verdict":  verdict,
    }


def run_full(question: str) -> dict:
    """
    Notebook 6 logic:
      CORRECT            → refine(internal only) → generate
      INCORRECT/AMBIGUOUS → rewrite → web_search → refine → generate
        AMBIGUOUS refine uses internal + web combined
        INCORRECT refine uses web only
    """
    docs = retriever.invoke(question)
    _, good_docs, verdict, reason = _eval_docs(question, docs)

    if verdict == "CORRECT":
        refined = _run_refine(question, good_docs)
        out = (_full_prompt | llm).invoke({"question": question, "context": refined})
        return {
            "answer":   out.content,
            "contexts": [d.page_content for d in good_docs],
            "verdict":  verdict,
        }

    # INCORRECT or AMBIGUOUS → rewrite + web
    web_docs = _web_search(question)

    if verdict == "AMBIGUOUS":
        source_docs = good_docs + web_docs   # internal + web combined
    else:
        source_docs = web_docs               # web only

    if source_docs:
        refined = _run_refine(question, source_docs)
    else:
        refined = ""

    out = (_full_prompt | llm).invoke({"question": question, "context": refined})
    return {
        "answer":   out.content,
        "contexts": [d.page_content for d in source_docs],
        "verdict":  verdict,
    }


# ── RAGAS helpers (compatible with ragas 0.1.x and 0.2.x) ────────────────────

def _build_ragas_dataset(results):
    questions  = [r["question"]     for r in results]
    answers    = [r["answer"]       for r in results]
    contexts   = [r["contexts"]     for r in results]
    references = [r["ground_truth"] for r in results]

    # Try ragas 0.2.x first
    try:
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        samples = [
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=c,
                reference=g,
            )
            for q, a, c, g in zip(questions, answers, contexts, references)
        ]
        return EvaluationDataset(samples=samples), "v2"
    except ImportError:
        pass

    # Fall back to ragas 0.1.x
    from datasets import Dataset
    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": references,
    }), "v1"


def _get_metrics(api_version):
    from langchain_openai import ChatOpenAI as _ChatOpenAI, OpenAIEmbeddings as _OAIEmb
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(
        _ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=8800)
    )
    ragas_emb = LangchainEmbeddingsWrapper(
        _OAIEmb(model="text-embedding-3-small")
    )

    if api_version == "v2":
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        return [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ]
    else:
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        faithfulness.llm          = ragas_llm
        answer_relevancy.llm      = ragas_llm
        answer_relevancy.embeddings = ragas_emb
        context_precision.llm     = ragas_llm
        context_recall.llm        = ragas_llm
        return [faithfulness, answer_relevancy, context_precision, context_recall]


def _extract_per_question(eval_result, results):
    try:
        df   = eval_result.to_pandas()
        rows = df.to_dict(orient="records")
    except Exception:
        rows = [{} for _ in results]

    combined = []
    for i, r in enumerate(results):
        row = rows[i] if i < len(rows) else {}
        combined.append({
            "question":          r["question"],
            "ground_truth":      r["ground_truth"],
            "answer":            r["answer"],
            "contexts":          r["contexts"],
            "verdict":           r.get("verdict", "N/A"),
            "faithfulness":      float(row.get("faithfulness",      row.get("Faithfulness",      0) or 0)),
            "answer_relevancy":  float(row.get("answer_relevancy",  row.get("AnswerRelevancy",   0) or 0)),
            "context_precision": float(row.get("context_precision", row.get("ContextPrecision",  0) or 0)),
            "context_recall":    float(row.get("context_recall",    row.get("ContextRecall",     0) or 0)),
        })
    return combined


def _extract_scores(per_question):
    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = {}
    for k in keys:
        values = [r[k] for r in per_question if r.get(k) is not None]
        scores[k] = round(sum(values) / len(values), 4) if values else None
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

VARIANTS = [
    ("basic",   run_basic,   "eval_results_crag_basic.json"),
    ("correct", run_correct, "eval_results_crag_correct.json"),
    ("full",    run_full,    "eval_results_crag_full.json"),
]

# Detect RAGAS API version once
_, _api_version = _build_ragas_dataset([{
    "question": "x", "ground_truth": "x", "answer": "x", "contexts": ["x"]
}])
print(f"📊 RAGAS API version: {_api_version}\n")

from ragas import evaluate as _ragas_evaluate

all_scores: dict = {}

for variant_name, runner, outfile in VARIANTS:
    print(f"\n{'='*60}")
    print(f"  VARIANT: {variant_name.upper()}")
    print(f"{'='*60}")

    pipeline_results = []
    for i, item in enumerate(GOLDEN_DATASET, 1):
        print(f"  [{i:02d}/{len(GOLDEN_DATASET)}] {item['question'][:70]}...")
        result = runner(item["question"])
        pipeline_results.append({
            "question":     item["question"],
            "ground_truth": item["ground_truth"],
            "answer":       result["answer"],
            "contexts":     result["contexts"],
            "verdict":      result.get("verdict", "N/A"),
        })

    dataset, _ = _build_ragas_dataset(pipeline_results)
    metrics = _get_metrics(_api_version)
    print("⏳ Running RAGAS evaluation...")
    eval_result = _ragas_evaluate(dataset, metrics=metrics)

    per_question = _extract_per_question(eval_result, pipeline_results)
    scores       = _extract_scores(per_question)
    all_scores[variant_name] = scores

    output = {
        "variant":          variant_name,
        "aggregate_scores": scores,
        "per_question":     per_question,
    }
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"💾 Saved → {outfile}")
    print(f"   {scores}")


# ── Comparison Table 1: all 3 CRAG variants ───────────────────────────────────
METRICS  = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
VARIANTS_NAMES = ["basic", "correct", "full"]
COL_W = 12

_sep = "=" * (24 + COL_W * len(VARIANTS_NAMES))
print(f"\n\n{_sep}")
print("  CRAG VARIANTS COMPARISON")
print(_sep)
print(f"  {'Metric':<22}" + "".join(f"{v:>{COL_W}}" for v in VARIANTS_NAMES))
print("-" * (24 + COL_W * len(VARIANTS_NAMES)))
for m in METRICS:
    row = f"  {m:<22}"
    for v in VARIANTS_NAMES:
        val = all_scores[v].get(m)
        row += f"{val:>{COL_W}.4f}" if val is not None else f"{'N/A':>{COL_W}}"
    print(row)
print(_sep)


# ── Comparison Table 2: CRAG "correct" vs LangGraph "none" ───────────────────
LANGGRAPH_NONE = {
    "faithfulness":      0.9104,
    "answer_relevancy":  0.8628,
    "context_precision": 0.6667,
    "context_recall":    0.8000,
}

_sep2 = "=" * (24 + COL_W * 2)
print(f"\n{_sep2}")
print("  CRAG correct  vs  LangGraph none")
print(_sep2)
print(f"  {'Metric':<22}" + f"{'CRAG-correct':>{COL_W}}" + f"{'LG-none':>{COL_W}}")
print("-" * (24 + COL_W * 2))
for m in METRICS:
    crag_val = all_scores["correct"].get(m)
    lg_val   = LANGGRAPH_NONE[m]
    row  = f"  {m:<22}"
    row += f"{crag_val:>{COL_W}.4f}" if crag_val is not None else f"{'N/A':>{COL_W}}"
    row += f"{lg_val:>{COL_W}.4f}"
    print(row)
print(_sep2)

print("\n✅ Evaluation complete.\n")
