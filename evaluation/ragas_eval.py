"""
evaluation/ragas_eval.py — Full evaluation with all missing metrics added.

New metrics vs. original:
  - intent_accuracy       : predicted vs ground-truth intent
  - tool_call_success_rate: from ToolRegistry.get_metrics()
  - task_completion_rate  : answer != fallback/error message
  - token_efficiency      : total tokens / number of queries
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import structlog

logger = structlog.get_logger()

# ── Retrieval metrics ─────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    hits = sum(1 for cid in retrieved_ids[:k] if cid in set(relevant_ids))
    return hits / k if k else 0.0

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    hits = sum(1 for cid in retrieved_ids[:k] if cid in set(relevant_ids))
    return hits / len(relevant_ids) if relevant_ids else 0.0

def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    rel = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in rel:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    rel = set(relevant_ids)
    def dcg(ids):
        return sum(1 / math.log2(i + 2) for i, c in enumerate(ids[:k]) if c in rel)
    ideal = [c for c in retrieved_ids if c in rel] + [c for c in retrieved_ids if c not in rel]
    return dcg(retrieved_ids) / dcg(ideal) if dcg(ideal) > 0 else 0.0

# ── Intent accuracy ───────────────────────────────────────────────────────

def intent_accuracy(predicted: list[str], ground_truth: list[str]) -> float:
    """Fraction of queries where predicted intent matches ground truth."""
    if not ground_truth:
        return 0.0
    correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
    return correct / len(ground_truth)

# ── Task completion ───────────────────────────────────────────────────────

_FAILURE_PHRASES = [
    "an error occurred", "i was unable", "i cannot", "no relevant documents",
    "i don't know", "i do not have", "cannot be found", "not found in",
]

def is_task_completed(answer: str) -> bool:
    """Heuristic: answer is complete if it doesn't contain failure phrases."""
    lower = answer.lower()
    return not any(phrase in lower for phrase in _FAILURE_PHRASES)

# ── LLM faithfulness (without RAGAS) ─────────────────────────────────────

def _llm_faithfulness(question: str, answer: str, context: str) -> float:
    """LLM-judged faithfulness via the shared client (works for Ollama and OpenAI-compatible)."""
    try:
        from agents.llm_client import chat_json
        res = chat_json([{"role": "user", "content":
            "Rate how faithful the answer is to the context, 0.0-1.0. "
            'Return only JSON: {"faithfulness": 0.0}\n\n'
            f"Question: {question}\nAnswer: {answer}\nContext: {context[:1500]}"}])
        return float(res.get("faithfulness", 0.8))
    except Exception:
        return 0.8

# ── RAGAS wrapper ─────────────────────────────────────────────────────────

def _run_ragas(questions, answers, contexts, ground_truths) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
        ds = Dataset.from_dict({"question": questions, "answer": answers,
                                 "contexts": contexts, "ground_truth": ground_truths})
        return dict(evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall, context_precision]))
    except Exception as exc:
        logger.warning("ragas_unavailable", error=str(exc))
        return {}

# ── Main evaluation runner ────────────────────────────────────────────────

def run_ragas_evaluation(dataset_path: str, top_k: int = 5) -> dict:
    from agents.orchestrator import run_query

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    with open(path) as f:
        dataset = json.load(f)

    logger.info("eval_start", samples=len(dataset))
    t0 = time.perf_counter()

    questions, answers, contexts_list, ground_truths = [], [], [], []
    retrieval_scores: list[dict] = []
    predicted_intents: list[str] = []
    ground_truth_intents: list[str] = []
    task_completions: list[bool] = []
    doc_scores: list[dict] = []
    fact_recalls: list[float] = []
    routing: list[bool] = []

    for item in dataset:
        question     = item["question"]
        ground_truth = item.get("ground_truth", "")
        relevant_ids = item.get("relevant_chunk_ids", [])
        gt_intent    = item.get("intent")           # optional ground-truth intent

        result = run_query(question, item.get("history"), mode=item.get("mode", "auto"))
        answer = result["final_answer"]
        chunks = result.get("retrieved_chunks", [])
        retrieved_ids = [c["chunk_id"] for c in chunks]
        chunk_texts   = [c["snippet"] for c in chunks]

        questions.append(question)
        answers.append(answer)
        contexts_list.append(chunk_texts)
        ground_truths.append(ground_truth)
        predicted_intents.append(result.get("intent", ""))
        task_completions.append(is_task_completed(answer))

        if gt_intent:
            ground_truth_intents.append(gt_intent)

        # Document-level relevance + required facts (new dataset format)
        from evaluation.evaluate import keyword_recall, rank_metrics
        if item.get("relevant_docs"):
            doc_scores.append(rank_metrics([c.get("doc_title") for c in chunks],
                                           set(item["relevant_docs"]), top_k))
        kr = keyword_recall(answer, item.get("must_contain") or [])
        if kr is not None:
            fact_recalls.append(kr)
        if item.get("expected_mode"):
            routing.append(result.get("mode") == item["expected_mode"])

        if relevant_ids:
            retrieval_scores.append({
                "question":     question[:60],
                "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, top_k),
                "recall_at_k":    recall_at_k(retrieved_ids, relevant_ids, top_k),
                "mrr":            mrr(retrieved_ids, relevant_ids),
                "ndcg_at_k":      ndcg_at_k(retrieved_ids, relevant_ids, top_k),
            })

    ragas_metrics = _run_ragas(questions, answers, contexts_list, ground_truths)

    avg_retrieval = {}
    if retrieval_scores:
        for key in ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"]:
            avg_retrieval[key] = round(sum(s[key] for s in retrieval_scores) / len(retrieval_scores), 4)

    # Tool call success rates
    try:
        from tools.registry import get_registry
        tool_metrics = get_registry().get_metrics()
    except Exception:
        tool_metrics = {}

    # Token efficiency
    try:
        from agents.llm_client import get_total_tokens
        usage = get_total_tokens()
        token_efficiency = round(usage["total"] / len(dataset), 1) if dataset else 0
    except Exception:
        token_efficiency = 0

    results = {
        "num_samples":          len(dataset),
        "top_k":                top_k,
        "evaluation_time_s":    round(time.perf_counter() - t0, 2),
        "ragas_metrics":        ragas_metrics,
        "retrieval_metrics":    avg_retrieval,
        "per_query_retrieval":  retrieval_scores,
        # NEW metrics
        "intent_accuracy":      round(intent_accuracy(predicted_intents, ground_truth_intents), 4)
                                 if ground_truth_intents else None,
        "task_completion_rate": round(sum(task_completions) / len(task_completions), 4) if task_completions else 0,
        "tool_call_metrics":    tool_metrics,
        "token_efficiency":     {"tokens_per_query": token_efficiency},
        "routing_accuracy":     round(sum(routing) / len(routing), 4) if routing else None,
        "answer_fact_recall":   round(sum(fact_recalls) / len(fact_recalls), 4) if fact_recalls else None,
        "answer_accuracy":      round(sum(1 for f in fact_recalls if f == 1.0) / len(fact_recalls), 4)
                                 if fact_recalls else None,
        "doc_retrieval_metrics": {k: round(sum(d[k] for d in doc_scores) / len(doc_scores), 4)
                                  for k in doc_scores[0]} if doc_scores else {},
    }

    out = Path("./evaluation/results/latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("eval_complete", **{k: v for k, v in results.items() if not isinstance(v, (list, dict))})
    return results
