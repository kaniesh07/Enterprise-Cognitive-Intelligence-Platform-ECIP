"""
agents/orchestrator.py — public entry points for running a query.

`run_query` is kept for backwards compatibility (CLI, evaluation, tools).
The heavy lifting lives in agents/pipeline.py.

The original LangGraph graph (intent → retrieval → multimodal → reasoning →
verifier → assembler) made up to three sequential LLM calls per question.
`build_graph()` now returns a tiny dependency-free graph with the same
`.invoke(state)` interface that runs the single-call pipeline.
"""
from __future__ import annotations

from typing import Optional

import structlog

from agents.pipeline import PROMPT_VERSION, finalize, generate, prepare, run_async  # noqa: F401
from agents.state import AgentState

logger = structlog.get_logger()


def _is_conversational(query: str) -> bool:
    from agents.pipeline import smalltalk_reply
    return smalltalk_reply(query) is not None


class _Graph:
    """Minimal stand-in for a compiled LangGraph graph."""

    def invoke(self, state: AgentState) -> dict:
        res = run_query(state.get("original_query", ""), state.get("conversation_history"),
                        state.get("user_id", "anonymous"))
        return {**state, **res}


def build_graph() -> _Graph:
    return _Graph()


def get_graph() -> _Graph:
    return _Graph()


def run_query(
    query: str,
    conversation_history: Optional[list[dict]] = None,
    user_id: str = "anonymous",
    mode: str = "auto",
    top_k: Optional[int] = None,
    doc_filter: Optional[str] = None,
) -> dict:
    """Run one query end-to-end (blocking). Never raises — errors are returned in `error`."""
    try:
        turn = prepare(query, conversation_history, user_id, mode, top_k, doc_filter)
        raw = generate(turn)
        result = finalize(turn, raw)
        logger.info("query_complete", mode=result["mode"], total_ms=result["latency_ms"].get("total"))
        return result
    except Exception as exc:
        logger.error("pipeline_error", query=query[:80], error=str(exc)[:300])
        return {
            "final_answer": f"Something went wrong: {exc}",
            "citations": [], "sources": [], "confidence": 0.0, "faithfulness_score": 0.0,
            "hallucination_flags": [], "reasoning_steps": [], "latency_ms": {},
            "retrieval_metrics": {}, "error": str(exc), "prompt_version": PROMPT_VERSION,
            "token_usage": {}, "retrieved_chunks": [], "mode": "error", "intent": "",
        }
