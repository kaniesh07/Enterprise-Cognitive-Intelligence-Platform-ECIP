"""
agents/state.py — Shared LangGraph state with all gap fixes applied.
"""
from __future__ import annotations
from typing import Any, Optional, TypedDict
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # avoid importing the retrieval stack just for type hints
    pass

class AgentState(TypedDict, total=False):
    # Input
    original_query:       str
    conversation_history: list[dict]
    user_id:              str           # NEW: for preference lookup
    user_preferences:     Any           # NEW: UserPreferences object
    prompt_version:       str           # NEW: track prompt version for observability

    # Intent agent
    intent:          str
    expanded_queries: list[str]
    language:        str

    # Retrieval agent
    retrieved_chunks:   list  # list[RetrievedChunk]
    retrieval_metrics:  dict

    # Multimodal agent
    multimodal_context: list[dict]

    # Reasoning agent
    reasoning_steps: list[str]
    draft_answer:    str

    # Verifier agent
    faithfulness_score:  float
    hallucination_flags: list[str]

    # Final output
    final_answer: str
    citations:    list[str]
    confidence:   float
    latency_ms:   dict
    token_usage:  dict    # NEW: prompt/completion token counts per agent
    error:        Optional[str]
    mode:         str     # chat | grounded | instant
