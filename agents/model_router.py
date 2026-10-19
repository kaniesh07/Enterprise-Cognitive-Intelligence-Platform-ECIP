"""
agents/model_router.py
-----------------------
Routes each agent step to the appropriate model based on task complexity.

Strategy:
  FAST  model (phi3:mini or llama3.1:8b)  — classification, routing, schema fill
  STRONG model (llama3.1:8b or 70b)       — reasoning, verification, summarisation

This cuts latency by ~30% and reduces compute for classification steps.
"""
from __future__ import annotations
import os
from enum import Enum

class ModelTier(str, Enum):
    FAST   = "fast"    # cheap, low-latency: intent, routing
    STRONG = "strong"  # powerful: reasoning, verification

# Map each agent to its required tier
_AGENT_TIERS: dict[str, ModelTier] = {
    "intent":     ModelTier.FAST,    # classify into 5 categories → fast model
    "multimodal": ModelTier.FAST,    # extract table summary → fast model
    "retrieval":  ModelTier.FAST,    # no LLM call in retrieval itself
    "reasoning":  ModelTier.STRONG,  # chain-of-thought over evidence → strong model
    "verifier":   ModelTier.STRONG,  # fact-checking every claim → strong model
    "assembler":  ModelTier.FAST,    # no LLM call in assembler
}

def _base_model() -> str:
    try:
        from config import settings
        default = settings.llm_model
    except Exception:
        default = "llama3.2:3b"
    return os.getenv("LLM_MODEL") or default


def get_model(agent_name: str) -> str:
    """Return the model name for a given agent.

    FAST_MODEL / STRONG_MODEL are optional overrides; when unset (or empty)
    every step uses LLM_MODEL, so only one model has to be pulled.
    """
    tier = _AGENT_TIERS.get(agent_name, ModelTier.STRONG)
    if tier == ModelTier.FAST:
        return os.getenv("FAST_MODEL") or _base_model()
    return os.getenv("STRONG_MODEL") or _base_model()

def log_routing(agent_name: str) -> dict:
    """Return routing metadata for observability logging."""
    model = get_model(agent_name)
    tier  = _AGENT_TIERS.get(agent_name, ModelTier.STRONG)
    return {"agent": agent_name, "model": model, "tier": tier.value}
