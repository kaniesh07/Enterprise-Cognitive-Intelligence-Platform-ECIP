"""
guardrails/pipeline.py
-----------------------
The GuardrailPipeline orchestrates all input and output guards in sequence.
Import and call this from api/main.py.

Input pipeline  (before query reaches agents):
  1. rate_limit
  2. query_length
  3. prompt_injection
  4. input_toxicity
  5. pii_detection  (redact)
  6. topic_restriction  (warn only)
  7. content_policy

Output pipeline (before answer reaches user):
  1. output_toxicity
  2. output_pii_scrub
  3. hallucination_signals
  4. citations_present
"""
from __future__ import annotations

import time
from typing import Optional

import structlog

from guardrails.types import GuardrailResult, GuardrailType, GuardrailViolation, Severity
from guardrails.input_guards import (
    check_query_length,
    check_input_toxicity,
    check_prompt_injection,
    check_pii,
    check_topic_restriction,
    check_content_policy,
)
from guardrails.output_guards import (
    check_output_toxicity,
    check_hallucination_signals,
    check_citations_present,
    check_output_pii,
)
from guardrails.rate_limiter import get_rate_limiter

logger = structlog.get_logger()


def _merge(base: GuardrailResult, new: GuardrailResult) -> GuardrailResult:
    """Merge a new check result into the running result."""
    for v in new.violations:
        base.add_violation(v)
    if not new.passed:
        base.passed = False
        if new.blocked_response:
            base.blocked_response = new.blocked_response
    if new.safe_text is not None:
        base.safe_text = new.safe_text
    return base


class GuardrailPipeline:
    """
    Stateless pipeline that runs every guard.
    Create one instance at application startup and reuse it.
    """

    def __init__(
        self,
        enable_rate_limit: bool = True,
        faithfulness_threshold: float = 0.60,
        require_citations: bool = True,
        redact_pii: bool = True,
    ):
        self.enable_rate_limit = enable_rate_limit
        self.faithfulness_threshold = faithfulness_threshold
        self.require_citations = require_citations
        self.redact_pii = redact_pii

    # ── Input pipeline ────────────────────────────────────────────────────

    def run_input(
        self,
        query: str,
        client_id: str = "anonymous",
    ) -> GuardrailResult:
        """
        Run all input guards on a user query.
        Returns GuardrailResult — if passed=False, the API should
        return blocked_response immediately without calling agents.
        """
        t0 = time.perf_counter()
        result = GuardrailResult(passed=True, safe_text=query)

        # 1. Rate limit (check first — cheapest)
        if self.enable_rate_limit:
            rl_result = get_rate_limiter().check(client_id)
            if not rl_result.passed:
                logger.warning("guardrail_rate_limit", client_id=client_id)
                return rl_result

        # 2. Query length
        result = _merge(result, check_query_length(result.safe_text or query))
        if result.is_blocked:
            return self._log_and_return(result, "input", t0)

        # 3. Prompt injection (highest priority after length)
        result = _merge(result, check_prompt_injection(result.safe_text or query))
        if result.is_blocked:
            return self._log_and_return(result, "input", t0)

        # 4. Input toxicity
        result = _merge(result, check_input_toxicity(result.safe_text or query))
        if result.is_blocked:
            return self._log_and_return(result, "input", t0)

        # 5. PII detection + redaction
        if self.redact_pii:
            result = _merge(result, check_pii(result.safe_text or query, redact=True))

        # 6. Topic restriction (warning only, never blocks)
        result = _merge(result, check_topic_restriction(result.safe_text or query))

        # 7. Content policy
        result = _merge(result, check_content_policy(result.safe_text or query))

        return self._log_and_return(result, "input", t0)

    # ── Output pipeline ───────────────────────────────────────────────────

    def run_output(
        self,
        answer: str,
        retrieved_chunks: Optional[list[dict]] = None,
        faithfulness_score: Optional[float] = None,
        block_on_low_faithfulness: bool = False,
    ) -> GuardrailResult:
        """
        Run output guards on a generated answer.

        Grounding checks (hedging, faithfulness, citations) only apply to answers that
        were generated from retrieved documents; general chat answers are only
        screened for toxicity and leaked PII. A low faithfulness score adds a
        warning instead of hiding the whole answer (set block_on_low_faithfulness
        to restore the old hard-block behaviour).
        """
        t0 = time.perf_counter()
        result = GuardrailResult(passed=True, safe_text=answer)

        # 1. Output toxicity (hard block)
        result = _merge(result, check_output_toxicity(result.safe_text or answer))
        if result.is_blocked:
            return self._log_and_return(result, "output", t0)

        # 2. PII scrub
        if self.redact_pii:
            result = _merge(result, check_output_pii(result.safe_text or answer))
            if result.is_blocked:
                return self._log_and_return(result, "output", t0)

        if not retrieved_chunks:
            return self._log_and_return(result, "output", t0)

        # 3. Hallucination signals + faithfulness threshold
        low = (faithfulness_score is not None and faithfulness_score < self.faithfulness_threshold)
        hall = check_hallucination_signals(
            result.safe_text or answer,
            retrieved_chunks=retrieved_chunks,
            faithfulness_score=faithfulness_score if block_on_low_faithfulness else None,
            threshold=self.faithfulness_threshold,
        )
        if low and not block_on_low_faithfulness:
            hall.add_violation(GuardrailViolation(
                guardrail=GuardrailType.HALLUCINATION, severity=Severity.MEDIUM,
                message=(f"Faithfulness score {faithfulness_score:.2f} below threshold "
                         f"{self.faithfulness_threshold}."),
            ))
        result = _merge(result, hall)
        if result.is_blocked:
            return self._log_and_return(result, "output", t0)

        # 4. Citation enforcement (warn only; sources are always returned separately)
        result = _merge(result, check_citations_present(
            result.safe_text or answer,
            retrieved_chunks=retrieved_chunks,
            require_citations=self.require_citations,
        ))

        return self._log_and_return(result, "output", t0)

    # ── Logging helper ────────────────────────────────────────────────────

    def _log_and_return(
        self, result: GuardrailResult, stage: str, t0: float
    ) -> GuardrailResult:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        violations = [
            {"guardrail": v.guardrail.value, "severity": v.severity.value, "message": v.message}
            for v in result.violations
        ]
        try:
            from monitoring.metrics import get_registry
            reg = get_registry()
            for v in result.violations:
                action = "block" if v.severity in (Severity.HIGH, Severity.BLOCK) else "warn"
                reg.record_guardrail(stage, v.guardrail.value, action)
        except Exception:
            pass
        if result.is_blocked:
            logger.warning(
                "guardrail_blocked",
                stage=stage,
                violations=violations,
                latency_ms=latency_ms,
            )
        elif result.violations:
            logger.info(
                "guardrail_warnings",
                stage=stage,
                violations=violations,
                latency_ms=latency_ms,
            )
        return result


# ── Global singleton ──────────────────────────────────────────────────────

_pipeline: Optional[GuardrailPipeline] = None


def get_pipeline() -> GuardrailPipeline:
    global _pipeline
    if _pipeline is None:
        import os
        _pipeline = GuardrailPipeline(
            enable_rate_limit=os.getenv("GUARDRAIL_RATE_LIMIT", "true").lower() == "true",
            faithfulness_threshold=float(os.getenv("FAITHFULNESS_THRESHOLD", "0.60")),
            require_citations=os.getenv("REQUIRE_CITATIONS", "true").lower() == "true",
            redact_pii=os.getenv("REDACT_PII", "true").lower() == "true",
        )
    return _pipeline
