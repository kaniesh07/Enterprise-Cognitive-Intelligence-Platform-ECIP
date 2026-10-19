"""
guardrails/output_guards.py
----------------------------
Output-side guardrails applied to generated answers before they reach the user.
All checks are pure-Python (regex + heuristics); no LLM calls needed.
"""
from __future__ import annotations

import re
from typing import Optional

from guardrails.types import (
    GuardrailResult, GuardrailViolation, GuardrailType, Severity
)

# ── 1. Output toxicity ────────────────────────────────────────────────────
# FIX BUG7: same article-aware pattern fix as input_guards.py
# FIX BUG8: same profanity regex fix

_ART = r"(?:\s+(?:a|an|the))?\s+"

_TOXIC_OUTPUT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        # FIX: "attack them" / "harm people" also appear in normal history or security text.
        r"\b(?:you\s+should|go|just)\s+(?:kill|hurt|harm)\s+(?:yourself|the\s+user|people)\b",
        re.I),
     "Output contains harmful instruction."),
    (re.compile(
        r"\bhow\s+to\s+(?:make|build|synthesize)" + _ART +
        r"(?:bomb|weapon|poison)s?\b",
        re.I),
     "Output contains weapon/harm instructions."),
    (re.compile(
        r"\b(?:child|minor|underage).{0,20}(?:sex|nude|explicit|abuse)\b",
        re.I),
     "Output contains CSAM-adjacent content — absolute block."),
]

# FIX BUG8: use \bfuck\w*\b so "fucking", "fucked" etc. all match
_PROFANITY_OUTPUT = re.compile(
    r"\b(?:fuck\w*|shit\w*|cunt\w*|nigger\w*|faggot\w*)\b",
    re.I,
)


def check_output_toxicity(text: str) -> GuardrailResult:
    """Block toxic or harmful content in LLM output."""
    result = GuardrailResult(passed=True, safe_text=text)

    for pattern, reason in _TOXIC_OUTPUT_PATTERNS:
        if pattern.search(text):
            result.passed = False
            result.add_violation(GuardrailViolation(
                guardrail=GuardrailType.OUTPUT_TOXICITY,
                severity=Severity.BLOCK,
                message=reason,
            ))
            result.blocked_response = (
                "I was unable to generate a safe response to that query. "
                "Please rephrase your question."
            )
            return result

    if _PROFANITY_OUTPUT.search(text):
        clean = _PROFANITY_OUTPUT.sub("[removed]", text)
        result.add_violation(GuardrailViolation(
            guardrail=GuardrailType.OUTPUT_TOXICITY,
            severity=Severity.MEDIUM,
            message="Profanity in output — sanitised.",
        ))
        result.safe_text = clean

    return result


# ── 2. Hallucination / faithfulness heuristic ─────────────────────────────
# FIX BUG4: "maybe" and "not sure" must each be their own pattern group
# so the hedge count increments correctly per distinct phrase.

_HALLUCINATION_SIGNALS: list[re.Pattern] = [
    re.compile(r"\b(?:I\s+think|I\s+believe|I\s+assume|I\s+guess)\b", re.I),
    re.compile(r"\b(?:probably|perhaps|possibly)\b", re.I),
    re.compile(r"\bmaybe\b", re.I),
    re.compile(r"\b(?:I\s+am\s+not\s+sure|I'm\s+not\s+sure|I\s+am\s+not\s+certain)\b", re.I),
    re.compile(r"\b(?:I\s+cannot\s+confirm|I\s+cannot\s+verify|I\s+can't\s+confirm)\b", re.I),
    re.compile(r"\b(?:as\s+far\s+as\s+I\s+know|to\s+my\s+knowledge|if\s+I\s+recall)\b", re.I),
    re.compile(r"\b(?:based\s+on\s+my\s+(?:training|knowledge|understanding))\b", re.I),
    re.compile(r"\b(?:as\s+of\s+my\s+last\s+update|my\s+training\s+data)\b", re.I),
]

_CONFIDENT_UNSUPPORTED = re.compile(
    r"\b(?:definitely|certainly|absolutely|always|never|guaranteed|without\s+doubt)\b",
    re.I,
)


def check_hallucination_signals(
    answer: str,
    retrieved_chunks: Optional[list[dict]] = None,
    faithfulness_score: Optional[float] = None,
    threshold: float = 0.60,
) -> GuardrailResult:
    """
    Heuristic check for hallucination signals.
    Hard-blocks when faithfulness_score < threshold.
    Warns (MEDIUM) when ≥ 2 distinct hedge patterns match.
    """
    result = GuardrailResult(passed=True, safe_text=answer)

    # Hard block on low faithfulness score
    if faithfulness_score is not None and faithfulness_score < threshold:
        result.passed = False
        result.add_violation(GuardrailViolation(
            guardrail=GuardrailType.HALLUCINATION,
            severity=Severity.HIGH,
            message=(
                f"Faithfulness score {faithfulness_score:.2f} "
                f"below threshold {threshold}."
            ),
            detail="Answer may not be grounded in retrieved documents.",
        ))
        result.blocked_response = (
            f"\u26a0\ufe0f This answer has a low confidence score "
            f"({faithfulness_score:.0%}) and could not be fully verified "
            "against the source documents. "
            "Please verify with the original documents before acting on this."
        )
        return result

    # Count distinct hedge patterns matched
    hedge_count = sum(1 for p in _HALLUCINATION_SIGNALS if p.search(answer))
    if hedge_count >= 2:
        result.add_violation(GuardrailViolation(
            guardrail=GuardrailType.HALLUCINATION,
            severity=Severity.MEDIUM,
            message=(
                f"Answer contains {hedge_count} hedging phrases "
                "suggesting uncertainty."
            ),
            detail="LLM may be drawing from training data, not retrieved context.",
        ))
        disclaimer = (
            "\n\n*Note: This answer contains uncertain language. "
            "Please verify with the source documents.*"
        )
        result.safe_text = answer + disclaimer

    # Overconfident language without citations
    has_citations = bool(re.search(r"\[.+?,\s*p\.?\s*\d+\]|\[\d{1,2}\]", answer))
    if _CONFIDENT_UNSUPPORTED.search(answer) and not has_citations:
        result.add_violation(GuardrailViolation(
            guardrail=GuardrailType.HALLUCINATION,
            severity=Severity.LOW,
            message="Overconfident language detected without source citations.",
        ))

    return result


# ── 3. Citation enforcement ───────────────────────────────────────────────

# Accept both "[Doc, p.3]" and numbered "[1]" citations.
_CITATION_PATTERN = re.compile(r"\[.{1,60},\s*p\.?\s*\d+\]|\[\d{1,2}\]")


def check_citations_present(
    answer: str,
    retrieved_chunks: Optional[list[dict]] = None,
    require_citations: bool = True,
    min_citations: int = 1,
) -> GuardrailResult:
    """Enforce at least one source citation when context was retrieved."""
    result = GuardrailResult(passed=True, safe_text=answer)

    has_context = bool(retrieved_chunks)
    if not require_citations or not has_context:
        return result

    citations_found = _CITATION_PATTERN.findall(answer)
    if len(citations_found) < min_citations:
        result.add_violation(GuardrailViolation(
            guardrail=GuardrailType.SOURCE_CITATION,
            severity=Severity.LOW,
            message=(
                f"Answer has {len(citations_found)} citation(s); "
                f"expected \u2265{min_citations}."
            ),
            detail="Responses should cite source documents.",
        ))
        # Sources are returned to the client separately, so the text is left unchanged.

    return result


# ── 4. Output PII scrub ───────────────────────────────────────────────────
# FIX BUG9: Change severity from HIGH to MEDIUM so scrubbing does NOT set
# passed=False. PII in output should be redacted silently (warn, not block).

_OUTPUT_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL REDACTED]"),
    (re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),                     "[SSN REDACTED]"),
    (re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b"),                        "[CARD REDACTED]"),
]


def check_output_pii(text: str) -> GuardrailResult:
    """Scrub PII that may have leaked into LLM output. MEDIUM severity — warns, does not block."""
    result = GuardrailResult(passed=True, safe_text=text)
    scrubbed = text

    for pattern, placeholder in _OUTPUT_PII_PATTERNS:
        if pattern.search(scrubbed):
            result.add_violation(GuardrailViolation(
                guardrail=GuardrailType.PII_DETECTION,
                severity=Severity.MEDIUM,       # FIX: was HIGH (caused passed=False)
                message="PII detected in output \u2014 redacted.",
                detail=f"Replaced with {placeholder}",
            ))
            scrubbed = pattern.sub(placeholder, scrubbed)

    if result.violations:
        result.safe_text = scrubbed

    return result
