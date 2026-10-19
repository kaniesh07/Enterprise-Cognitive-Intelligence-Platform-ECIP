"""
guardrails/types.py
--------------------
Shared types used across all guardrail modules.
No external dependencies — safe to import anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    LOW    = "low"      # log only
    MEDIUM = "medium"   # warn user, still answer
    HIGH   = "high"     # block, return safe response
    BLOCK  = "block"    # hard block, no response


class GuardrailType(str, Enum):
    INPUT_TOXICITY      = "input_toxicity"
    PROMPT_INJECTION    = "prompt_injection"
    PII_DETECTION       = "pii_detection"
    TOPIC_RESTRICTION   = "topic_restriction"
    QUERY_LENGTH        = "query_length"
    OUTPUT_TOXICITY     = "output_toxicity"
    HALLUCINATION       = "hallucination"
    SOURCE_CITATION     = "source_citation"
    FAITHFULNESS        = "faithfulness"
    RATE_LIMIT          = "rate_limit"
    CONTENT_POLICY      = "content_policy"


@dataclass
class GuardrailViolation:
    guardrail: GuardrailType
    severity:  Severity
    message:   str                        # human-readable reason
    detail:    Optional[str]  = None      # optional extra context
    redacted:  Optional[str]  = None      # sanitised version of the input


@dataclass
class GuardrailResult:
    """Return value from every guardrail check."""
    passed:     bool
    violations: list[GuardrailViolation] = field(default_factory=list)
    safe_text:  Optional[str] = None      # sanitised/redacted text when passed=True
    blocked_response: Optional[str] = None  # what to tell the user when passed=False

    @property
    def is_blocked(self) -> bool:
        return not self.passed

    @property
    def highest_severity(self) -> Optional[Severity]:
        if not self.violations:
            return None
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.BLOCK]
        return max(self.violations, key=lambda v: order.index(v.severity)).severity

    def add_violation(self, v: GuardrailViolation):
        self.violations.append(v)
        if v.severity in (Severity.HIGH, Severity.BLOCK):
            self.passed = False
