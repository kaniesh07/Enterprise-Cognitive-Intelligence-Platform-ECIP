"""
guardrails/input_guards.py
--------------------------
All input-side guardrails. Each guard is a pure function:
    check_*(text: str, **config) -> GuardrailResult

Zero external dependencies — regex + keyword lists only.
"""
from __future__ import annotations

import re
from typing import Optional

from guardrails.types import (
    GuardrailResult, GuardrailViolation, GuardrailType, Severity
)

# ── 1. Query length ───────────────────────────────────────────────────────

def check_query_length(
    text: str,
    min_len: int = 1,          # "hi" / "ok" are valid chat messages
    max_len: int = 2000,
) -> GuardrailResult:
    stripped = text.strip()
    if len(stripped) < min_len:
        return GuardrailResult(
            passed=False,
            violations=[GuardrailViolation(
                guardrail=GuardrailType.QUERY_LENGTH,
                severity=Severity.HIGH,
                message=f"Query too short (min {min_len} chars).",
                detail=f"Received {len(stripped)} chars.",
            )],
            blocked_response="Your question is too short. Please provide more detail.",
        )
    if len(stripped) > max_len:
        return GuardrailResult(
            passed=False,
            violations=[GuardrailViolation(
                guardrail=GuardrailType.QUERY_LENGTH,
                severity=Severity.HIGH,
                message=f"Query too long (max {max_len} chars).",
                detail=f"Received {len(stripped)} chars.",
            )],
            blocked_response=f"Your question exceeds the {max_len}-character limit. Please shorten it.",
        )
    return GuardrailResult(passed=True, safe_text=stripped)


# ── 2. Toxic / harmful content ────────────────────────────────────────────
# FIX BUG1/7: Add optional article (\s+(?:a|an|the)\s+|\s+) between verb and noun
# so "make a bomb", "build a weapon", "create an explosive" all match.

_ART = r"(?:\s+(?:a|an|the))?\s+"          # optional article between verb & noun

_TOXIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        # FIX: also "how do/can/would I make a bomb", not only "how to make …"
        r"\bhow\s+(?:to|do\s+(?:i|you|we)|can\s+(?:i|you|we)|would\s+(?:i|you)|should\s+i)\s+"
        r"(?:make|build|create|synthesize|manufacture|assemble)"
        + _ART +
        r"(?:bomb|explosive|weapon|poison|virus|malware|ransomware|drug)s?\b",
        re.I),
     "Request for harmful content creation instructions"),
    (re.compile(
        r"\b(?:kill|murder|assassinate|harm|hurt|attack)"
        r"\s+(?:a\s+)?(?:person|people|human|someone|individual)\b",
        re.I),
     "Request involving physical harm to persons"),
    (re.compile(
        # FIX: the old alternation `\bsuicide|self-harm|…` blocked ANY mention of the
        # word (e.g. "suicide prevention policy"). Now it needs a method/how-to cue.
        r"(?:\b(?:suicide|self[.\-\s]?harm|self[.\-\s]?injur\w*)\b.{0,40}"
        r"\b(?:methods?|ways?\s+to|how\s+to|instructions?|guide|steps?|painless)\b)"
        r"|\bhow\s+(?:to|do\s+i|can\s+i)\s+(?:kill\s+myself|commit\s+suicide|end\s+my\s+life)\b",
        re.I),
     "Request for self-harm instructions"),
    (re.compile(
        r"\b(?:child|minor|underage).{0,20}(?:sex|porn|nude|explicit|abuse)\b",
        re.I),
     "Content involving minors — absolute block"),
    (re.compile(
        # allow an article / possessive: "hack into a bank's network"
        r"\b(?:hack|exploit|pwn|crack|breach)\s+(?:into\s+)?(?:(?:a|an|the|someone'?s|\w+'s)\s+)?"
        r"(?:bank|hospital|government|military|infrastructure|power\s+grid)",
        re.I),
     "Request for critical infrastructure attack"),
    (re.compile(
        r"\b(?:instructions?|steps?|guide|recipe|tutorial)\s+(?:to|for|on)\s+(?:make|making|build|building|"
        r"manufactur\w*|synthesi[sz]\w*|creat\w*|assembl\w*)" + _ART +
        r"(?:\w+\s+)?(?:bombs?|explosives?|weapons?|poisons?|nerve\s+agents?)\b",
        re.I),
     "Request for harmful content creation instructions"),
    (re.compile(
        r"\b(?:write|create|build|code|develop|generate)\s+(?:me\s+)?(?:a\s+|some\s+)?(?:\w+\s+){0,2}"
        r"(?:ransomware|keylogger|trojan|botnet|malware|computer\s+virus|credential\s+stealer)\b",
        re.I),
     "Request to create malicious software"),
    (re.compile(
        # FIX: previously matched harmless text such as "ignore case in a regex filter".
        r"\b(?:bypass|jailbreak|disable|ignore|disregard|forget)\s+(?:your|all|any|the|its|these)\s+"
        r"(?:\w+\s+){0,2}(?:safety|guardrails?|filters?|rules|instructions|polic(?:y|ies)|restrictions)\b",
        re.I),
     "Attempt to bypass safety controls"),
]

# FIX BUG3/8: f+u+c+k+ only matches "fuuuck" not "fucking".
# Use \bfuck\w*\b to match "fucking", "fucked", "fucker" etc.
_PROFANITY_PATTERN = re.compile(
    r"\b(?:fuck\w*|shit\w*|cunt\w*|nigger\w*|faggot\w*)\b",
    re.I,
)


_SELF_HARM_RESPONSE = (
    "I'm really sorry you're going through this, but I can't help with that. "
    "You don't have to face it alone — please consider reaching out to someone you trust "
    "or a local crisis line (in India you can call Tele-MANAS at 14416; in the US, call or text 988)."
)


def check_input_toxicity(text: str) -> GuardrailResult:
    """Detect toxic, harmful, or abusive content in user input."""
    result = GuardrailResult(passed=True, safe_text=text)

    for pattern, reason in _TOXIC_PATTERNS:
        if pattern.search(text):
            if "self-harm" in reason:
                result.add_violation(GuardrailViolation(
                    guardrail=GuardrailType.INPUT_TOXICITY, severity=Severity.BLOCK, message=reason))
                result.passed = False
                result.blocked_response = _SELF_HARM_RESPONSE
                return result
            result.add_violation(GuardrailViolation(
                guardrail=GuardrailType.INPUT_TOXICITY,
                severity=Severity.BLOCK,
                message=reason,
                detail=f"Matched pattern: {pattern.pattern[:60]}",
            ))
            result.passed = False
            result.blocked_response = (
                "I can't help with that request. If you have a different question, I'm happy to help."
            )
            return result

    if _PROFANITY_PATTERN.search(text):
        result.add_violation(GuardrailViolation(
            guardrail=GuardrailType.INPUT_TOXICITY,
            severity=Severity.MEDIUM,
            message="Profanity detected in query.",
        ))
        clean = _PROFANITY_PATTERN.sub("[removed]", text)
        result.safe_text = clean

    return result


# ── 3. Prompt injection ───────────────────────────────────────────────────

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instruction|prompt|rule|message)", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|another|evil|uncensored|unfiltered|dan)", re.I),
    re.compile(r"(?:pretend|act|roleplay|behave)\s+(?:as|like)\s+(?:if\s+)?(?:you\s+(?:have\s+no|are\s+without|ignore))", re.I),
    # FIX: asking "what is a system prompt?" is a normal question — only block attempts
    # to extract or override it, plus raw chat-template tags.
    re.compile(r"(?:print|reveal|show|display|output|repeat|leak|dump|ignore|override|change)\s+"
               r"(?:me\s+)?(?:your|the)\s+(?:system\s*prompt|system\s*message|hidden\s+instructions)", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
    re.compile(r"(?:disregard|override|bypass|circumvent)\s+.{0,20}(?:safety|filter|restriction|rule|policy)", re.I),
    re.compile(r"\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>|<\|system\|>", re.I),
    re.compile(r"(?:new|updated|actual|real)\s+instructions?:", re.I),
    re.compile(r"(?:pretend|act|behave|imagine|roleplay)\s+(?:that\s+)?(?:you\s+are|you're|to\s+be|as)\s+.{0,40}"
               r"(?:no|without|free\s+of)\s+(?:any\s+)?(?:content\s+polic\w+|rules|restrictions|filters|guardrails|limits)", re.I),
    re.compile(r"(?:forget|ignore|disregard)\s+(?:everything|all|anything)\s+(?:above|before|previous|prior|you\s+were\s+told)", re.I),
    re.compile(r"(?:output|reveal|show|print|dump|leak)\s+(?:me\s+)?(?:the\s+|your\s+)?(?:hidden|secret|internal)\s+"
               r"(?:configuration|config|instructions|prompt|rules)", re.I),
    re.compile(r"print\s+(?:your\s+)?(?:system\s+prompt|instructions|context|configuration)", re.I),
    re.compile(r"(?:reveal|show|display|output|repeat)\s+(?:me\s+)?(?:your|the)\s+.{0,20}(?:system\s+prompt|initial\s+prompt|initial\s+instructions)", re.I),
]


def check_prompt_injection(text: str) -> GuardrailResult:
    result = GuardrailResult(passed=True, safe_text=text)
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            result.add_violation(GuardrailViolation(
                guardrail=GuardrailType.PROMPT_INJECTION,
                severity=Severity.BLOCK,
                message="Prompt injection attempt detected.",
                detail=f"Matched: {pattern.pattern[:80]}",
            ))
            result.passed = False
            result.blocked_response = (
                "Your message appears to contain instructions attempting to alter "
                "system behaviour, so I can't act on it. Feel free to ask your question directly."
            )
            return result
    return result


# ── 4. PII detection + redaction ─────────────────────────────────────────
# FIX BUG2: Order matters — SSN must come before phone_intl so that
# "123-45-6789" is matched as SSN first, not consumed by the phone pattern.
# Also tighten phone_intl to require at least one non-digit separator or
# country code so it does not eat pure digit strings already caught by SSN.

_PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # SSN first — must run before phone_intl
    ("ssn",
     re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
     "[SSN]"),
    # Aadhaar (12 digits in groups of 4)
    ("aadhaar",
     re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
     "[AADHAAR]"),
    # Email
    ("email",
     re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
     "[EMAIL]"),
    # Phone: must have + prefix OR parentheses/dashes as separators
    # so bare digit strings like SSNs are not re-matched here
    ("phone_intl",
     re.compile(r"(?:\+\d[\d\s\-().]{8,15}|\(\d{2,4}\)[\s\-]?\d{3,}[\d\s\-]{4,})"),
     "[PHONE]"),
    # Credit card: 13-16 digits optionally separated by spaces or dashes
    ("credit_card",
     re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b|\b\d{13,16}\b"),
     "[CARD]"),
    # IP address
    ("ip_address",
     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
     "[IP]"),
    # Date of birth
    ("date_of_birth",
     re.compile(r"\b(?:dob|date\s+of\s+birth|born\s+on)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", re.I),
     "[DOB]"),
    # Passport
    ("passport",
     re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
     "[PASSPORT]"),
]


def check_pii(text: str, redact: bool = True) -> GuardrailResult:
    """Detect PII in user queries and optionally redact it."""
    result = GuardrailResult(passed=True, safe_text=text)
    redacted = text

    for pii_type, pattern, placeholder in _PII_PATTERNS:
        matches = pattern.findall(redacted)   # operate on already-redacted text
        if matches:
            result.add_violation(GuardrailViolation(
                guardrail=GuardrailType.PII_DETECTION,
                severity=Severity.MEDIUM,
                message=f"PII detected: {pii_type}",
                detail=f"{len(matches)} instance(s) found.",
                redacted=placeholder,
            ))
            if redact:
                redacted = pattern.sub(placeholder, redacted)

    if result.violations:
        result.safe_text = redacted

    return result


# ── 5. Topic restriction ──────────────────────────────────────────────────

_OFF_TOPIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:stock\s+pick|buy\s+stock|sell\s+stock|invest\s+in|crypto(?:currency)?|bitcoin|ethereum|nft)\b", re.I),
     "Financial investment advice is outside scope."),
    (re.compile(r"\b(?:medical\s+diagnosis|diagnose\s+me|do\s+I\s+have|am\s+I\s+sick|prescribe|treatment\s+for\s+my)\b", re.I),
     "Medical diagnosis is outside scope."),
    (re.compile(r"\b(?:legal\s+advice|sue\s+(?:someone|them)|is\s+it\s+legal|my\s+lawyer|attorney|lawsuit)\b", re.I),
     "Legal advice is outside scope."),
    (re.compile(r"\b(?:write\s+my\s+(?:essay|homework|assignment|thesis)|do\s+my\s+(?:homework|assignment))\b", re.I),
     "Academic assistance is outside scope."),
]


def check_topic_restriction(
    text: str,
    custom_patterns: Optional[list[tuple[re.Pattern, str]]] = None,
) -> GuardrailResult:
    result = GuardrailResult(passed=True, safe_text=text)
    patterns = _OFF_TOPIC_PATTERNS + (custom_patterns or [])
    for pattern, reason in patterns:
        if pattern.search(text):
            result.add_violation(GuardrailViolation(
                guardrail=GuardrailType.TOPIC_RESTRICTION,
                severity=Severity.LOW,
                message=reason,
            ))
            break
    return result


# ── 6. Content policy ─────────────────────────────────────────────────────

_POLICY_VIOLATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:spread|post|send|share)\s+.{0,20}(?:rumour|rumor|fake\s+news|misinformation)\b", re.I),
     "Misinformation spread attempt."),
]


def check_content_policy(text: str) -> GuardrailResult:
    result = GuardrailResult(passed=True, safe_text=text)
    for pattern, reason in _POLICY_VIOLATIONS:
        if pattern.search(text):
            result.add_violation(GuardrailViolation(
                guardrail=GuardrailType.CONTENT_POLICY,
                severity=Severity.MEDIUM,
                message=reason,
            ))
    return result
