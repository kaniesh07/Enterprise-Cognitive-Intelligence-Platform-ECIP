"""
guardrails/rate_limiter.py
---------------------------
Sliding-window rate limiter. Pure Python, thread-safe, no Redis needed.

FIX BUG5/6: blocked_response now includes the limit type ("burst" / "per-minute"
/ "per-hour") so tests that check for those words in the response pass.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from guardrails.types import (
    GuardrailResult, GuardrailViolation, GuardrailType, Severity
)


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 30
    requests_per_hour:   int = 500
    burst_limit:         int = 5
    burst_window_seconds: float = 5.0


@dataclass
class _WindowState:
    minute_window: deque = field(default_factory=deque)
    hour_window:   deque = field(default_factory=deque)
    burst_window:  deque = field(default_factory=deque)


class RateLimiter:
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._state: dict[str, _WindowState] = defaultdict(_WindowState)
        self._lock = Lock()

    def check(self, client_id: str) -> GuardrailResult:
        now = time.monotonic()
        cfg = self.config

        with self._lock:
            state = self._state[client_id]
            _prune(state.minute_window, now - 60)
            _prune(state.hour_window,   now - 3600)
            _prune(state.burst_window,  now - cfg.burst_window_seconds)

            # Burst check
            if len(state.burst_window) >= cfg.burst_limit:
                retry = cfg.burst_window_seconds
                return _rl_result(
                    kind="burst",
                    violation_msg=(
                        f"Burst limit exceeded: {cfg.burst_limit} requests "
                        f"in {cfg.burst_window_seconds}s window."
                    ),
                    retry_after=retry,
                )

            # Per-minute check
            if len(state.minute_window) >= cfg.requests_per_minute:
                oldest = state.minute_window[0]
                retry = max(60.0 - (now - oldest), 1.0)
                return _rl_result(
                    kind="minute",
                    violation_msg=(
                        f"Per-minute rate limit: "
                        f"{cfg.requests_per_minute} requests/minute exceeded."
                    ),
                    retry_after=retry,
                )

            # Per-hour check
            if len(state.hour_window) >= cfg.requests_per_hour:
                oldest = state.hour_window[0]
                retry = max(3600.0 - (now - oldest), 1.0)
                return _rl_result(
                    kind="hour",
                    violation_msg=(
                        f"Hourly rate limit: "
                        f"{cfg.requests_per_hour} requests/hour exceeded."
                    ),
                    retry_after=retry,
                )

            # All clear — record request
            state.minute_window.append(now)
            state.hour_window.append(now)
            state.burst_window.append(now)

        return GuardrailResult(passed=True)

    def get_stats(self, client_id: str) -> dict:
        now = time.monotonic()
        with self._lock:
            state = self._state.get(client_id)
            if not state:
                return {"minute": 0, "hour": 0, "burst": 0}
            _prune(state.minute_window, now - 60)
            _prune(state.hour_window,   now - 3600)
            _prune(state.burst_window,  now - self.config.burst_window_seconds)
            return {
                "minute": len(state.minute_window),
                "hour":   len(state.hour_window),
                "burst":  len(state.burst_window),
            }

    def reset(self, client_id: str):
        with self._lock:
            self._state.pop(client_id, None)

    def reset_all(self):
        with self._lock:
            self._state.clear()


def _prune(window: deque, cutoff: float):
    while window and window[0] < cutoff:
        window.popleft()


def _rl_result(kind: str, violation_msg: str, retry_after: float) -> GuardrailResult:
    """
    FIX BUG5/6: blocked_response includes the limit kind so tests that check
    for 'Burst' or 'minute' in blocked_response pass correctly.
    """
    kind_label = {
        "burst":  "Burst rate limit",
        "minute": "Per-minute rate limit",
        "hour":   "Hourly rate limit",
    }.get(kind, "Rate limit")

    return GuardrailResult(
        passed=False,
        violations=[GuardrailViolation(
            guardrail=GuardrailType.RATE_LIMIT,
            severity=Severity.HIGH,
            message=violation_msg,
            detail=f"Retry after {retry_after:.0f}s",
        )],
        blocked_response=(
            f"{kind_label} reached. "
            f"Please wait {retry_after:.0f} seconds before retrying."
        ),
    )


# ── Global singleton ──────────────────────────────────────────────────────

_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        import os
        cfg = RateLimitConfig(
            requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "30")),
            requests_per_hour=int(os.getenv("RATE_LIMIT_RPH", "500")),
            burst_limit=int(os.getenv("RATE_LIMIT_BURST", "5")),
        )
        _limiter = RateLimiter(cfg)
    return _limiter
