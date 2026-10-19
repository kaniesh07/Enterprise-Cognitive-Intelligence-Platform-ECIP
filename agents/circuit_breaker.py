"""
agents/circuit_breaker.py
--------------------------
Circuit breaker for Ollama/ChromaDB calls.

States:
  CLOSED  — normal operation, requests flow through
  OPEN    — too many failures, requests fail immediately (fail-fast)
  HALF_OPEN — probe: one request allowed through to test if service recovered

This prevents the system from hammering a failing service with retries.
When Ollama goes down, the circuit opens and all requests fail immediately
with a clear "service unavailable" message instead of waiting 30s each.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Callable, Optional, TypeVar
import structlog

logger = structlog.get_logger()

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED    = "closed"     # normal
    OPEN      = "open"       # failing fast
    HALF_OPEN = "half_open"  # testing recovery


@dataclass
class CircuitBreakerConfig:
    failure_threshold:    int   = 5       # failures before opening
    success_threshold:    int   = 2       # successes in HALF_OPEN before closing
    timeout_seconds:      float = 60.0   # seconds to wait before HALF_OPEN probe
    expected_exceptions:  tuple = (Exception,)


class CircuitBreakerOpen(Exception):
    """Raised when the circuit is OPEN and a call is attempted."""
    pass


class CircuitBreaker:
    """
    Thread-safe circuit breaker.

    Usage:
        cb = CircuitBreaker("ollama", CircuitBreakerConfig(failure_threshold=5))
        result = cb.call(lambda: my_llm_function(prompt))
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name   = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count   = 0
        self._success_count   = 0
        self._last_failure_ts: Optional[float] = None
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._current_state()

    def _current_state(self) -> CircuitState:
        """Must be called with self._lock held."""
        if self._state == CircuitState.OPEN:
            if (self._last_failure_ts and
                    time.monotonic() - self._last_failure_ts >= self.config.timeout_seconds):
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("circuit_half_open", name=self.name)
        return self._state

    def call(self, fn: Callable[[], T], fallback: Optional[Callable[[], T]] = None) -> T:
        """
        Execute fn through the circuit breaker.
        If circuit is OPEN, raises CircuitBreakerOpen (or calls fallback if provided).
        """
        with self._lock:
            state = self._current_state()

        if state == CircuitState.OPEN:
            logger.warning("circuit_open_rejected", name=self.name)
            if fallback:
                return fallback()
            raise CircuitBreakerOpen(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Service appears to be down. Retry in "
                f"{self.config.timeout_seconds:.0f}s."
            )

        try:
            result = fn()
            self._on_success()
            return result
        except self.config.expected_exceptions:
            self._on_failure()
            raise

    # ── Explicit API (used by async/streaming callers) ────────────────────
    def allow(self) -> None:
        """Raise CircuitBreakerOpen if calls are currently rejected."""
        with self._lock:
            state = self._current_state()
        if state == CircuitState.OPEN:
            logger.warning("circuit_open_rejected", name=self.name)
            raise CircuitBreakerOpen(
                f"Circuit breaker '{self.name}' is OPEN. Service appears to be down. "
                f"Retry in {self.config.timeout_seconds:.0f}s."
            )

    def record_success(self) -> None:
        self._on_success()

    def record_failure(self) -> None:
        self._on_failure()

    def _on_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("circuit_closed", name=self.name)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # reset on success

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_ts = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("circuit_reopened", name=self.name)
            elif (self._state == CircuitState.CLOSED and
                  self._failure_count >= self.config.failure_threshold):
                self._state = CircuitState.OPEN
                logger.error("circuit_opened", name=self.name,
                             failures=self._failure_count)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "name":          self.name,
                "state":         self._current_state().value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure":  self._last_failure_ts,
            }

    def reset(self):
        """Manually reset the circuit to CLOSED (for testing or admin action)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_ts = None
        logger.info("circuit_reset", name=self.name)


# ── Global circuit breakers for each external service ────────────────────

import os
_DEFAULT_CFG = CircuitBreakerConfig(
    failure_threshold=int(os.getenv("CB_FAILURE_THRESHOLD", "5")),
    success_threshold=int(os.getenv("CB_SUCCESS_THRESHOLD", "2")),
    timeout_seconds=float(os.getenv("CB_TIMEOUT_SECONDS", "60")),
)

_breakers: dict[str, CircuitBreaker] = {}


_breakers_lock = Lock()


def get_breaker(name: str) -> CircuitBreaker:
    """Get or create a named circuit breaker (thread-safe)."""
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(name, _DEFAULT_CFG)
        return _breakers[name]


def get_all_stats() -> list[dict]:
    return [b.get_stats() for b in _breakers.values()]


# Convenience wrappers for the two main external services
ollama_breaker  = get_breaker("ollama")
chroma_breaker  = get_breaker("chromadb")
