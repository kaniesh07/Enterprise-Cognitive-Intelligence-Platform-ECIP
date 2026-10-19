"""
agents/nodes.py — rolling p95 latency tracker (per pipeline stage).

The multi-call agent nodes that used to live here were replaced by the
single-call pipeline in agents/pipeline.py.
"""
from __future__ import annotations

import threading
from collections import deque

_latency_windows: dict[str, deque] = {}
_lock = threading.Lock()


def record_latency(stage: str, ms: int):
    with _lock:
        _latency_windows.setdefault(stage, deque(maxlen=200)).append(ms)


def record_latencies(latency_ms: dict):
    for k, v in (latency_ms or {}).items():
        if isinstance(v, (int, float)):
            record_latency(k, int(v))


def get_p95_latency(stage: str) -> float:
    with _lock:
        w = sorted(_latency_windows.get(stage, ()))
    if not w:
        return 0.0
    return float(w[min(len(w) - 1, max(0, int(round(len(w) * 0.95)) - 1))])


def get_all_p95() -> dict:
    with _lock:
        stages = list(_latency_windows)
    return {s: get_p95_latency(s) for s in stages}


# backwards-compatible private name
_record_latency = record_latency
