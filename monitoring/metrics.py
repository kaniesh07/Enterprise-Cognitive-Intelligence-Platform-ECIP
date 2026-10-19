"""
monitoring/metrics.py — Prometheus metrics for ECIP (scraped by Prometheus, shown in Grafana).

Built on the official `prometheus_client` library so the /metrics exposition is
always valid (the previous hand-written exporter emitted duplicate HELP/TYPE
lines for labelled series, which made Prometheus reject every scrape).

The `MetricsRegistry` keeps the old attribute API (`.value`, dict-style labelled
counters) so existing code and tests keep working.
"""
from __future__ import annotations

import time
from typing import Optional

from prometheus_client import disable_created_metrics
from prometheus_client import (
    CollectorRegistry, Counter as _PCounter, Gauge as _PGauge, Histogram as _PHistogram,
    generate_latest, CONTENT_TYPE_LATEST,
)

disable_created_metrics()   # drop the noisy *_created series

_LAT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60, 120)
_SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


# ── Thin wrappers that expose `.value` (backwards compatible API) ─────────

class _Value:
    def __init__(self, metric):
        self._m = metric

    @property
    def value(self) -> float:
        return self._m._value.get()


class Counter(_Value):
    def inc(self, n: float = 1.0):
        if n:
            self._m.inc(n)


class Gauge(_Value):
    def set(self, v: float):
        self._m.set(v)

    def inc(self, n: float = 1.0):
        self._m.inc(n)

    def dec(self, n: float = 1.0):
        self._m.dec(n)


class Histogram:
    def __init__(self, metric):
        self._m = metric

    def observe(self, v: float):
        self._m.observe(v)


class _Labelled(dict):
    """dict-like: registry.tool_calls_total["search"].inc()"""
    def __init__(self, metric, wrapper=Counter):
        super().__init__()
        self._metric, self._wrapper = metric, wrapper

    def __missing__(self, key):
        child = self._wrapper(self._metric.labels(key))
        self[key] = child
        return child


class MetricsRegistry:
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        r = self.registry = registry or CollectorRegistry(auto_describe=True)

        # ── HTTP ──────────────────────────────────────────────────────────
        self._http_requests = _PCounter("ecip_http_requests_total", "HTTP requests",
                                        ["method", "route", "status"], registry=r)
        self._http_latency = _PHistogram("ecip_http_request_duration_seconds", "HTTP latency",
                                         ["method", "route"], buckets=_LAT_BUCKETS, registry=r)
        self.http_in_flight = Gauge(_PGauge("ecip_http_in_flight_requests",
                                            "Requests being served", registry=r))

        # ── Chat pipeline ─────────────────────────────────────────────────
        self._pipeline = _PCounter("ecip_pipeline_requests_total", "Chat/RAG requests",
                                   ["mode", "status"], registry=r)
        self.pipeline_total = Counter(_PCounter("ecip_pipeline_requests_all_total",
                                                "All pipeline requests", registry=r))
        self.pipeline_errors = Counter(_PCounter("ecip_pipeline_errors_total",
                                                 "Pipeline errors", registry=r))
        self._ttft = _PHistogram("ecip_time_to_first_token_seconds", "Time to first token",
                                 ["mode"], buckets=_LAT_BUCKETS, registry=r)
        self._e2e = _PHistogram("ecip_response_seconds", "End-to-end answer latency",
                                ["mode"], buckets=_LAT_BUCKETS, registry=r)
        self._stage = _PHistogram("ecip_stage_latency_seconds", "Latency per pipeline stage",
                                  ["stage"], buckets=_LAT_BUCKETS, registry=r)
        # legacy name used by tests / old code
        self.agent_latency = {a: Histogram(self._stage.labels(a))
                              for a in ["intent", "retrieval", "multimodal", "reasoning", "verifier"]}

        # ── Quality ───────────────────────────────────────────────────────
        self.faithfulness_score = Histogram(_PHistogram(
            "ecip_faithfulness_score", "Grounding score of document answers",
            buckets=_SCORE_BUCKETS, registry=r))
        self.confidence_score = Histogram(_PHistogram(
            "ecip_confidence_score", "Confidence score", buckets=_SCORE_BUCKETS, registry=r))
        self.hallucination_flags = Counter(_PCounter(
            "ecip_hallucination_flags_total", "Unsupported-claim flags raised", registry=r))
        self.retrieval_chunks = Histogram(_PHistogram(
            "ecip_retrieval_chunks_returned", "Chunks used per grounded answer",
            buckets=(0, 1, 2, 3, 5, 7, 10, 15, 20), registry=r))
        self.retrieval_top_score = Histogram(_PHistogram(
            "ecip_retrieval_top_score", "Best dense similarity per query",
            buckets=_SCORE_BUCKETS, registry=r))
        self._guardrail = _PCounter("ecip_guardrail_events_total", "Guardrail events",
                                    ["stage", "guardrail", "action"], registry=r)

        # ── LLM ───────────────────────────────────────────────────────────
        self._llm_calls = _PCounter("ecip_llm_calls_total", "LLM calls",
                                    ["model", "status"], registry=r)
        self._llm_latency = _PHistogram("ecip_llm_call_duration_seconds", "LLM call latency",
                                        ["model"], buckets=_LAT_BUCKETS, registry=r)
        self._tokens = _PCounter("ecip_llm_tokens_total", "Tokens processed",
                                 ["model", "kind"], registry=r)
        self._tok_rate = _PHistogram("ecip_llm_tokens_per_second", "Generation speed",
                                     ["model"], buckets=(1, 2, 5, 10, 15, 20, 30, 50, 80, 120, 200),
                                     registry=r)
        self.prompt_tokens_total = Counter(_PCounter(
            "ecip_prompt_tokens_total", "Total prompt tokens", registry=r))
        self.completion_tokens_total = Counter(_PCounter(
            "ecip_completion_tokens_total", "Total completion tokens", registry=r))
        self.projected_cost_usd = Counter(_PCounter(
            "ecip_projected_cost_usd_total", "Projected USD cost (cloud-equivalent)", registry=r))
        self.tokens_per_request = Histogram(_PHistogram(
            "ecip_tokens_per_request", "Tokens per request",
            buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000), registry=r))
        self.model_call_counts = _Labelled(_PCounter(
            "ecip_model_calls_total", "Calls per model", ["model"], registry=r))
        self._embed_latency = _PHistogram("ecip_embedding_duration_seconds", "Embedding latency",
                                          ["provider"], buckets=_LAT_BUCKETS, registry=r)

        # ── Tools / intent / tasks ────────────────────────────────────────
        self.tool_calls_total = _Labelled(_PCounter(
            "ecip_tool_calls_total", "Tool calls", ["tool"], registry=r))
        self.tool_errors_total = _Labelled(_PCounter(
            "ecip_tool_errors_total", "Tool errors", ["tool"], registry=r))
        self.intent_correct_total = Counter(_PCounter(
            "ecip_intent_correct_total", "Correctly classified intents", registry=r))
        self.intent_total = Counter(_PCounter(
            "ecip_intent_total", "Intent classifications", registry=r))
        self.task_completed_total = Counter(_PCounter(
            "ecip_task_completed_total", "Requests answered successfully", registry=r))

        # ── System state ──────────────────────────────────────────────────
        self.circuit_open = Gauge(_PGauge(
            "ecip_circuit_breaker_open", "1 = LLM circuit open (failing fast)", registry=r))
        self.indexed_chunks = Gauge(_PGauge(
            "ecip_indexed_chunks", "Chunks in the vector store", registry=r))
        self.indexed_documents = Gauge(_PGauge(
            "ecip_indexed_documents", "Documents indexed", registry=r))
        self.llm_up = Gauge(_PGauge("ecip_llm_up", "1 = LLM backend reachable", registry=r))
        self.vectordb_up = Gauge(_PGauge("ecip_vectordb_up", "1 = vector DB reachable", registry=r))

        # ── Ingestion / sync ──────────────────────────────────────────────
        self.ingestion_total = Counter(_PCounter(
            "ecip_ingestion_requests_total", "Documents ingested", registry=r))
        self.ingestion_errors = Counter(_PCounter(
            "ecip_ingestion_errors_total", "Ingestion errors", registry=r))
        self.ingestion_duration = Histogram(_PHistogram(
            "ecip_ingestion_duration_seconds", "Ingestion latency", buckets=_LAT_BUCKETS, registry=r))
        self.ingested_chunks = Counter(_PCounter(
            "ecip_ingested_chunks_total", "Chunks created by ingestion", registry=r))
        self.sync_docs_indexed = Counter(_PCounter(
            "ecip_sync_docs_indexed_total", "Docs indexed via connectors", registry=r))
        self.sync_docs_failed = Counter(_PCounter(
            "ecip_sync_docs_failed_total", "Docs failed in sync", registry=r))
        self.sync_docs_skipped = Counter(_PCounter(
            "ecip_sync_docs_skipped_total", "Docs skipped (unchanged)", registry=r))
        self.sync_throughput = Histogram(_PHistogram(
            "ecip_sync_throughput_docs_per_min", "Sync throughput",
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000), registry=r))
        self.auth_failures = Counter(_PCounter(
            "ecip_auth_failures_total", "Authentication failures", registry=r))

        # legacy: generic HTTP histogram (no labels)
        self.http_request_duration = Histogram(_PHistogram(
            "ecip_http_request_duration_all_seconds", "HTTP latency (all routes)",
            buckets=_LAT_BUCKETS, registry=r))

    # ── Recording helpers ─────────────────────────────────────────────────
    def observe_http(self, method: str, route: str, status: int, seconds: float):
        self._http_requests.labels(method, route, str(status)).inc()
        self._http_latency.labels(method, route).observe(seconds)
        self.http_request_duration.observe(seconds)

    def observe_llm_call(self, model: str, seconds: float, prompt_tokens: int,
                         completion_tokens: int, ok: bool = True, gen_seconds: float = 0.0,
                         cost_usd: float = 0.0):
        self._llm_calls.labels(model, "ok" if ok else "error").inc()
        self.model_call_counts[model].inc()
        if not ok:
            return
        self._llm_latency.labels(model).observe(seconds)
        self._tokens.labels(model, "prompt").inc(max(prompt_tokens, 0))
        self._tokens.labels(model, "completion").inc(max(completion_tokens, 0))
        self.prompt_tokens_total.inc(max(prompt_tokens, 0))
        self.completion_tokens_total.inc(max(completion_tokens, 0))
        self.projected_cost_usd.inc(max(cost_usd, 0.0))
        if gen_seconds > 0 and completion_tokens > 0:
            self._tok_rate.labels(model).observe(completion_tokens / gen_seconds)

    def observe_embedding(self, provider: str, seconds: float):
        self._embed_latency.labels(provider).observe(seconds)

    def observe_stage(self, stage: str, seconds: float):
        self._stage.labels(stage).observe(seconds)

    def observe_ttft(self, mode: str, seconds: float):
        self._ttft.labels(mode).observe(seconds)

    def record_guardrail(self, stage: str, guardrail: str, action: str):
        self._guardrail.labels(stage, guardrail, action).inc()

    def record_request(self, mode: str, status: str, seconds: Optional[float] = None):
        self._pipeline.labels(mode, status).inc()
        if seconds is not None and status == "ok":
            self._e2e.labels(mode).observe(seconds)

    def record_pipeline_result(self, result: dict):
        """Record quality metrics for one finished answer (also used by tests)."""
        self.pipeline_total.inc()
        if result.get("error"):
            self.pipeline_errors.inc()
            return
        mode = result.get("mode", "")
        if mode == "grounded" or (mode == "" and result.get("retrieved_chunks")):
            if (fs := result.get("faithfulness_score")) is not None:
                self.faithfulness_score.observe(float(fs))
            self.retrieval_chunks.observe(len(result.get("retrieved_chunks") or []))
        if (cs := result.get("confidence")) is not None:
            self.confidence_score.observe(float(cs))
        flags = result.get("hallucination_flags") or []
        if flags:
            self.hallucination_flags.inc(len(flags))

        for stage, ms in (result.get("latency_ms") or {}).items():
            if isinstance(ms, (int, float)) and stage != "total":
                self.observe_stage(stage.replace("_agent", ""), ms / 1000.0)

        tu = result.get("token_usage") or {}
        if tu.get("total"):
            self.tokens_per_request.observe(tu.get("total", 0))
        if not result.get("_llm_metrics_recorded"):
            # Usage reported by a caller that did not go through llm_client
            self.prompt_tokens_total.inc(tu.get("prompt", 0) or 0)
            self.completion_tokens_total.inc(tu.get("completion", 0) or 0)
            self.projected_cost_usd.inc(tu.get("projected_cost_usd", 0) or 0)

        answer = (result.get("final_answer") or "").lower()
        fail_phrases = ["error occurred", "something went wrong", "unable to", "temporarily unavailable"]
        if answer and not any(p in answer for p in fail_phrases):
            self.task_completed_total.inc()

    def record_tool_call(self, tool_name: str, success: bool):
        self.tool_calls_total[tool_name].inc()
        if not success:
            self.tool_errors_total[tool_name].inc()

    def record_model_call(self, model_name: str):
        self.model_call_counts[model_name].inc()

    def record_intent(self, predicted: str, ground_truth: Optional[str]):
        self.intent_total.inc()
        if ground_truth and predicted == ground_truth:
            self.intent_correct_total.inc()

    def set_circuit_breaker(self, is_open: bool):
        self.circuit_open.set(1.0 if is_open else 0.0)

    def to_prometheus_text(self) -> str:
        return generate_latest(self.registry).decode("utf-8")


CONTENT_TYPE = CONTENT_TYPE_LATEST

_registry: Optional[MetricsRegistry] = None


def get_registry() -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


def _route_template(scope) -> str:
    """Use the matched route template (e.g. /documents/{doc_title}) to keep label cardinality low."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    p = scope.get("path", "")
    return p if p in ("/", "/health", "/metrics") else "unmatched"


def make_metrics_middleware():
    """Pure ASGI middleware (does not buffer streaming responses)."""

    class MetricsMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await self.app(scope, receive, send)
            reg = get_registry()
            t0 = time.perf_counter()
            status = {"code": 500}

            async def _send(message):
                if message["type"] == "http.response.start":
                    status["code"] = message["status"]
                await send(message)

            reg.http_in_flight.inc()
            try:
                await self.app(scope, receive, _send)
            finally:
                reg.http_in_flight.dec()
                route = _route_template(scope)
                if route != "/metrics":
                    reg.observe_http(scope.get("method", "GET"), route, status["code"],
                                     time.perf_counter() - t0)

    return MetricsMiddleware
