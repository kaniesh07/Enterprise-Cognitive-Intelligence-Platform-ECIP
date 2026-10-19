"""
Generates monitoring/grafana/dashboards/ecip.json (run: python monitoring/build_dashboard.py).
Keeping the dashboard as code makes panels/queries reviewable and easy to extend.
"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
panels, _id = [], [0]
y = [0]


def nid():
    _id[0] += 1
    return _id[0]


def row(title):
    panels.append({"type": "row", "title": title, "id": nid(), "collapsed": False,
                   "gridPos": {"h": 1, "w": 24, "x": 0, "y": y[0]}, "panels": []})
    y[0] += 1


def stat(title, expr, x, w=3, unit="short", thresholds=None, decimals=None, desc="", mappings=None):
    p = {"type": "stat", "title": title, "id": nid(), "datasource": DS, "description": desc,
         "gridPos": {"h": 4, "w": w, "x": x, "y": y[0]},
         "targets": [{"refId": "A", "expr": expr, "datasource": DS, "instant": True}],
         "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                     "colorMode": "background", "graphMode": "none", "textMode": "auto"},
         "fieldConfig": {"defaults": {"unit": unit, "thresholds": {"mode": "absolute", "steps":
                         thresholds or [{"color": "blue", "value": None}]}, "mappings": mappings or []},
                         "overrides": []}}
    if decimals is not None:
        p["fieldConfig"]["defaults"]["decimals"] = decimals
    panels.append(p)


def ts(title, targets, x, w=12, h=8, unit="short", desc="", stack=False):
    panels.append({
        "type": "timeseries", "title": title, "id": nid(), "datasource": DS, "description": desc,
        "gridPos": {"h": h, "w": w, "x": x, "y": y[0]},
        "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": l, "datasource": DS}
                    for i, (e, l) in enumerate(targets)],
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
        "fieldConfig": {"defaults": {"unit": unit, "custom": {
            "drawStyle": "line", "lineWidth": 2, "fillOpacity": 12, "showPoints": "never",
            "spanNulls": True, "stacking": {"mode": "normal" if stack else "none"}}}, "overrides": []},
    })


RED_GREEN = [{"color": "red", "value": None}, {"color": "green", "value": 1}]
UPDOWN = [{"type": "value", "options": {"0": {"text": "DOWN"}, "1": {"text": "UP"}}}]
LAT = [{"color": "green", "value": None}, {"color": "orange", "value": 5}, {"color": "red", "value": 15}]

# ── Overview ──────────────────────────────────────────────────────────────
row("Overview")
stat("LLM backend", "max(ecip_llm_up)", 0, thresholds=RED_GREEN, mappings=UPDOWN)
stat("Vector DB", "max(ecip_vectordb_up)", 3, thresholds=RED_GREEN, mappings=UPDOWN)
stat("Chats / min", "sum(rate(ecip_pipeline_requests_total[5m])) * 60", 6, decimals=1)
stat("p50 first token", 'histogram_quantile(0.5, sum by (le) (rate(ecip_time_to_first_token_seconds_bucket{mode!="instant"}[15m])))',
     9, unit="s", thresholds=LAT, decimals=2, desc="Time until the user sees the first word")
stat("p95 full answer", "histogram_quantile(0.95, sum by (le) (rate(ecip_response_seconds_bucket[15m])))",
     12, unit="s", thresholds=LAT, decimals=1)
stat("Error rate", '(sum(rate(ecip_pipeline_requests_total{status="error"}[15m])) or vector(0)) / clamp_min(sum(rate(ecip_pipeline_requests_total[15m])), 1e-9)',
     15, unit="percentunit", decimals=1,
     thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 0.02}, {"color": "red", "value": 0.1}])
stat("Documents", "max(ecip_indexed_documents)", 18)
stat("Chunks", "max(ecip_indexed_chunks)", 21)
y[0] += 4

# ── Latency ───────────────────────────────────────────────────────────────
row("Latency")
ts("Time to first token (p50 / p95)", [
    ('histogram_quantile(0.5, sum by (le, mode) (rate(ecip_time_to_first_token_seconds_bucket[5m])))', "p50 {{mode}}"),
    ('histogram_quantile(0.95, sum by (le, mode) (rate(ecip_time_to_first_token_seconds_bucket[5m])))', "p95 {{mode}}"),
], 0, unit="s")
ts("End-to-end answer time (p50 / p95)", [
    ('histogram_quantile(0.5, sum by (le, mode) (rate(ecip_response_seconds_bucket[5m])))', "p50 {{mode}}"),
    ('histogram_quantile(0.95, sum by (le, mode) (rate(ecip_response_seconds_bucket[5m])))', "p95 {{mode}}"),
], 12, unit="s")
y[0] += 8
ts("Pipeline stage latency p95", [
    ('histogram_quantile(0.95, sum by (le, stage) (rate(ecip_stage_latency_seconds_bucket[5m])))', "{{stage}}"),
], 0, w=8, unit="s", desc="route / retrieval / llm / verify")
ts("LLM call duration p95 by model", [
    ('histogram_quantile(0.95, sum by (le, model) (rate(ecip_llm_call_duration_seconds_bucket[5m])))', "{{model}}"),
], 8, w=8, unit="s")
ts("Generation speed (tokens/s, median)", [
    ('histogram_quantile(0.5, sum by (le, model) (rate(ecip_llm_tokens_per_second_bucket[5m])))', "{{model}}"),
], 16, w=8, unit="short")
y[0] += 8

# ── Traffic ───────────────────────────────────────────────────────────────
row("Traffic")
ts("Requests by answer mode", [
    ('sum by (mode) (rate(ecip_pipeline_requests_total{status="ok"}[5m])) * 60', "{{mode}}"),
    ('(sum(rate(ecip_pipeline_requests_total{status="error"}[5m])) or vector(0)) * 60', "errors"),
    ('(sum(rate(ecip_pipeline_requests_total{status="blocked"}[5m])) or vector(0)) * 60', "blocked"),
], 0, w=8, unit="short", desc="per minute; grounded = answered from documents", stack=True)
ts("HTTP requests by route", [
    ('sum by (route, status) (rate(ecip_http_requests_total[5m])) * 60', "{{route}} {{status}}"),
], 8, w=8, unit="short")
ts("HTTP latency p95 by route", [
    ('histogram_quantile(0.95, sum by (le, route) (rate(ecip_http_request_duration_seconds_bucket[5m])))', "{{route}}"),
], 16, w=8, unit="s")
y[0] += 8

# ── Quality & safety ──────────────────────────────────────────────────────
row("Answer quality & guardrails")
ts("Grounding (faithfulness) of document answers", [
    ("sum(rate(ecip_faithfulness_score_sum[15m])) / clamp_min(sum(rate(ecip_faithfulness_score_count[15m])), 1e-9)", "avg faithfulness"),
    ("histogram_quantile(0.5, sum by (le) (rate(ecip_retrieval_top_score_bucket[15m])))", "median top retrieval score"),
], 0, w=8, unit="percentunit")
ts("Guardrail events", [
    ("sum by (guardrail, action) (increase(ecip_guardrail_events_total[15m]))", "{{guardrail}} ({{action}})"),
], 8, w=8, unit="short", stack=True)
ts("Unsupported-claim flags & auth failures", [
    ("sum(increase(ecip_hallucination_flags_total[15m]))", "unsupported claims"),
    ("sum(increase(ecip_auth_failures_total[15m]))", "auth failures"),
], 16, w=8, unit="short")
y[0] += 8

# ── Tokens & ingestion ────────────────────────────────────────────────────
row("Tokens, cost & ingestion")
ts("Tokens per minute", [
    ('sum by (kind) (rate(ecip_llm_tokens_total[5m])) * 60', "{{kind}}"),
], 0, w=8, unit="short", stack=True)
ts("Projected cost (cloud-equivalent USD, last 1h)", [
    ("sum(increase(ecip_projected_cost_usd_total[1h]))", "USD"),
], 8, w=8, unit="currencyUSD")
ts("Ingestion", [
    ("sum(increase(ecip_ingestion_requests_total[15m]))", "documents"),
    ("sum(increase(ecip_ingested_chunks_total[15m]))", "chunks"),
    ("sum(increase(ecip_ingestion_errors_total[15m]))", "errors"),
], 16, w=8, unit="short")
y[0] += 8

dash = {
    "uid": "ecip-overview", "title": "ECIP — Chat & RAG", "tags": ["ecip", "llm", "rag"],
    "timezone": "browser", "schemaVersion": 39, "version": 3, "editable": True,
    "refresh": "10s", "time": {"from": "now-1h", "to": "now"},
    "annotations": {"list": []}, "templating": {"list": []}, "panels": panels,
}
out = Path(__file__).parent / "grafana" / "dashboards" / "ecip.json"
out.write_text(json.dumps(dash, indent=2))
print(f"wrote {out} ({len(panels)} panels)")
