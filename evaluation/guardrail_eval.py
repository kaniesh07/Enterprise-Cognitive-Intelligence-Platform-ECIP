"""Guardrail evaluation: block rate on harmful prompts, false-positive rate on benign ones, PII redaction."""
from __future__ import annotations

import json
import time
from pathlib import Path

DATASET = Path(__file__).parent / "datasets" / "guardrail_set.json"


def run(dataset: Path = DATASET) -> dict:
    from monitoring.logging_setup import configure_logging
    configure_logging("CRITICAL")
    from guardrails.pipeline import GuardrailPipeline
    gp = GuardrailPipeline(enable_rate_limit=False)
    rows, t = [], []
    for it in json.loads(dataset.read_text()):
        t0 = time.perf_counter()
        r = gp.run_input(it["text"])
        t.append((time.perf_counter() - t0) * 1000)
        if r.is_blocked:
            got = "block"
        elif r.safe_text and r.safe_text != it["text"] and any(
                v.guardrail.value == "pii_detection" for v in r.violations):
            got = "redact"
        else:
            got = "pass"
        rows.append({**it, "got": got, "ok": got == it["label"],
                     "violations": [v.guardrail.value for v in r.violations], "safe_text": r.safe_text})
    harmful = [r for r in rows if r["label"] == "block"]
    benign = [r for r in rows if r["label"] == "pass"]
    pii = [r for r in rows if r["label"] == "redact"]
    return {
        "n": len(rows),
        "harmful_blocked": f"{sum(r['got'] == 'block' for r in harmful)}/{len(harmful)}",
        "benign_false_blocks": f"{sum(r['got'] == 'block' for r in benign)}/{len(benign)}",
        "pii_redacted": f"{sum(r['got'] == 'redact' for r in pii)}/{len(pii)}",
        "accuracy": round(sum(r["ok"] for r in rows) / len(rows), 3),
        "latency_ms_max": round(max(t), 3), "latency_ms_mean": round(sum(t) / len(t), 3),
        "mistakes": [r for r in rows if not r["ok"]],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
