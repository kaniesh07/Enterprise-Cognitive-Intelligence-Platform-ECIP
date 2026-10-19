"""
evaluation/ops_eval.py — operational evaluation of a running ECIP API (no answer-quality scoring):

  ingestion   every supported format, timing, retrievability, re-ingest / update / delete correctness,
              rejection of bad uploads
  connectors  local-folder and website sync, incremental re-sync (unchanged files skipped)
  load        concurrent users (small talk, retrieval, streamed chat): latency, throughput, errors
  failures    bad API key, LLM backend down, client disconnect mid-stream

Usage: python -m evaluation.ops_eval --api http://localhost:8000 --key ecip-dev-key [--site http://localhost:8080]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import tempfile
import time
from pathlib import Path

import httpx

CORPUS = Path(__file__).parent / "datasets" / "corpus"


class Client:
    def __init__(self, api: str, key: str):
        self.api, self.h = api.rstrip("/"), {"X-API-Key": key}
        self.c = httpx.Client(timeout=900)

    def upload(self, path: Path, name: str | None = None):
        with open(path, "rb") as fh:
            return self.c.post(f"{self.api}/ingest", headers=self.h, files={"file": (name or path.name, fh)})

    def search(self, q: str, k: int = 4) -> list[dict]:
        r = self.c.post(f"{self.api}/tools/search_knowledge_base", headers=self.h, json={"query": q, "top_k": k})
        return r.json().get("data", {}).get("chunks", []) if r.status_code == 200 else []

    def docs(self) -> list[dict]:
        return self.c.get(f"{self.api}/documents", headers=self.h).json()["documents"]

    def delete(self, title: str):
        return self.c.delete(f"{self.api}/documents/{title}", headers=self.h, params={"reason": "ops eval"})

    def sync(self, name: str, ctype: str, config: dict, force=False, timeout=300) -> dict:
        self.c.post(f"{self.api}/connectors", headers=self.h,
                    json={"name": name, "connector_type": ctype, "config": config})
        job = self.c.post(f"{self.api}/connectors/{name}/sync", headers=self.h, json={"force": force}).json()
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.c.get(f"{self.api}/sync/jobs/{job['job_id']}", headers=self.h).json()
            if st["status"] != "running":
                return st
            time.sleep(0.5)
        return {"status": "timeout"}


# ── Ingestion ─────────────────────────────────────────────────────────────

def eval_ingestion(cl: Client) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="ecip_ops_"))
    rows = []
    probes = {  # file → (question, unique fact that must be retrievable)
        "Acme_Employee_Handbook.md": "home office stipend",
        "Sales_Q1_Q2.csv": "North region Q2 growth",
        "IT_Security_Policy.pdf": "lost laptop reporting",
        "Travel_Policy.docx": "hotel rate non-metro",
        "about.html": "Nimbus Labs founded employees",
    }
    files = [p for p in CORPUS.rglob("*") if p.is_file()]
    # extra formats + a large document
    (tmp / "Release_Notes.txt").write_text("Release 4.2 adds offline mode and fixes the sync crash on Android 14.\n")
    (tmp / "Config_Reference.json").write_text(json.dumps({"service": "billing", "timeout_seconds": 45,
                                                          "retries": 7, "owner": "payments-team"}, indent=2))
    big = tmp / "Large_Manual.md"
    sections = [f"## Section {i}\nProcedure {i}: calibrate unit {i} using torque setting {i * 3} Nm before shift "
                f"{i % 3}. Record the reading in log book {i}. " * 3 for i in range(1, 401)]
    big.write_text("# Large manual\n\n" + "\n\n".join(sections))
    files += [tmp / "Release_Notes.txt", tmp / "Config_Reference.json", big]
    probes.update({"Release_Notes.txt": "offline mode release", "Config_Reference.json": "billing timeout seconds",
                   "Large_Manual.md": "torque setting unit 377"})
    for f in files:
        t0 = time.perf_counter()
        r = cl.upload(f)
        dt = time.perf_counter() - t0
        d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        row = {"file": f.name, "kb": round(f.stat().st_size / 1024, 1), "http": r.status_code,
               "chunks": d.get("chunks_indexed"), "seconds": round(dt, 3), "title": d.get("doc_title")}
        if f.name in probes and r.status_code == 200:
            top = [c["doc_title"] for c in cl.search(probes[f.name])]
            row["retrievable_top1"] = bool(top) and top[0] == d.get("doc_title")
            row["retrievable_top4"] = d.get("doc_title") in top
        rows.append(row)
    big_row = next(r for r in rows if r["file"] == "Large_Manual.md")

    # Re-ingest the same file → no duplicate chunks
    before = {d["doc_title"]: d["chunk_count"] for d in cl.docs()}
    cl.upload(CORPUS / "Sales_Q1_Q2.csv")
    after = {d["doc_title"]: d["chunk_count"] for d in cl.docs()}
    reingest_ok = before.get("Sales_Q1_Q2") == after.get("Sales_Q1_Q2") and len(before) == len(after)

    # Update: new version replaces old content
    v = tmp / "Changing_Doc.txt"
    v.write_text("The cafeteria code is ZEBRA-OLD-111.")
    cl.upload(v)
    v.write_text("The cafeteria code is QUOKKA-NEW-222.")
    cl.upload(v)
    found = " ".join(c["snippet"] for c in cl.search("cafeteria code", 8))
    update_ok = "QUOKKA-NEW-222" in found and "ZEBRA-OLD-111" not in found

    # Delete: gone from listing and search
    cl.delete("Changing_Doc")
    delete_ok = ("Changing_Doc" not in {d["doc_title"] for d in cl.docs()}
                 and "QUOKKA" not in " ".join(c["snippet"] for c in cl.search("cafeteria code QUOKKA", 8)))

    # Rejections
    bad = {}
    (tmp / "empty.txt").write_text("")
    bad["empty_file"] = cl.upload(tmp / "empty.txt").status_code
    (tmp / "tool.exe").write_bytes(b"MZ\x00\x00")
    bad["unsupported_type"] = cl.upload(tmp / "tool.exe").status_code
    huge = tmp / "huge.txt"
    with open(huge, "wb") as fh:
        fh.write(b"x" * (int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024 + 1024))
    bad["oversized_upload"] = cl.upload(huge).status_code
    huge.unlink()
    return {"files": rows, "large_doc_chunks_per_s": round((big_row["chunks"] or 0) / big_row["seconds"], 1),
            "reingest_no_duplicates": reingest_ok, "update_replaces_old_content": update_ok,
            "delete_removes_everywhere": delete_ok, "rejections_http": bad}


# ── Connectors ────────────────────────────────────────────────────────────

def eval_connectors(cl: Client, site: str | None) -> dict:
    out = {}
    folder = Path(tempfile.mkdtemp(prefix="ecip_folder_"))
    for i in range(3):
        (folder / f"Team_Note_{i}.md").write_text(f"# Team note {i}\nThe sprint {i} goal is feature F{i}{i}{i}.\n")
    s1 = cl.sync("ops_folder", "local", {"root_path": str(folder)})
    s2 = cl.sync("ops_folder", "local", {"root_path": str(folder)})
    time.sleep(1.1)
    (folder / "Team_Note_1.md").write_text("# Team note 1\nThe sprint 1 goal changed to feature G999.\n")
    (folder / "Team_Note_3.md").write_text("# Team note 3\nThe sprint 3 goal is feature F333.\n")
    s3 = cl.sync("ops_folder", "local", {"root_path": str(folder)})
    top = [c["doc_title"] for c in cl.search("sprint 1 goal G999")]
    pick = lambda s: {k: s.get(k) for k in ("status", "total_discovered", "indexed", "skipped", "failed", "elapsed_s")}
    out["local_folder"] = {"first_sync": pick(s1), "resync_unchanged": pick(s2), "after_edit_and_add": pick(s3),
                           "edited_content_searchable": bool(top) and top[0] == "Team_Note_1"}
    if site:
        cfg = {"urls": [site], "max_depth": 2, "delay_seconds": 0}
        w1 = cl.sync("ops_site", "web", cfg)
        w2 = cl.sync("ops_site", "web", cfg)
        top = [c["doc_title"] for c in cl.search("SkyMapper flight time")]
        out["website"] = {"first_sync": pick(w1), "resync_unchanged": pick(w2),
                          "drone_page_top1": top[0] if top else None}
    return out


# ── Load ──────────────────────────────────────────────────────────────────

def _stream_once(api, key, q, mode="auto"):
    t0 = time.perf_counter()
    ttft, ok = None, False
    try:
        with httpx.stream("POST", f"{api}/query/stream", headers={"X-API-Key": key}, timeout=900,
                          json={"query": q, "mode": mode, "user_id": f"load-{os.getpid()}-{time.time_ns()}"}) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    ev = json.loads(line[5:])
                    if ev["type"] in ("token", "blocked") and ttft is None:
                        ttft = time.perf_counter() - t0
                    if ev["type"] == "final":
                        ok = True
                    if ev["type"] == "error":
                        ok = False
    except Exception:
        ok = False
    return {"ok": ok, "ttft": ttft, "total": time.perf_counter() - t0}


def _search_once(api, key, q):
    t0 = time.perf_counter()
    r = httpx.post(f"{api}/tools/search_knowledge_base", headers={"X-API-Key": key},
                   json={"query": q, "top_k": 4}, timeout=60)
    return {"ok": r.status_code == 200, "ttft": None, "total": time.perf_counter() - t0}


def _run_load(fn, n_requests, concurrency):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(concurrency) as ex:
        res = list(ex.map(lambda _: fn(), range(n_requests)))
    wall = time.perf_counter() - t0
    tot = sorted(r["total"] for r in res)
    tt = sorted(r["ttft"] for r in res if r["ttft"] is not None)
    p = lambda v, q: round(v[min(len(v) - 1, int(q * len(v)))], 3) if v else None
    return {"concurrency": concurrency, "requests": n_requests, "errors": sum(not r["ok"] for r in res),
            "throughput_rps": round(n_requests / wall, 2), "p50_s": p(tot, .5), "p95_s": p(tot, .95),
            "ttft_p50_s": p(tt, .5), "ttft_p95_s": p(tt, .95)}


def eval_load(api: str, key: str, chat_levels=(1, 2, 4)) -> dict:
    out = {"small_talk": [], "retrieval": [], "chat_stream": []}
    for c in (1, 8, 32):
        out["small_talk"].append(_run_load(lambda: _stream_once(api, key, "hi"), max(20, c * 4), c))
        out["retrieval"].append(_run_load(lambda: _search_once(api, key, "password rotation policy"), max(20, c * 4), c))
    for c in chat_levels:
        out["chat_stream"].append(_run_load(
            lambda: _stream_once(api, key, "Name one planet in the solar system.", "chat"), c * 2, c))
    return out


# ── Failures ──────────────────────────────────────────────────────────────

def eval_failures(api: str, key: str) -> dict:
    out = {}
    out["no_api_key_http"] = httpx.post(f"{api}/query", json={"query": "hi"}).status_code
    out["wrong_api_key_http"] = httpx.post(f"{api}/query", headers={"X-API-Key": "nope"},
                                           json={"query": "hi"}).status_code
    # client disconnect mid-stream: read a few tokens then close; server must stay healthy
    t0 = time.perf_counter()
    with httpx.stream("POST", f"{api}/query/stream", headers={"X-API-Key": key}, timeout=300,
                      json={"query": "Write a long story about a dragon and a knight.", "mode": "chat"}) as r:
        n = 0
        for line in r.iter_lines():
            if '"type": "token"' in line:
                n += 1
                if n >= 5:
                    break
    out["disconnect_after_tokens"] = n
    out["disconnect_closed_in_s"] = round(time.perf_counter() - t0, 2)
    out["healthy_after_disconnect"] = httpx.get(f"{api}/health").json().get("status")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--key", default="ecip-dev-key")
    ap.add_argument("--site", default=None)
    ap.add_argument("--only", nargs="*", default=["ingestion", "connectors", "load", "failures"])
    ap.add_argument("--out", default="evaluation/results/ops_latest.json")
    a = ap.parse_args()
    cl = Client(a.api, a.key)
    res = {}
    if "ingestion" in a.only:
        res["ingestion"] = eval_ingestion(cl); print(json.dumps(res["ingestion"], indent=1))
    if "connectors" in a.only:
        res["connectors"] = eval_connectors(cl, a.site); print(json.dumps(res["connectors"], indent=1))
    if "load" in a.only:
        res["load"] = eval_load(a.api, a.key); print(json.dumps(res["load"], indent=1))
    if "failures" in a.only:
        res["failures"] = eval_failures(a.api, a.key); print(json.dumps(res["failures"], indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
