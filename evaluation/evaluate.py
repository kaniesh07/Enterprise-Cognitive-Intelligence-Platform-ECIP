"""
evaluation/evaluate.py — end-to-end evaluation of a running ECIP deployment.

Measures what users experience, through the public API:
  * routing accuracy        — chat vs grounded vs instant, as expected
  * retrieval quality       — Hit@1, Hit@k, MRR, nDCG@k, context precision (document-level relevance)
  * answer correctness      — required facts present in the answer (keyword/number match)
  * citation accuracy       — grounded answers cite a relevant document
  * grounding               — ECIP's own faithfulness score, split by correct / incorrect answers
  * latency & tokens        — time-to-first-token and full-answer time per mode

Plus an in-process retrieval ablation (dense vs BM25 vs hybrid) on a temporary index.

Dataset format (evaluation/datasets/eval_set.json):
  {"id", "question", "history"?, "ground_truth", "must_contain": ["24", "one hour|1 hour"],
   "relevant_docs": ["Doc_Title"], "expected_mode": "grounded|chat|instant", "intent"}

CLI: python scripts/run_eval.py --help
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Optional

CORPUS_DIR = Path(__file__).parent / "datasets" / "corpus"
DATASET = Path(__file__).parent / "datasets" / "eval_set.json"


# ── Metric helpers ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"(?<=\d)[,\s](?=\d)", "", t)          # 2,500 → 2500 ; 4,80,000 → 480000
    return t


def fact_present(answer: str, fact: str) -> bool:
    """`fact` may list alternatives separated by '|'. Numbers must match as whole numbers."""
    a = normalize(answer)
    for alt in fact.split("|"):
        alt = normalize(alt.strip())
        if not alt:
            continue
        if re.fullmatch(r"[\d.]+", alt):
            if re.search(rf"(?<![\d.]){re.escape(alt)}(?![\d])", a):
                return True
        elif alt in a:
            return True
    return False


def keyword_recall(answer: str, facts: list[str]) -> Optional[float]:
    if not facts:
        return None
    return sum(fact_present(answer, f) for f in facts) / len(facts)


def rank_metrics(ranked_docs: list[str], relevant: set[str], k: int) -> dict:
    rels = [1 if d in relevant else 0 for d in ranked_docs[:k]]
    first = next((i for i, r in enumerate(rels, 1) if r), None)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    n_ideal = min(k, max(1, sum(1 for d in ranked_docs if d in relevant)))
    idcg = sum(1 / math.log2(i + 2) for i in range(n_ideal))
    return {
        "hit@1": 1.0 if rels[:1] == [1] else 0.0,
        f"hit@{k}": 1.0 if any(rels) else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        "context_precision": (sum(rels) / len(rels)) if rels else 0.0,
    }


def pct(values: list[float], q: float) -> Optional[float]:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    idx = min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))
    return round(v[idx], 3)


def mean(values) -> Optional[float]:
    v = [x for x in values if x is not None]
    return round(statistics.fmean(v), 4) if v else None


# ── API evaluation ────────────────────────────────────────────────────────

def ingest_corpus(api: str, key: str, corpus: Path = CORPUS_DIR) -> list[dict]:
    import httpx
    out = []
    with httpx.Client(timeout=600) as c:
        for f in sorted(p for p in corpus.rglob("*") if p.is_file()):
            with open(f, "rb") as fh:
                r = c.post(f"{api}/ingest", headers={"X-API-Key": key}, files={"file": (f.name, fh)})
            out.append({"file": str(f.relative_to(corpus)), "status": r.status_code,
                        **(r.json() if r.headers.get("content-type", "").startswith("application/json") else {})})
    return out


def _stream(client, api: str, key: str, item: dict) -> dict:
    t0 = time.perf_counter()
    ttft, final, meta, err = None, {}, {}, None
    with client.stream("POST", f"{api}/query/stream", headers={"X-API-Key": key},
                       json={"query": item["question"], "conversation_history": item.get("history", []),
                             "mode": item.get("mode", "auto"), "user_id": "eval"}) as r:
        if r.status_code != 200:
            r.read()
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "total_s": time.perf_counter() - t0}
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            if ev["type"] == "meta":
                meta = ev["data"]
            elif ev["type"] in ("token", "blocked") and ttft is None:
                ttft = time.perf_counter() - t0
            elif ev["type"] == "final":
                final = ev["data"]
            elif ev["type"] == "error":
                err = ev["data"]
    return {"ttft_s": ttft, "total_s": time.perf_counter() - t0, "final": final, "meta": meta, "error": err}


def run_api_evaluation(api: str, key: str, dataset_path: Path = DATASET, top_k: int = 4,
                       limit: Optional[int] = None, progress=print) -> dict:
    import httpx

    items = json.loads(Path(dataset_path).read_text())[:limit]
    rows = []
    with httpx.Client(timeout=900) as c:
        health = c.get(f"{api}/health").json()
        for i, item in enumerate(items, 1):
            row = {k: item.get(k) for k in ("id", "question", "expected_mode", "relevant_docs", "must_contain",
                                            "ground_truth", "intent")}
            relevant = set(item.get("relevant_docs") or [])

            # 1. Retrieval quality (no LLM) through the search tool
            if relevant:
                q = item["question"]
                if item.get("history"):
                    from agents.pipeline import retrieval_query
                    q = retrieval_query(q, item["history"])
                r = c.post(f"{api}/tools/search_knowledge_base", headers={"X-API-Key": key},
                           json={"query": q, "top_k": top_k})
                chunks = r.json().get("data", {}).get("chunks", []) if r.status_code == 200 else []
                row["retrieved_docs"] = [ch.get("doc_title") for ch in chunks]
                row["retrieval"] = rank_metrics(row["retrieved_docs"], relevant, top_k)

            # 2. Full answer through the streaming chat endpoint
            res = _stream(c, api, key, item)
            fin = res.get("final") or {}
            answer = fin.get("final_answer", "")
            row.update({
                "mode": fin.get("mode") or res.get("meta", {}).get("mode"),
                "route_reason": fin.get("route_reason"),
                "predicted_intent": fin.get("intent"),
                "answer": answer, "error": res.get("error"),
                "ttft_s": round(res["ttft_s"], 3) if res.get("ttft_s") is not None else None,
                "total_s": round(res["total_s"], 3),
                "faithfulness": fin.get("faithfulness_score"),
                "tokens": (fin.get("token_usage") or {}).get("total"),
                "citations": fin.get("citations", []),
                "sources": [s.get("doc_title") for s in fin.get("sources", [])],
                "cited_sources": [s.get("doc_title") for s in fin.get("sources", []) if s.get("cited")],
            })
            row["routing_correct"] = row["mode"] == item.get("expected_mode")
            row["keyword_recall"] = keyword_recall(answer, item.get("must_contain") or [])
            row["answer_correct"] = None if row["keyword_recall"] is None else row["keyword_recall"] == 1.0
            if row["mode"] == "grounded" and relevant:
                cited = row["cited_sources"] or row["sources"]
                row["citation_correct"] = any(d in relevant for d in cited)
            rows.append(row)
            progress(f"[{i:>2}/{len(items)}] {row['id']:<7} expected={item.get('expected_mode'):<8} "
                     f"got={str(row['mode']):<8} correct={row['answer_correct']} "
                     f"ttft={row['ttft_s']}s total={row['total_s']}s")

    return {"config": {"api": api, "top_k": top_k, "model": health.get("model"),
                       "llm_provider": health.get("llm_provider"), "embeddings": health.get("embed"),
                       "vectordb": health.get("vectordb_mode"), "dataset": str(dataset_path),
                       "n_items": len(items)},
            "summary": summarize(rows, top_k), "rows": rows}


def summarize(rows: list[dict], top_k: int) -> dict:
    by_mode = {}
    for m in ("grounded", "chat", "instant"):
        sub = [r for r in rows if r.get("expected_mode") == m]
        if sub:
            by_mode[m] = {
                "n": len(sub),
                "routing_accuracy": mean(1.0 if r["routing_correct"] else 0.0 for r in sub),
                "answer_accuracy": mean(1.0 if r["answer_correct"] else 0.0
                                        for r in sub if r["answer_correct"] is not None),
                "keyword_recall": mean(r["keyword_recall"] for r in sub),
                "ttft_p50_s": pct([r["ttft_s"] for r in sub], 0.5),
                "ttft_p95_s": pct([r["ttft_s"] for r in sub], 0.95),
                "total_p50_s": pct([r["total_s"] for r in sub], 0.5),
                "total_p95_s": pct([r["total_s"] for r in sub], 0.95),
                "tokens_mean": mean(r["tokens"] for r in sub),
            }
    confusion: dict = {}
    for r in rows:
        confusion.setdefault(r.get("expected_mode"), {}).setdefault(str(r.get("mode")), 0)
        confusion[r.get("expected_mode")][str(r.get("mode"))] += 1

    ret = [r["retrieval"] for r in rows if "retrieval" in r]
    retrieval = {k: mean(x[k] for x in ret) for k in (ret[0] if ret else {})}
    grounded = [r for r in rows if r.get("mode") == "grounded" and r.get("relevant_docs")]
    correct = [r for r in grounded if r.get("answer_correct")]
    wrong = [r for r in grounded if r.get("answer_correct") is False]
    return {
        "n": len(rows),
        "routing_accuracy": mean(1.0 if r["routing_correct"] else 0.0 for r in rows),
        "routing_confusion": confusion,
        "retrieval": {"n": len(ret), "top_k": top_k, **retrieval},
        "answer_accuracy_all": mean(1.0 if r["answer_correct"] else 0.0
                                    for r in rows if r["answer_correct"] is not None),
        "citation_accuracy": mean(1.0 if r.get("citation_correct") else 0.0 for r in grounded),
        "faithfulness_mean_grounded": mean(r["faithfulness"] for r in grounded),
        "faithfulness_when_correct": mean(r["faithfulness"] for r in correct),
        "faithfulness_when_wrong": mean(r["faithfulness"] for r in wrong),
        "errors": sum(1 for r in rows if r.get("error")),
        "by_expected_mode": by_mode,
        "intent_accuracy": mean(1.0 if r.get("predicted_intent") == r.get("intent") else 0.0
                                for r in rows if r.get("intent")),
    }


# ── Retrieval ablation (in-process, temporary index) ──────────────────────

def run_retrieval_ablation(dataset_path: Path = DATASET, corpus: Path = CORPUS_DIR, top_k: int = 4) -> dict:
    """Ingests the corpus into a temporary embedded index and compares retrieval strategies.
    Set DATA_DIR to a temp folder BEFORE importing ECIP modules (scripts/run_eval.py does this)."""
    from agents.pipeline import retrieval_query
    from ingestion.indexer import ingest_file
    from retrieval import retriever as R
    from retrieval.embeddings import signature

    for f in sorted(p for p in corpus.rglob("*") if p.is_file()):
        ingest_file(str(f), title=f.stem, source_path=f.name)

    items = [x for x in json.loads(Path(dataset_path).read_text()) if x.get("relevant_docs")]
    strategies = {
        "bm25_only": lambda q: R._bm25_search(q, top_k),
        "dense_only": lambda q: R._dense_search(R._embed_query(q), top_k),
        "hybrid_rrf": lambda q: R.HybridRetriever().retrieve(q, top_k_final=top_k, rerank=False),
    }
    out = {}
    for name, fn in strategies.items():
        scores, t = [], []
        for it in items:
            q = retrieval_query(it["question"], it.get("history", []))
            t0 = time.perf_counter()
            docs = [c.doc_title for c in fn(q)]
            t.append(time.perf_counter() - t0)
            scores.append(rank_metrics(docs, set(it["relevant_docs"]), top_k))
        out[name] = {k: mean(s[k] for s in scores) for k in scores[0]}
        out[name]["latency_p50_ms"] = round(1000 * statistics.median(t), 1)
    # Routing decision only (no LLM call): does auto mode pick chat / documents / small talk correctly?
    from agents.pipeline import prepare
    routed = []
    for it in json.loads(Path(dataset_path).read_text()):
        turn = prepare(it["question"], it.get("history"), "eval", "auto", top_k)
        routed.append({"id": it["id"], "expected": it["expected_mode"], "got": turn.mode,
                       "reason": turn.route_reason, **{k: turn.retrieval_metrics.get(k)
                       for k in ("top_dense_score", "lexical_overlap")}})
    routing = {"accuracy": mean(1.0 if r["expected"] == r["got"] else 0.0 for r in routed),
               "mistakes": [r for r in routed if r["expected"] != r["got"]]}
    return {"embeddings": signature(), "n_queries": len(items), "top_k": top_k, "results": out,
            "routing_no_llm": routing}


# ── Report ────────────────────────────────────────────────────────────────

def _f(x, p=False):
    if x is None:
        return "–"
    return f"{x * 100:.0f}%" if p else (f"{x:.3f}" if isinstance(x, float) else str(x))


def to_markdown(api_res: Optional[dict], ablation: Optional[dict]) -> str:
    L = ["# ECIP evaluation results\n"]
    if api_res:
        c, s = api_res["config"], api_res["summary"]
        L.append(f"Model **{c['model']}** ({c['llm_provider']}) · embeddings **{c['embeddings']}** · "
                 f"vector DB {c['vectordb']} · {c['n_items']} questions · top-k {c['top_k']}\n")
        L.append("## Summary\n")
        L.append("| Metric | Value |\n|---|---|")
        L.append(f"| Routing accuracy (chat / documents / small talk) | {_f(s['routing_accuracy'], True)} |")
        r = s["retrieval"]
        k = r.get("top_k")
        L.append(f"| Retrieval Hit@1 / Hit@{k} | {_f(r.get('hit@1'), True)} / {_f(r.get(f'hit@{k}'), True)} |")
        L.append(f"| Retrieval MRR / nDCG@{k} | {_f(r.get('mrr'))} / {_f(r.get(f'ndcg@{k}'))} |")
        L.append(f"| Context precision (relevant share of retrieved chunks) | {_f(r.get('context_precision'), True)} |")
        L.append(f"| Answer accuracy (all required facts present) | {_f(s['answer_accuracy_all'], True)} |")
        L.append(f"| Citation accuracy (document answers citing a relevant doc) | {_f(s['citation_accuracy'], True)} |")
        L.append(f"| Grounding score: mean / when correct / when wrong | {_f(s['faithfulness_mean_grounded'])} / "
                 f"{_f(s['faithfulness_when_correct'])} / {_f(s['faithfulness_when_wrong'])} |")
        L.append(f"| Heuristic intent accuracy | {_f(s['intent_accuracy'], True)} |")
        L.append(f"| Errors | {s['errors']} |\n")
        L.append("## By question type\n")
        L.append("| Expected | n | Routing | Answer accuracy | Fact recall | First token p50 / p95 | Full answer p50 / p95 | Tokens |")
        L.append("|---|---|---|---|---|---|---|---|")
        for m, v in s["by_expected_mode"].items():
            L.append(f"| {m} | {v['n']} | {_f(v['routing_accuracy'], True)} | {_f(v['answer_accuracy'], True)} | "
                     f"{_f(v['keyword_recall'], True)} | {_f(v['ttft_p50_s'])}s / {_f(v['ttft_p95_s'])}s | "
                     f"{_f(v['total_p50_s'])}s / {_f(v['total_p95_s'])}s | {_f(v['tokens_mean'])} |")
        L.append("\nRouting confusion (expected → actual): " + json.dumps(s["routing_confusion"]) + "\n")
    if ablation:
        L.append(f"## Retrieval ablation ({ablation['n_queries']} document questions, embeddings "
                 f"`{ablation['embeddings']}`, top-{ablation['top_k']})\n")
        k = ablation["top_k"]
        L.append(f"| Strategy | Hit@1 | Hit@{k} | MRR | nDCG@{k} | Context precision | p50 latency |")
        L.append("|---|---|---|---|---|---|---|")
        for name, v in ablation["results"].items():
            L.append(f"| {name} | {_f(v['hit@1'], True)} | {_f(v[f'hit@{k}'], True)} | {_f(v['mrr'])} | "
                     f"{_f(v[f'ndcg@{k}'])} | {_f(v['context_precision'], True)} | {v['latency_p50_ms']} ms |")
        rt = ablation.get("routing_no_llm")
        if rt:
            L.append(f"Routing decision on all questions (no LLM involved): **{_f(rt['accuracy'], True)}** correct"
                     + (f"; mistakes: {json.dumps(rt['mistakes'])}" if rt["mistakes"] else "") + "\n")
        L.append("")
    if api_res:
        L.append("## Per question\n")
        L.append("| id | Expected → got | Top retrieved doc | Correct | Grounding | First token | Total | Answer (truncated) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in api_res["rows"]:
            ans = (r.get("answer") or r.get("error") or "").replace("\n", " ").replace("|", "/")[:110]
            top = (r.get("retrieved_docs") or ["–"])[0] if r.get("relevant_docs") else "–"
            corr = "–" if r["answer_correct"] is None else ("✅" if r["answer_correct"] else "❌")
            route = f"{r['expected_mode']} → {r['mode']}" + ("" if r["routing_correct"] else " ⚠️")
            L.append(f"| {r['id']} | {route} | {top} | {corr} | {_f(r.get('faithfulness'))} | "
                     f"{_f(r.get('ttft_s'))}s | {_f(r.get('total_s'))}s | {ans} |")
    return "\n".join(L) + "\n"
