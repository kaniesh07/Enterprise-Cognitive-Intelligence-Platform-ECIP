#!/usr/bin/env python3
"""
Evaluate ECIP end-to-end.

  # 1) against a running server (local, Docker or cloud); uploads the eval corpus first
  python scripts/run_eval.py --api http://localhost:8000 --key ecip-dev-key --ingest

  # 2) retrieval ablation only (dense vs BM25 vs hybrid) — in-process, temporary index, no LLM
  python scripts/run_eval.py --ablation-only

Results: evaluation/results/eval_<timestamp>.json and .md (and latest.json / latest.md)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=os.getenv("ECIP_API", "http://localhost:8000"))
    ap.add_argument("--key", default=os.getenv("ECIP_API_KEY", "ecip-dev-key"))
    ap.add_argument("--dataset", default=str(ROOT / "evaluation/datasets/eval_set.json"))
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--ingest", action="store_true", help="upload evaluation/datasets/corpus first")
    ap.add_argument("--no-ablation", action="store_true", help="skip the in-process retrieval ablation")
    ap.add_argument("--ablation-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "evaluation/results"))
    args = ap.parse_args()

    run_ablation = not args.no_ablation
    if run_ablation:
        # isolate the ablation index from any real data (must happen before ECIP imports)
        os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="ecip_eval_")
        os.environ["CHROMA_HOST"] = ""
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    os.environ.setdefault("LOG_FORMAT", "console")

    from monitoring.logging_setup import configure_logging
    configure_logging("WARNING")
    from evaluation.evaluate import (ingest_corpus, run_api_evaluation, run_retrieval_ablation,
                                     to_markdown)

    api_res = None
    if not args.ablation_only:
        if args.ingest:
            print("Uploading evaluation corpus …")
            for r in ingest_corpus(args.api, args.key):
                print(f"  {r['file']:<35} HTTP {r['status']}  chunks={r.get('chunks_indexed')}")
        print(f"Evaluating {args.api} …")
        api_res = run_api_evaluation(args.api, args.key, Path(args.dataset), args.top_k, args.limit)

    ablation = None
    if run_ablation:
        print("Running retrieval ablation (temporary index) …")
        ablation = run_retrieval_ablation(Path(args.dataset), top_k=args.top_k)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload = {"api_evaluation": api_res, "retrieval_ablation": ablation}
    md = to_markdown(api_res, ablation)
    for name in (f"eval_{stamp}", "latest"):
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
        (out / f"{name}.md").write_text(md)
    print("\n" + md.split("## Per question")[0])
    print(f"Saved: {out / f'eval_{stamp}.md'}")


if __name__ == "__main__":
    main()
