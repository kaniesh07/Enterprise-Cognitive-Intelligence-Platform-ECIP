"""Unit tests for the evaluation metrics (evaluation/evaluate.py)."""
import json
from pathlib import Path


def test_fact_matching_normalises_numbers_and_alternatives():
    from evaluation.evaluate import fact_present
    assert fact_present("The allowance is INR 2,500 per day.", "2500")
    assert fact_present("It is 4,80,000 rupees", "480000")
    assert not fact_present("Leave is 240 days", "24")          # whole numbers only
    assert fact_present("Report within 1 hour", "one hour|1 hour")
    assert fact_present("Growth was 3.1%", "3.1")
    assert not fact_present("Growth was 13.1%", "3.1")


def test_keyword_recall():
    from evaluation.evaluate import keyword_recall
    assert keyword_recall("Founded in 2019 with 42 staff", ["2019", "42"]) == 1.0
    assert keyword_recall("Founded in 2019", ["2019", "42"]) == 0.5
    assert keyword_recall("anything", []) is None


def test_rank_metrics():
    from evaluation.evaluate import rank_metrics
    m = rank_metrics(["A", "B", "A", "C"], {"B"}, 4)
    assert m["hit@1"] == 0.0 and m["hit@4"] == 1.0 and m["mrr"] == 0.5
    assert m["context_precision"] == 0.25
    assert rank_metrics(["B"], {"B"}, 4)["ndcg@4"] == 1.0


def test_dataset_is_well_formed():
    data = json.loads(Path("evaluation/datasets/eval_set.json").read_text())
    corpus = {p.stem for p in Path("evaluation/datasets/corpus").rglob("*") if p.is_file()}
    assert len(data) >= 30
    for it in data:
        assert it["expected_mode"] in ("grounded", "chat", "instant")
        for d in it["relevant_docs"]:
            assert d in corpus, f"{it['id']}: unknown doc {d}"
