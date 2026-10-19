"""
tests/test_pipeline.py — Comprehensive ECIP unit tests (no live services needed)
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 1. PARSER ──────────────────────────────────────────────────────────────
class TestParser:
    def test_parse_text_file(self, sample_text_doc):
        from ingestion.parser import parse_document
        doc = parse_document(sample_text_doc)
        assert doc.title == "refund_policy"
        assert doc.doc_type == "txt"
        assert "30 days" in doc.full_text
        assert len(doc.pages) >= 1

    def test_parse_nonexistent_raises(self):
        from ingestion.parser import parse_document
        with pytest.raises(FileNotFoundError):
            parse_document("/no/such/file.pdf")

    def test_full_text_non_empty(self, sample_text_doc):
        from ingestion.parser import parse_document
        doc = parse_document(sample_text_doc)
        assert len(doc.full_text) > 50

    def test_page_numbers_start_at_1(self, sample_text_doc):
        from ingestion.parser import parse_document
        doc = parse_document(sample_text_doc)
        for page in doc.pages:
            assert page.page_number >= 1

# ── 2. CHUNKER ─────────────────────────────────────────────────────────────
class TestTableSerialiser:
    def test_basic_table(self):
        from ingestion.chunker import _table_to_text
        out = _table_to_text([["Name","Value"],["Alice","100"]])
        assert "Alice" in out and "|" in out and "---" in out

    def test_empty_table(self):
        from ingestion.chunker import _table_to_text
        assert _table_to_text([]) == ""

class TestSentenceSplitter:
    def test_splits_sentences(self):
        from ingestion.chunker import _split_sentences
        sents = _split_sentences("Hello world. This is a test. Third sentence.")
        assert len(sents) >= 2

    def test_handles_empty(self):
        from ingestion.chunker import _split_sentences
        assert _split_sentences("") == []

class TestSemanticChunker:
    def test_basic_chunking(self, sample_text_doc):
        from ingestion.parser import parse_document
        from ingestion.chunker import SemanticChunker
        doc = parse_document(sample_text_doc)
        chunks = SemanticChunker(chunk_size=50, chunk_overlap=10).chunk_document(doc)
        assert len(chunks) >= 1

    def test_unique_chunk_ids(self, sample_text_doc):
        from ingestion.parser import parse_document
        from ingestion.chunker import SemanticChunker
        doc = parse_document(sample_text_doc)
        chunks = SemanticChunker().chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata(self, sample_text_doc):
        from ingestion.parser import parse_document
        from ingestion.chunker import SemanticChunker
        doc = parse_document(sample_text_doc)
        for chunk in SemanticChunker().chunk_document(doc):
            assert chunk.doc_title == doc.title
            assert chunk.page_number >= 1
            assert chunk.text.strip()

    def test_table_chunks(self):
        from ingestion.parser import ParsedDocument, ParsedPage
        from ingestion.chunker import SemanticChunker
        doc = ParsedDocument(
            source_path="t.pdf", doc_type="pdf", title="t",
            pages=[ParsedPage(page_number=1, text="Some text.",
                tables=[[["Col A","Col B"],["1","2"],["3","4"]]])]
        )
        chunks = SemanticChunker().chunk_document(doc)
        table_chunks = [c for c in chunks if c.layout_type == "table"]
        assert len(table_chunks) >= 1
        assert "Col A" in table_chunks[0].text

    def test_empty_doc_no_chunks(self):
        from ingestion.parser import ParsedDocument, ParsedPage
        from ingestion.chunker import SemanticChunker
        doc = ParsedDocument(source_path="e.txt", doc_type="txt", title="e",
                             pages=[ParsedPage(page_number=1, text="")])
        assert SemanticChunker().chunk_document(doc) == []

    def test_to_dict_has_required_keys(self, sample_text_doc):
        from ingestion.parser import parse_document
        from ingestion.chunker import SemanticChunker
        doc = parse_document(sample_text_doc)
        chunks = SemanticChunker().chunk_document(doc)
        d = chunks[0].to_dict()
        for key in ["chunk_id", "text", "doc_title", "page_number", "layout_type"]:
            assert key in d

# ── 3. BM25 ────────────────────────────────────────────────────────────────
class TestBM25Index:
    def test_search_finds_relevant(self):
        from ingestion.indexer import BM25Index
        idx = BM25Index()
        idx.add(["id1","id2","id3"], [
            "return policy 30 days receipt refund",
            "safety goggles boots protective equipment",
            "emergency exit evacuation fire alarm",
        ])
        results = idx.search("return refund", top_k=3)
        assert results[0][0] == "id1" and results[0][1] > 0

    def test_empty_index(self):
        from ingestion.indexer import BM25Index
        assert BM25Index().search("query") == []

    def test_persistence(self, tmp_path):
        from ingestion.indexer import BM25Index
        # BM25 IDF needs a reasonably sized corpus; with only 2 docs and a term
        # in 1 of them, IDF can round to 0. Use a larger corpus where the
        # target doc uniquely contains the query term.
        idx = BM25Index()
        ids   = ["target", "d2", "d3", "d4", "d5"]
        texts = [
            "return policy refund thirty days receipt purchase",
            "safety equipment goggles boots protection",
            "emergency evacuation exit procedure alarm",
            "invoice payment billing accounts receivable",
            "shipping logistics warehouse distribution centre",
        ]
        idx.add(ids, texts)
        path = tmp_path / "bm25.pkl"
        idx.save(path)
        loaded = BM25Index.load(path)
        assert loaded.corpus_ids == ids
        results = loaded.search("return refund", top_k=1)
        assert len(results) == 1
        assert results[0][0] == "target"
        assert results[0][1] > 0

    def test_top_k_respected(self):
        from ingestion.indexer import BM25Index
        idx = BM25Index()
        idx.add([f"d{i}" for i in range(10)], [f"document topic {i}" for i in range(10)])
        assert len(idx.search("document", top_k=3)) <= 3

    def test_zero_score_excluded(self):
        from ingestion.indexer import BM25Index
        idx = BM25Index()
        idx.add(["match","nomatch"], ["cats dogs animals pets","xyz qqq zzz aaa"])
        for _, score in idx.search("cats animals", top_k=5):
            assert score > 0

# ── 4. RETRIEVAL METRICS ───────────────────────────────────────────────────
class TestRetrievalMetrics:
    def test_precision_at_k(self):
        from evaluation.ragas_eval import precision_at_k
        assert precision_at_k(["a","b","c","d","e"], ["a","c"], k=5) == pytest.approx(2/5)
        assert precision_at_k(["a"], ["a"], k=1) == pytest.approx(1.0)
        assert precision_at_k(["z"], ["a"], k=1) == pytest.approx(0.0)

    def test_recall_at_k(self):
        from evaluation.ragas_eval import recall_at_k
        assert recall_at_k(["a","b","c"], ["a","c","z"], k=3) == pytest.approx(2/3)
        assert recall_at_k(["a","b"], ["a","b"], k=2) == pytest.approx(1.0)

    def test_mrr(self):
        from evaluation.ragas_eval import mrr
        assert mrr(["a","b","c"], ["a"]) == pytest.approx(1.0)
        assert mrr(["x","a","b"], ["a"]) == pytest.approx(0.5)
        assert mrr(["a","b"], ["z"]) == 0.0

    def test_ndcg_perfect(self):
        from evaluation.ragas_eval import ndcg_at_k
        assert ndcg_at_k(["a","b","c"], ["a","b","c"], k=3) == pytest.approx(1.0)

    def test_ndcg_zero(self):
        from evaluation.ragas_eval import ndcg_at_k
        assert ndcg_at_k(["z","y","x"], ["a","b","c"], k=3) == pytest.approx(0.0)

    def test_ndcg_partial(self):
        from evaluation.ragas_eval import ndcg_at_k
        score = ndcg_at_k(["a","z","b"], ["a","b"], k=3)
        assert 0 < score < 1.0

    def test_ndcg_large_k(self):
        from evaluation.ragas_eval import ndcg_at_k
        score = ndcg_at_k(["a","b"], ["a","b"], k=100)
        assert 0 <= score <= 1.0

# ── 5. RRF FUSION ──────────────────────────────────────────────────────────
class TestRRFFusion:
    def _chunks(self, ids):
        from retrieval.retriever import RetrievedChunk
        return [RetrievedChunk(chunk_id=cid, text=f"text {cid}",
            doc_title="D", source_path="/t.pdf", page_number=1, layout_type="text")
            for cid in ids]

    def test_overlap_ranks_higher(self):
        from retrieval.retriever import _rrf_fuse
        fused = _rrf_fuse(self._chunks(["a","b","c"]), self._chunks(["b","a","d"]))
        top2 = {c.chunk_id for c in fused[:2]}
        assert top2 == {"a","b"}

    def test_all_included(self):
        from retrieval.retriever import _rrf_fuse
        fused = _rrf_fuse(self._chunks(["a","b"]), self._chunks(["c","d"]))
        assert {c.chunk_id for c in fused} == {"a","b","c","d"}

    def test_scores_positive(self):
        from retrieval.retriever import _rrf_fuse
        for c in _rrf_fuse(self._chunks(["x","y"]), self._chunks(["y","z"])):
            assert c.rrf_score > 0

    def test_descending_order(self):
        from retrieval.retriever import _rrf_fuse
        fused = _rrf_fuse(self._chunks(["a","b","c"]), self._chunks(["a","b","d"]))
        scores = [c.rrf_score for c in fused]
        assert scores == sorted(scores, reverse=True)

# ── 6. AGENT STATE ─────────────────────────────────────────────────────────
class TestAgentState:
    def test_minimal_state(self):
        from agents.state import AgentState
        state: AgentState = {"original_query": "test?",
                              "conversation_history": [], "latency_ms": {}}
        assert state["original_query"] == "test?"

    def test_full_state(self):
        from agents.state import AgentState
        state: AgentState = {
            "original_query": "q", "conversation_history": [],
            "intent": "factual", "expanded_queries": ["q"], "language": "en",
            "retrieved_chunks": [], "retrieval_metrics": {},
            "multimodal_context": [], "reasoning_steps": ["step 1"],
            "draft_answer": "answer", "faithfulness_score": 0.9,
            "hallucination_flags": [], "final_answer": "answer",
            "citations": ["[Doc, p.1]"], "confidence": 0.85,
            "latency_ms": {"total": 1200}, "error": None,
        }
        assert state["faithfulness_score"] == 0.9

# ── 7. API MODELS ──────────────────────────────────────────────────────────
class TestAPIModels:
    def test_defaults(self):
        from api.main import QueryRequest
        req = QueryRequest(query="What is the refund policy?")
        assert req.top_k is None and not req.stream and req.conversation_history == []
        assert req.mode == "auto"

    def test_short_query_allowed(self):
        """Short queries like 'hi', 'ok' are valid — min_length is 1 not 3."""
        from api.main import QueryRequest
        q = QueryRequest(query="hi")
        assert q.query == "hi"
        q2 = QueryRequest(query="ok")
        assert q2.query == "ok"

    def test_empty_query_rejected(self):
        """Empty string is still rejected (min_length=1)."""
        from api.main import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_top_k_bounds(self):
        from api.main import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError): QueryRequest(query="valid query", top_k=0)
        with pytest.raises(ValidationError): QueryRequest(query="valid query", top_k=21)

    def test_ingest_response(self):
        from api.main import IngestResponse
        r = IngestResponse(file="/tmp/t.pdf", doc_title="t",
                           pages=3, chunks_created=10, chunks_indexed=9, duration_ms=500)
        assert r.chunks_indexed == 9

# ── 8. LLM CLIENT (mocked) ────────────────────────────────────────────────
class TestLLMClient:
    def test_json_strips_markdown_fences(self):
        from agents.llm_client import chat_json
        for raw in ['```json\n{"k":1}\n```', '{"k":1}', '```\n{"k":1}\n```']:
            with patch("agents.llm_client.chat", return_value=raw):
                assert chat_json([{"role":"user","content":"x"}]).get("k") == 1

    def test_embed_returns_list(self):
        from agents.llm_client import embed
        with patch("retrieval.embeddings._ollama", return_value=[[0.1, 0.2, 0.3]]), \
             patch("retrieval.embeddings.PROVIDER", "ollama"):
            result = embed("text")
            assert isinstance(result, list) and len(result) == 3

    def test_ollama_down_returns_false(self):
        from agents.llm_client import is_ollama_running
        with patch("httpx.get", side_effect=Exception("refused")):
            assert is_ollama_running() is False

# ── 9. CONFIG ──────────────────────────────────────────────────────────────
class TestConfig:
    def test_loads_defaults(self):
        from config import get_settings
        s = get_settings()
        assert s.llm_model == "llama3.2:3b"
        assert s.embed_model == "nomic-embed-text"
        assert s.chunk_size == 300
        assert s.top_k_final == 4
        assert s.chroma_host == ""          # embedded Chroma by default (no server needed)

    def test_ensure_dirs(self, tmp_path):
        from config import Settings
        s = Settings(DATA_DIR=str(tmp_path / "data"))
        s.ensure_dirs()
        assert (tmp_path / "data" / "raw").exists()
        assert (tmp_path / "data" / "processed").exists()
