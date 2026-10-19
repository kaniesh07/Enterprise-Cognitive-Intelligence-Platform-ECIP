"""
retrieval/retriever.py
----------------------
Hybrid retrieval: dense vectors (ChromaDB) + keywords (BM25), fused with
Reciprocal Rank Fusion, optionally reranked with a cross-encoder.

Latency notes
* Shared Chroma client / collection / in-memory BM25 (no per-query disk I/O).
* Dense and BM25 search run concurrently.
* Query embeddings are cached.
* Cross-encoder reranking is opt-in (RERANK_ENABLED=true) and preloaded at start-up.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import structlog

from config import settings

logger = structlog.get_logger()

TOP_K_DENSE = settings.top_k_dense
TOP_K_BM25 = settings.top_k_bm25
TOP_K_FINAL = settings.top_k_final
RRF_K = 60

_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieval")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_title: str
    source_path: str
    page_number: int
    layout_type: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score

    def citation(self) -> str:
        return f"[{self.doc_title}, p.{self.page_number}]"

    def to_dict(self, snippet_chars: int = 300) -> dict:
        return {
            "chunk_id": self.chunk_id, "doc_title": self.doc_title,
            "page_number": self.page_number, "score": round(self.final_score, 4),
            "dense_score": None if self.dense_score is None else round(self.dense_score, 4),
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 4),
            "snippet": self.text[:snippet_chars], "source_path": self.source_path,
            "layout_type": self.layout_type,
        }


def _to_chunk(cid: str, doc: str, meta: dict, **scores) -> RetrievedChunk:
    meta = meta or {}
    return RetrievedChunk(
        chunk_id=cid, text=doc or "",
        doc_title=meta.get("doc_title", ""), source_path=meta.get("source_path", ""),
        page_number=int(meta.get("page_number", 0) or 0),
        layout_type=meta.get("layout_type", "text"), metadata=meta, **scores,
    )


# ── Dense retrieval ───────────────────────────────────────────────────────

def _embed_query(query: str) -> list[float]:
    from retrieval.embeddings import embed_query
    return embed_query(query)


def _dense_search(query_embedding: list[float], top_k: int,
                  where: Optional[dict] = None) -> list[RetrievedChunk]:
    from ingestion.indexer import get_collection
    collection = get_collection()
    n = collection.count()
    if n == 0:
        return []
    kwargs = dict(query_embeddings=[query_embedding], n_results=min(top_k, n),
                  include=["documents", "metadatas", "distances"])
    if where:
        kwargs["where"] = where
    res = collection.query(**kwargs)
    return [
        _to_chunk(cid, doc, meta, dense_score=float(1.0 - dist))      # cosine similarity
        for doc, meta, dist, cid in zip(res["documents"][0], res["metadatas"][0],
                                        res["distances"][0], res["ids"][0])
    ]


# ── BM25 retrieval ────────────────────────────────────────────────────────

def _bm25_search(query: str, top_k: int) -> list[RetrievedChunk]:
    from ingestion.indexer import get_bm25, get_collection
    hits = get_bm25().search(query, top_k=top_k)
    if not hits:
        return []
    score_map = dict(hits)
    try:
        res = get_collection().get(ids=[h[0] for h in hits], include=["documents", "metadatas"])
    except Exception as exc:
        logger.warning("bm25_fetch_failed", error=str(exc)[:100])
        return []
    by_id = {cid: (doc, meta) for doc, meta, cid in
             zip(res["documents"], res["metadatas"], res["ids"])}
    # keep BM25 rank order
    return [_to_chunk(cid, by_id[cid][0], by_id[cid][1], bm25_score=score_map[cid])
            for cid, _ in hits if cid in by_id]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────

def _rrf_fuse(dense: list[RetrievedChunk], bm25: list[RetrievedChunk],
              k: int = RRF_K) -> list[RetrievedChunk]:
    chunk_map: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for rank, chunk in enumerate(dense, start=1):
        chunk_map[chunk.chunk_id] = chunk
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
    for rank, chunk in enumerate(bm25, start=1):
        if chunk.chunk_id not in chunk_map:
            chunk_map[chunk.chunk_id] = chunk
        else:
            chunk_map[chunk.chunk_id].bm25_score = chunk.bm25_score
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
    for cid, score in scores.items():
        chunk_map[cid].rrf_score = score
    return sorted(chunk_map.values(), key=lambda c: c.rrf_score, reverse=True)


# ── Cross-encoder reranker (optional) ─────────────────────────────────────

_reranker_model = None


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _reranker_model


def preload_reranker() -> bool:
    if not settings.rerank_enabled:
        return False
    try:
        _get_reranker()
        return True
    except Exception as exc:
        logger.warning("reranker_unavailable", error=str(exc)[:120])
        return False


def _rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return chunks
    try:
        scores = _get_reranker().predict([(query, c.text) for c in chunks])
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
        return sorted(chunks, key=lambda c: c.rerank_score or 0, reverse=True)
    except Exception as exc:
        logger.warning("reranker_failed", error=str(exc)[:120])
        return chunks


# ── Public interface ──────────────────────────────────────────────────────

class HybridRetriever:
    def retrieve(
        self,
        query: str,
        top_k_dense: int = TOP_K_DENSE,
        top_k_bm25: int = TOP_K_BM25,
        top_k_final: int = TOP_K_FINAL,
        filter_doc: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> list[RetrievedChunk]:
        t0 = time.perf_counter()
        where = {"doc_title": filter_doc} if filter_doc else None

        def dense():
            return _dense_search(_embed_query(query), top_k_dense, where=where)

        f_dense = _pool.submit(dense)
        f_bm25 = _pool.submit(_bm25_search, query, top_k_bm25)
        dense_results = f_dense.result()
        bm25_results = [c for c in f_bm25.result()
                        if not filter_doc or c.doc_title == filter_doc]

        fused = _rrf_fuse(dense_results, bm25_results)
        candidates = fused[:max(top_k_final * 3, 10)]
        if rerank if rerank is not None else settings.rerank_enabled:
            candidates = _rerank(query, candidates)
        final = candidates[:top_k_final]
        logger.info("retrieval_done", dense=len(dense_results), bm25=len(bm25_results),
                    returned=len(final), ms=int((time.perf_counter() - t0) * 1000))
        return final

    def get_metrics(self, query: str, retrieved: list[RetrievedChunk], relevant_ids: list[str]) -> dict:
        """Compute Recall@K, Precision@K, MRR for a known-relevant set."""
        retrieved_ids = [c.chunk_id for c in retrieved]
        k = len(retrieved_ids)
        relevant_set = set(relevant_ids)
        hits = [cid for cid in retrieved_ids if cid in relevant_set]
        mrr = next((1.0 / r for r, cid in enumerate(retrieved_ids, 1) if cid in relevant_set), 0.0)
        return {
            "precision_at_k": len(hits) / k if k else 0,
            "recall_at_k": len(hits) / len(relevant_set) if relevant_set else 0,
            "mrr": mrr, "k": k,
        }
