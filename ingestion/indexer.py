"""
ingestion/indexer.py
--------------------
Vector store (ChromaDB) + keyword index (BM25) management.

* ChromaDB runs either embedded on disk (CHROMA_HOST empty — great for local
  runs and single-container cloud deploys) or as a server (CHROMA_HOST set,
  used by Docker Compose / Kubernetes).
* One shared client, collection and BM25 index per process (previously every
  request opened a new client and re-read + rebuilt BM25 from disk).
* All writes go through one lock, so parallel connector syncs can no longer
  overwrite each other's BM25 updates.
* Re-ingesting a document replaces its old chunks; deleting removes it from
  both indexes.
"""
from __future__ import annotations

import os
import pickle
import re
import threading
import time
from pathlib import Path
from typing import Optional

import structlog

from config import settings
from ingestion.chunker import DocumentChunk

logger = structlog.get_logger()

CHROMA_HOST = settings.chroma_host
CHROMA_PORT = settings.chroma_port
BASE_COLLECTION = settings.chroma_collection
BATCH_SIZE = 64

_write_lock = threading.RLock()
_client = None
_collection = None
_client_lock = threading.Lock()


def collection_name() -> str:
    from retrieval.embeddings import signature
    name = f"{BASE_COLLECTION}__{signature()}"
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)[:63].strip("-._")
    return name


COLLECTION_NAME = BASE_COLLECTION  # kept for backwards compatibility


def bm25_path() -> Path:
    p = Path(settings.bm25_index_path)
    return p.with_name(f"{p.stem}__{collection_name()}{p.suffix}")


# ── ChromaDB ──────────────────────────────────────────────────────────────

def _get_chroma_client():
    global _client
    with _client_lock:
        if _client is None:
            os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
            import logging
            logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            cs = ChromaSettings(anonymized_telemetry=False, allow_reset=False)
            if CHROMA_HOST:
                _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=cs)
                logger.info("chroma_connected", mode="http", host=CHROMA_HOST, port=CHROMA_PORT)
            else:
                path = settings.data_dir / "chroma"
                path.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(path), settings=cs)
                logger.info("chroma_connected", mode="embedded", path=str(path))
        return _client


def get_collection():
    global _collection
    if _collection is None:
        client = _get_chroma_client()
        with _client_lock:
            if _collection is None:
                _collection = client.get_or_create_collection(
                    name=collection_name(), metadata={"hnsw:space": "cosine"})
    return _collection


def reset_clients():
    """Drop cached clients (used after connection errors and in tests)."""
    global _client, _collection
    with _client_lock:
        _client, _collection = None, None


def ping_vector_db() -> bool:
    try:
        _get_chroma_client().heartbeat()
        return True
    except Exception:
        reset_clients()
        return False


# ── BM25 keyword index ────────────────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")
_STOP = frozenset("a an the of to in on for and or is are was were be been it this that with as at by "
                  "from what which who whom how why when where do does did can could should would "
                  "i you he she we they me my your our their its about into than then there".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _make_bm25(corpus):
    """BM25Okapi with Lucene's always-positive IDF.

    FIX: rank_bm25's default IDF is negative for terms found in more than half of the
    chunks — with one or two indexed chunks *every* term scored <= 0, so keyword
    search silently returned nothing for small knowledge bases.
    """
    import math
    from rank_bm25 import BM25Okapi

    class _LuceneBM25(BM25Okapi):
        def _calc_idf(self, nd):
            for word, freq in nd.items():
                self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    return _LuceneBM25(corpus)


class BM25Index:
    """rank_bm25.BM25Okapi with persistence, deletion and a document catalog."""

    def __init__(self):
        self.corpus_ids: list[str] = []
        self.corpus_texts: list[str] = []
        self.catalog: dict[str, dict] = {}     # doc_title → {chunk_ids, doc_type, source_path, …}
        self._bm25 = None
        self._lock = threading.RLock()

    def add(self, chunk_ids: list[str], texts: list[str]):
        with self._lock:
            existing = set(self.corpus_ids)
            for cid, t in zip(chunk_ids, texts):
                if cid not in existing:
                    self.corpus_ids.append(cid)
                    self.corpus_texts.append(t)
            self._bm25 = None

    def remove(self, chunk_ids):
        drop = set(chunk_ids)
        if not drop:
            return
        with self._lock:
            keep = [(i, t) for i, t in zip(self.corpus_ids, self.corpus_texts) if i not in drop]
            self.corpus_ids = [i for i, _ in keep]
            self.corpus_texts = [t for _, t in keep]
            self._bm25 = None

    def _build(self):
        self._bm25 = _make_bm25([tokenize(t) or ["_"] for t in self.corpus_texts])

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        with self._lock:
            if not self.corpus_ids:
                return []
            if self._bm25 is None:
                self._build()
            q = tokenize(query) or query.lower().split()
            scores = self._bm25.get_scores(q)
            ids = list(self.corpus_ids)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(ids[i], float(s)) for i, s in ranked if s > 0]

    def save(self, path: Path):
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"ids": self.corpus_ids, "texts": self.corpus_texts,
                             "catalog": self.catalog}, f)
            os.replace(tmp, path)                      # atomic — never a half-written index
        logger.info("bm25_saved", path=str(path), chunks=len(self.corpus_ids))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        idx = cls()
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                idx.corpus_ids = list(data.get("ids", []))
                idx.corpus_texts = list(data.get("texts", []))
                idx.catalog = dict(data.get("catalog", {}))
                if idx.corpus_ids:
                    idx._build()
                logger.info("bm25_loaded", path=str(path), chunks=len(idx.corpus_ids))
            except Exception as exc:
                logger.error("bm25_load_failed", path=str(path), error=str(exc)[:120])
        return idx


_bm25: Optional[BM25Index] = None
_bm25_mtime = 0.0
_bm25_checked = 0.0


def get_bm25() -> BM25Index:
    """Shared in-memory BM25 index. Reloads automatically (checked at most every 5 s) when
    another process — a second API replica or the Celery worker — rewrote the file."""
    global _bm25, _bm25_mtime, _bm25_checked
    now = time.monotonic()
    if _bm25 is not None and now - _bm25_checked < 5:
        return _bm25
    _bm25_checked = now
    path = bm25_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _bm25 is None or mtime > _bm25_mtime + 1e-6:
        with _write_lock:
            if _bm25 is None or mtime > _bm25_mtime + 1e-6:
                _bm25 = BM25Index.load(path)
                _bm25_mtime = mtime
    return _bm25


def _mark_saved():
    """Record our own write so it does not trigger a reload."""
    global _bm25_mtime
    try:
        _bm25_mtime = bm25_path().stat().st_mtime
    except OSError:
        pass


# ── Indexer facade ────────────────────────────────────────────────────────

class DocumentIndexer:
    def __init__(self):
        self.collection = get_collection()
        self.bm25_index = get_bm25()

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and index chunks. Re-indexing a document replaces its previous chunks."""
        if not chunks:
            return 0
        from retrieval.embeddings import embed_texts

        # Embed outside the lock (slowest part) so parallel ingestion overlaps.
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        with _write_lock:
            for title in {c.doc_title for c in chunks}:
                self._delete_title_locked(title, save=False)
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                metas = []
                for c in batch:
                    m = {k: v for k, v in c.to_dict().items()
                         if k != "text" and isinstance(v, (str, int, float, bool))}
                    metas.append(m)
                self.collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    embeddings=embeddings[i:i + BATCH_SIZE],
                    documents=[c.text for c in batch],
                    metadatas=metas,
                )
            self.bm25_index.add([c.chunk_id for c in chunks], texts)
            for title in {c.doc_title for c in chunks}:
                own = [c for c in chunks if c.doc_title == title]
                self.bm25_index.catalog[title] = {
                    "doc_title": title,
                    "doc_type": own[0].metadata.get("doc_type", ""),
                    "source_path": own[0].source_path,
                    "chunk_ids": [c.chunk_id for c in own],
                    "chunk_count": len(own),
                    "indexed_at": time.time(),
                }
            self.bm25_index.save(bm25_path())
            _mark_saved()
        logger.info("indexing_done", chunks=len(chunks))
        return len(chunks)

    def _delete_title_locked(self, doc_title: str, save: bool = True) -> int:
        entry = self.bm25_index.catalog.pop(doc_title, None)
        ids = list(entry["chunk_ids"]) if entry else []
        try:
            found = self.collection.get(where={"doc_title": doc_title}, include=[])
            ids = list(set(ids) | set(found.get("ids") or []))
        except Exception as exc:
            logger.warning("chroma_lookup_failed", doc=doc_title, error=str(exc)[:100])
        if ids:
            self.collection.delete(ids=ids)
            self.bm25_index.remove(ids)
        if save and (ids or entry):
            self.bm25_index.save(bm25_path())
            _mark_saved()
        return len(ids)

    def delete_document(self, doc_title: str) -> int:
        with _write_lock:
            n = self._delete_title_locked(doc_title)
        logger.info("document_deleted", doc=doc_title, chunks=n)
        return n

    def list_documents(self) -> list[dict]:
        docs = [{k: v for k, v in e.items() if k != "chunk_ids"}
                for e in self.bm25_index.catalog.values()]
        return sorted(docs, key=lambda d: d.get("indexed_at", 0), reverse=True)

    def count(self) -> int:
        return self.collection.count()

    def stats(self) -> dict:
        return {
            "total_chunks": self.collection.count(),
            "bm25_docs": len(self.bm25_index.corpus_ids),
            "documents": len(self.bm25_index.catalog),
            "collection": collection_name(),
        }


def index_is_empty() -> bool:
    """Cheap check used by the router (no network when BM25 catalog is loaded)."""
    try:
        bm = get_bm25()
        if bm.corpus_ids:
            return False
        return get_collection().count() == 0
    except Exception:
        return True


# ── Pipeline convenience ──────────────────────────────────────────────────

def ingest_file(file_path: str, chunk_size: int = settings.chunk_size,
                chunk_overlap: int = settings.chunk_overlap, title: Optional[str] = None,
                source_path: Optional[str] = None) -> dict:
    """parse → chunk → embed → index. `title` overrides the file-name-derived title."""
    from ingestion.chunker import chunk_document
    from ingestion.parser import parse_document

    doc = parse_document(file_path)
    if settings.embed_provider == "onnx":
        # MiniLM-style ONNX encoders only see the first 256 word-pieces of a chunk
        chunk_size = min(chunk_size, 200)
        chunk_overlap = min(chunk_overlap, 40)
    if title:
        doc.title = title
    if source_path:
        doc.source_path = source_path
    chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    indexed = DocumentIndexer().index_chunks(chunks)
    return {
        "file": source_path or file_path,
        "doc_title": doc.title,
        "pages": len(doc.pages),
        "chunks_created": len(chunks),
        "chunks_indexed": indexed,
        "parser": doc.metadata.get("parser", ""),
    }
