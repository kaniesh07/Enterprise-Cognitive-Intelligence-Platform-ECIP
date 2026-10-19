"""
pipeline/sync_manifest.py
--------------------------
Incremental sync manifest — tracks which documents have been processed
and their fingerprint (source_id + last_modified hash).

Re-running a connector sync only processes NEW or CHANGED documents.
This is how a million-document index stays current without re-processing everything.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Optional
import structlog

logger = structlog.get_logger()

@dataclass
class ManifestEntry:
    source_id:    str
    source_type:  str
    fingerprint:  str      # SHA256 of source_id + last_modified
    indexed_at:   float
    chunk_count:  int
    status:       str      # "indexed" | "failed" | "skipped"
    error:        Optional[str] = None


class SyncManifest:
    """
    Persistent manifest stored as JSON.
    Production swap: replace with PostgreSQL table or Redis HSET.

    Key design: fingerprint-based change detection.
    If fingerprint matches → skip (document unchanged).
    If fingerprint differs → re-index (document updated).
    If source_id missing  → new document, index it.
    """
    def __init__(self, path: str = "./data/processed/sync_manifest.json"):
        self._path  = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock  = Lock()
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                return {}
        return {}

    def _save(self):
        self._path.write_text(json.dumps(self._data, indent=2))

    def needs_indexing(self, source_id: str, fingerprint: str) -> bool:
        """Return True if document is new or has changed since last index."""
        with self._lock:
            entry = self._data.get(source_id)
            if not entry:
                return True           # new document
            if entry.get("status") == "failed":
                return True           # retry failed documents
            return entry.get("fingerprint") != fingerprint  # changed

    def mark_indexed(self, source_id: str, source_type: str, fingerprint: str, chunk_count: int):
        with self._lock:
            self._data[source_id] = asdict(ManifestEntry(
                source_id=source_id, source_type=source_type,
                fingerprint=fingerprint, indexed_at=time.time(),
                chunk_count=chunk_count, status="indexed",
            ))
            self._save()

    def mark_failed(self, source_id: str, source_type: str, fingerprint: str, error: str):
        with self._lock:
            self._data[source_id] = asdict(ManifestEntry(
                source_id=source_id, source_type=source_type,
                fingerprint=fingerprint, indexed_at=time.time(),
                chunk_count=0, status="failed", error=error,
            ))
            self._save()

    def mark_skipped(self, source_id: str):
        with self._lock:
            if source_id in self._data:
                self._data[source_id]["status"] = "skipped"
            self._save()

    def stats(self) -> dict:
        with self._lock:
            statuses = [e.get("status", "unknown") for e in self._data.values()]
            total_chunks = sum(e.get("chunk_count", 0) for e in self._data.values())
            return {
                "total_documents": len(self._data),
                "indexed":  statuses.count("indexed"),
                "failed":   statuses.count("failed"),
                "skipped":  statuses.count("skipped"),
                "total_chunks_indexed": total_chunks,
            }

    def get_failed(self) -> list[dict]:
        with self._lock:
            return [
                {"source_id": sid, **entry}
                for sid, entry in self._data.items()
                if entry.get("status") == "failed"
            ]

    def reset_failed(self):
        """Mark all failed entries for retry on next sync."""
        with self._lock:
            for entry in self._data.values():
                if entry.get("status") == "failed":
                    del entry["fingerprint"]   # force re-index
            self._save()


_manifest: Optional[SyncManifest] = None

def get_manifest() -> SyncManifest:
    global _manifest
    if _manifest is None:
        from config import settings
        _manifest = SyncManifest(str(settings.manifest_path))
    return _manifest
