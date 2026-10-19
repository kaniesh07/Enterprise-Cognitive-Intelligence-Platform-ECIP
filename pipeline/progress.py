"""
pipeline/progress.py
---------------------
Real-time progress tracking for large connector sync jobs.
Provides: documents processed, failed, skipped, ETA, throughput.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional
import structlog

logger = structlog.get_logger()


@dataclass
class SyncJobProgress:
    job_id:        str
    connector_name: str
    status:        str = "running"   # running | completed | failed | cancelled
    total_discovered: int = 0
    processed:     int = 0
    indexed:       int = 0
    skipped:       int = 0
    failed:        int = 0
    total_chunks:  int = 0
    started_at:    float = field(default_factory=time.time)
    finished_at:   Optional[float] = None
    current_doc:   str = ""
    errors:        list[dict] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def throughput_docs_per_min(self) -> float:
        elapsed = self.elapsed_s
        if elapsed < 1 or self.processed == 0:
            return 0.0
        return round(self.processed / elapsed * 60, 1)

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.total_discovered == 0 or self.processed == 0:
            return None
        elapsed = self.elapsed_s
        rate    = self.processed / elapsed           # docs/second
        remaining = self.total_discovered - self.processed
        return remaining / rate if rate > 0 else None

    @property
    def progress_pct(self) -> float:
        if self.total_discovered == 0:
            return 0.0
        return round(self.processed / self.total_discovered * 100, 1)

    def to_dict(self) -> dict:
        eta = self.eta_seconds
        return {
            "job_id":           self.job_id,
            "connector_name":   self.connector_name,
            "status":           self.status,
            "progress_pct":     self.progress_pct,
            "total_discovered": self.total_discovered,
            "processed":        self.processed,
            "indexed":          self.indexed,
            "skipped":          self.skipped,
            "failed":           self.failed,
            "total_chunks":     self.total_chunks,
            "elapsed_s":        round(self.elapsed_s, 1),
            "throughput_docs_per_min": self.throughput_docs_per_min,
            "eta_seconds":      round(eta, 0) if eta else None,
            "current_doc":      self.current_doc,
            "recent_errors":    self.errors[-5:],   # last 5 errors
        }


class ProgressTracker:
    """Thread-safe progress tracker for all active and completed sync jobs."""
    def __init__(self):
        self._jobs: dict[str, SyncJobProgress] = {}
        self._lock = Lock()

    def create_job(self, job_id: str, connector_name: str) -> SyncJobProgress:
        with self._lock:
            job = SyncJobProgress(job_id=job_id, connector_name=connector_name)
            self._jobs[job_id] = job
            return job

    def update(self, job_id: str, **kwargs):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    if hasattr(job, k):
                        setattr(job, k, v)

    def increment(self, job_id: str, **deltas):
        """Atomically add to numeric fields (safe with many worker threads)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, d in deltas.items():
                    if hasattr(job, k):
                        setattr(job, k, getattr(job, k) + d)

    def add_error(self, job_id: str, source_id: str, error: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.errors.append({"source_id": source_id, "error": error, "ts": time.time()})
                job.failed += 1

    def complete_job(self, job_id: str, status: str = "completed"):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status      = status
                job.finished_at = time.time()
                logger.info("sync_job_done", job_id=job_id, status=status,
                            indexed=job.indexed, failed=job.failed,
                            elapsed_s=round(job.elapsed_s, 1))

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list_all(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def list_active(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values() if j.status == "running"]


_tracker: Optional[ProgressTracker] = None

def get_progress_tracker() -> ProgressTracker:
    global _tracker
    if _tracker is None:
        _tracker = ProgressTracker()
    return _tracker
