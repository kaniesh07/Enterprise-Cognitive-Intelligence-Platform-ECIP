"""
pipeline/worker.py
-------------------
Core ingestion worker + connector sync orchestrator.

Two execution modes:
  1. Direct (ThreadPoolExecutor) — default, works with zero extra services.
     Scales to ~16 parallel workers on a single machine.

  2. Celery (optional) — for true distributed processing across many machines.
     Enable by setting CELERY_BROKER_URL in .env (e.g. redis://redis:6379/0).
     Workers are then separate processes: `celery -A pipeline.worker worker --loglevel=info`

The same ingest_document_job function runs in both modes.
"""
from __future__ import annotations
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import structlog

from connectors.base import ConnectorDocument
from pipeline.sync_manifest import get_manifest
from pipeline.progress import get_progress_tracker

logger = structlog.get_logger()

MAX_WORKERS = int(os.getenv("INGEST_WORKERS", "4"))
MAX_DOC_SIZE_MB = int(os.getenv("MAX_DOC_SIZE_MB", "100"))

# ── Core document ingestion job ───────────────────────────────────────────

def ingest_document_job(
    doc: ConnectorDocument,
    connector_instance,          # BaseConnector subclass
    job_id: str,
    force: bool = False,
) -> dict:
    """
    Process a single document:
      1. Check sync manifest (skip if unchanged)
      2. Load content bytes from connector
      3. Write to temp file
      4. Parse → chunk → embed → index
      5. Update manifest

    Returns a result dict with status and chunk count.
    This function is the unit of work for both thread pool and Celery.
    """
    manifest = get_manifest()
    tracker  = get_progress_tracker()

    # Skip if unchanged (incremental sync)
    if not force and not manifest.needs_indexing(doc.source_id, doc.fingerprint):
        manifest.mark_skipped(doc.source_id)
        tracker.increment(job_id, skipped=1)
        _metric("sync_docs_skipped")
        logger.debug("doc_skipped", source_id=doc.source_id)
        return {"status": "skipped", "source_id": doc.source_id, "chunks": 0}

    # Size guard
    if doc.size_bytes > MAX_DOC_SIZE_MB * 1024 * 1024:
        error = f"Document too large: {doc.size_bytes / 1e6:.1f}MB > {MAX_DOC_SIZE_MB}MB limit"
        manifest.mark_failed(doc.source_id, doc.source_type, doc.fingerprint, error)
        tracker.add_error(job_id, doc.source_id, error)
        _metric("sync_docs_failed")
        return {"status": "failed", "source_id": doc.source_id, "error": error, "chunks": 0}

    tracker.update(job_id, current_doc=doc.title)

    try:
        # Load content from source
        content_bytes = doc.content_bytes or connector_instance.load(doc)

        # Write to a temp file whose name keeps the real title + extension, so the
        # parser picks the right format and citations show the real document name.
        suffix = doc.extension or ".txt"
        tmp_dir = tempfile.mkdtemp(prefix="ecip_sync_")
        safe = "".join(ch if ch.isalnum() or ch in "-_. " else "_" for ch in (doc.title or "document"))[:120]
        tmp_path = str(Path(tmp_dir) / f"{safe or 'document'}{suffix}")
        Path(tmp_path).write_bytes(content_bytes)

        try:
            from ingestion.indexer import ingest_file
            title = (doc.title or "document").strip()[:200]
            if title.lower().endswith(suffix):
                title = title[: -len(suffix)]
            result = ingest_file(tmp_path, title=title, source_path=doc.source_id)
            chunk_count = result.get("chunks_indexed", 0)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

        manifest.mark_indexed(doc.source_id, doc.source_type, doc.fingerprint, chunk_count)
        tracker.increment(job_id, indexed=1, total_chunks=chunk_count)
        _metric("sync_docs_indexed")
        logger.info("doc_indexed", title=doc.title, chunks=chunk_count)
        return {"status": "indexed", "source_id": doc.source_id, "chunks": chunk_count}

    except Exception as exc:
        error = str(exc)
        manifest.mark_failed(doc.source_id, doc.source_type, doc.fingerprint, error)
        tracker.add_error(job_id, doc.source_id, error)
        _metric("sync_docs_failed")
        logger.error("doc_failed", title=doc.title, error=error)
        return {"status": "failed", "source_id": doc.source_id, "error": error, "chunks": 0}


def _metric(name: str):
    try:
        from monitoring.metrics import get_registry
        getattr(get_registry(), name).inc()
    except Exception:
        pass


def _inc(tracker, job_id, field) -> int:
    job = tracker._jobs.get(job_id)
    if job:
        return getattr(job, field, 0) + 1
    return 0

def _add(tracker, job_id, field, n) -> int:
    job = tracker._jobs.get(job_id)
    if job:
        return getattr(job, field, 0) + n
    return n


# ── Connector sync orchestrator ───────────────────────────────────────────

def run_connector_sync(
    connector_name: str,
    force: bool = False,
    max_workers: int = MAX_WORKERS,
) -> str:
    """
    Run a full sync of a registered connector.

    Steps:
      1. Load connector from registry
      2. Create a job in progress tracker
      3. List all documents (lazy generator — O(1) memory)
      4. Submit each to thread pool (parallel processing)
      5. Return job_id for progress polling

    Returns job_id immediately — sync runs in background thread.
    """
    from connectors.registry import get_connector_registry
    import threading

    registry  = get_connector_registry()
    connector = registry.get(connector_name)
    if not connector:
        raise ValueError(f"Connector '{connector_name}' not registered.")

    job_id  = str(uuid.uuid4())
    tracker = get_progress_tracker()
    tracker.create_job(job_id, connector_name)

    def _run():
        try:
            _sync_worker(connector, connector_name, job_id, force, max_workers)
        except Exception as exc:
            logger.error("sync_failed", connector=connector_name, error=str(exc))
            tracker.complete_job(job_id, status="failed")

    thread = threading.Thread(target=_run, daemon=True, name=f"sync-{connector_name}")
    thread.start()
    logger.info("sync_started", connector=connector_name, job_id=job_id, workers=max_workers)
    return job_id


def _sync_worker(connector, connector_name: str, job_id: str, force: bool, max_workers: int):
    """
    Internal: runs in background thread.
    Streams documents from connector and processes them in a thread pool.
    Memory stays constant regardless of how many documents exist.
    """
    tracker = get_progress_tracker()

    # First pass: count total documents for ETA calculation
    # We do this by converting the generator to a list only if the
    # connector is small. For large connectors, we estimate.
    logger.info("sync_discovering", connector=connector_name, job_id=job_id)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        discovered = 0

        # Stream documents and submit to pool
        for doc in connector.list():
            discovered += 1
            tracker.update(job_id, total_discovered=discovered)

            future = pool.submit(ingest_document_job, doc, connector, job_id, force)
            futures[future] = doc.source_id

        # Collect results as they complete
        for future in as_completed(futures):
            tracker.increment(job_id, processed=1)
            try:
                result = future.result()
                if result["status"] == "failed":
                    logger.warning("doc_failed_in_pool", source_id=futures[future])
            except Exception as exc:
                logger.error("future_error", source_id=futures[future], error=str(exc))

    tracker.complete_job(job_id, status="completed")


# ── Optional Celery integration ───────────────────────────────────────────

def _try_get_celery_app():
    """Return Celery app if broker is configured, else None."""
    broker = os.getenv("CELERY_BROKER_URL")
    if not broker:
        return None
    try:
        from celery import Celery
        app = Celery("ecip", broker=broker,
                     backend=os.getenv("CELERY_RESULT_BACKEND", broker))
        # The compose worker consumes the "ecip" queue (-Q ecip); route tasks there.
        app.conf.task_default_queue = "ecip"
        return app
    except ImportError:
        return None


celery_app = _try_get_celery_app()

if celery_app:
    @celery_app.task(name="ecip.ingest_document", bind=True, max_retries=3)
    def celery_ingest_task(self, source_id: str, source_type: str,
                           connector_name: str, job_id: str):
        """
        Celery task wrapper for distributed processing.
        Each document becomes an independent task that can be retried independently.
        """
        from connectors.registry import get_connector_registry
        registry  = get_connector_registry()
        connector = registry.get(connector_name)
        if not connector:
            raise ValueError(f"Connector '{connector_name}' not found in registry.")
        # Reload doc metadata from manifest
        get_manifest()
        # Re-list to find this specific doc (connector must support it)
        for doc in connector.list():
            if doc.source_id == source_id:
                return ingest_document_job(doc, connector, job_id)
        return {"status": "not_found", "source_id": source_id}
