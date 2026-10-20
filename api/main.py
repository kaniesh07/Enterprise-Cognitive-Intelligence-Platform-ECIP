"""
api/main.py — ECIP FastAPI application.

Chat
  POST /query/stream        ChatGPT-style streaming answer (Server-Sent Events)
  POST /query               same pipeline, single JSON response
Documents
  POST /ingest              upload & index a file      (writer)
  GET  /documents           list indexed documents     (reader)
  DELETE /documents/{t}     remove a document          (admin)
Ops
  GET  /health  /ready  /metrics (Prometheus)  /stats  /token-usage  /circuit-breakers
Enterprise extras (unchanged API)
  /connectors /sync/* /approvals /users/{id}/preferences /tools /evaluate
UI
  GET  /                    the chat UI (frontend/index.html) when SERVE_FRONTEND=true
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Literal, Optional

import structlog
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from agents.circuit_breaker import get_all_stats as _cb_stats, get_breaker as _get_breaker
from auth.middleware import _public_role, require_admin, require_auth, require_writer
from config import settings
from guardrails.pipeline import get_pipeline as _get_guardrail_pipeline
from monitoring.logging_setup import configure_logging
from monitoring.metrics import CONTENT_TYPE, get_registry as _get_registry, make_metrics_middleware

configure_logging(settings.log_level)
logger = structlog.get_logger()
_metrics_enabled = True

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
                      ".webp", ".html", ".htm", ".csv", ".json", ".log", ".rst"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

_state = {"llm_up": False, "vectordb_up": False, "warm": False, "started": time.time()}


# ── Background tasks ──────────────────────────────────────────────────────

async def _refresh_health_gauges():
    from agents.llm_client import ping_llm
    from ingestion.indexer import ping_vector_db
    reg = _get_registry()
    llm_ok, vdb_ok = await asyncio.gather(ping_llm(), asyncio.to_thread(ping_vector_db))
    _state["llm_up"], _state["vectordb_up"] = llm_ok, vdb_ok
    reg.llm_up.set(1 if llm_ok else 0)
    reg.vectordb_up.set(1 if vdb_ok else 0)
    reg.set_circuit_breaker(any(s["state"] == "open" for s in _cb_stats()))
    if vdb_ok:
        try:
            from ingestion.indexer import DocumentIndexer
            st = await asyncio.to_thread(lambda: DocumentIndexer().stats())
            reg.indexed_chunks.set(st.get("total_chunks", 0))
            reg.indexed_documents.set(st.get("documents", 0))
        except Exception:
            pass


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")


def _config_hint() -> str:
    """Explain the most common deploy mistake: cloud container still pointing at a local Ollama."""
    uses_ollama = settings.llm_provider == "ollama" or settings.embed_provider == "ollama"
    if uses_ollama and any(h in settings.ollama_host for h in _LOCAL_HOSTS):
        return (f"LLM_PROVIDER={settings.llm_provider}, EMBED_PROVIDER={settings.embed_provider} and "
                f"OLLAMA_HOST={settings.ollama_host}, but no Ollama server runs inside this container. "
                "On a cloud host set LLM_PROVIDER=openai, OPENAI_BASE_URL, OPENAI_API_KEY, LLM_MODEL "
                "and EMBED_PROVIDER=openai (or hash), or point OLLAMA_HOST at a reachable Ollama server.")
    return ""


async def _warmup():
    """Load everything the first request needs so it is not slow (never blocks start-up)."""
    t0 = time.perf_counter()
    try:
        from ingestion.indexer import get_bm25, get_collection
        await asyncio.to_thread(get_collection)
        await asyncio.to_thread(get_bm25)
    except Exception as exc:
        logger.warning("warmup_vectordb_failed", error=str(exc)[:150])
    from agents.llm_client import warmup as llm_warmup
    from agents.model_router import get_model
    from retrieval.embeddings import warmup as embed_warmup
    from retrieval.retriever import preload_reranker
    llm_ok, embed_ok, _ = await asyncio.gather(
        llm_warmup(get_model("reasoning")),
        asyncio.to_thread(embed_warmup),
        asyncio.to_thread(preload_reranker),
        return_exceptions=True,
    )
    hint = _config_hint()
    if hint and (llm_ok is not True or embed_ok is not True):
        logger.error("model_backend_unreachable", hint=hint)
    _state["warm"] = True
    logger.info("warmup_done", ms=int((time.perf_counter() - t0) * 1000))


async def _gauge_loop():
    while True:
        try:
            await _refresh_health_gauges()
        except Exception as exc:
            logger.warning("health_gauge_refresh_failed", error=str(exc)[:120])
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    tasks = [asyncio.create_task(_warmup()), asyncio.create_task(_gauge_loop())]
    logger.info("ecip_started", provider=settings.llm_provider, model=settings.llm_model,
                embed=f"{settings.embed_provider}:{settings.embed_model}",
                vectordb="http" if settings.chroma_host else "embedded")
    yield
    for t in tasks:
        t.cancel()
    from agents.llm_client import aclose_clients
    await aclose_clients()


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ECIP — Enterprise Cognitive Intelligence Platform",
    description="ChatGPT-style assistant with grounded answers from your documents (RAG).",
    version="3.0.0", docs_url="/docs", redoc_url="/redoc", lifespan=lifespan,
)
_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=_origins != ["*"])
app.add_middleware(make_metrics_middleware())


class _ApiPrefixMiddleware:
    """Serve every endpoint under /api/* as well, so the UI uses one base path whether it
    is served by FastAPI directly or behind the nginx /api proxy."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/api/"):
            scope = dict(scope, path=scope["path"][4:], raw_path=scope.get("raw_path", b"")[4:])
        await self.app(scope, receive, send)


app.add_middleware(_ApiPrefixMiddleware)


# ── Models ────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str = ""


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    conversation_history: list[dict] = Field(default_factory=list)
    stream: bool = False
    top_k: Optional[int] = Field(None, ge=1, le=20)   # None → TOP_K_FINAL
    user_id: str = Field("anonymous", max_length=128)
    mode: Literal["auto", "chat", "docs"] = "auto"
    doc_filter: Optional[str] = None


class QueryResponse(BaseModel):
    final_answer: str = ""
    mode: str = ""
    route_reason: str = ""
    model: str = ""
    citations: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    confidence: float = 0.0
    faithfulness_score: float = 0.0
    hallucination_flags: list[str] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    intent: str = ""
    latency_ms: dict = Field(default_factory=dict)
    p95_latency_ms: dict = Field(default_factory=dict)
    retrieval_metrics: dict = Field(default_factory=dict)
    retrieved_chunks: list[dict] = Field(default_factory=list)
    token_usage: dict = Field(default_factory=dict)
    prompt_version: str = ""


class IngestResponse(BaseModel):
    file: str = ""
    doc_title: str = ""
    pages: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    duration_ms: int = 0


class HealthResponse(BaseModel):
    status: str
    ollama: str          # LLM backend (name kept for compatibility)
    chromadb: str        # vector DB
    timestamp: float
    llm_provider: str = ""
    model: str = ""
    embed: str = ""
    vectordb_mode: str = ""
    warm: bool = False


class ConnectorRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1)
    connector_type: str
    config: dict


class ConnectorSyncRequest(BaseModel):
    force: bool = Field(False, description="Re-index even unchanged documents")
    max_workers: int = Field(4, ge=1, le=32)


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    decided_by: str = Field(..., min_length=1)


class UserPreferencesRequest(BaseModel):
    answer_format: Optional[str] = None
    language: Optional[str] = None
    domain_focus: Optional[str] = None
    verbosity: Optional[str] = None
    top_k_override: Optional[int] = None


# ── Helpers ───────────────────────────────────────────────────────────────

def _client_id(request: Request, user_id: str) -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}:{user_id}"


def _blocked_detail(check) -> dict:
    return {"blocked": True, "message": check.blocked_response,
            "violations": [{"guardrail": v.guardrail.value, "severity": v.severity.value,
                            "message": v.message} for v in check.violations]}


def _apply_output_guardrails(result: dict) -> dict:
    gp = _get_guardrail_pipeline()
    grounded = result.get("mode") == "grounded" or (
        not result.get("mode") and result.get("retrieved_chunks"))
    out = gp.run_output(
        answer=result.get("final_answer", ""),
        retrieved_chunks=result.get("retrieved_chunks", []) if grounded else [],
        faithfulness_score=result.get("faithfulness_score") if grounded else None,
    )
    if out.is_blocked:
        result["final_answer"] = out.blocked_response
        result["confidence"] = 0.0
    elif out.safe_text:
        result["final_answer"] = out.safe_text
    return result


def _record(result: dict, seconds: float):
    try:
        from agents.nodes import record_latencies
        record_latencies(result.get("latency_ms") or {})
        reg = _get_registry()
        reg.record_pipeline_result(result)
        reg.record_request(result.get("mode") or "unknown",
                           "error" if result.get("error") else "ok", seconds)
    except Exception as exc:
        logger.warning("metrics_record_failed", error=str(exc)[:100])


def _embed_label() -> str:
    from retrieval.embeddings import signature
    return signature()


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


# ── Open endpoints ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Fast liveness/dependency check (probes run in parallel, ~2 s worst case). Always 200."""
    try:
        await asyncio.wait_for(_refresh_health_gauges(), timeout=4.0)
    except Exception:
        pass
    llm_ok, vdb_ok = _state["llm_up"], _state["vectordb_up"]
    return HealthResponse(
        status="healthy" if llm_ok and vdb_ok else "degraded",
        ollama="ok" if llm_ok else "unreachable",
        chromadb="ok" if vdb_ok else "unreachable",
        timestamp=time.time(), llm_provider=settings.llm_provider, model=settings.llm_model,
        embed=_embed_label(),
        vectordb_mode="http" if settings.chroma_host else "embedded", warm=_state["warm"],
    )


@app.get("/ready", tags=["System"])
async def ready():
    """Readiness probe for orchestrators (Kubernetes, Cloud Run, ECS)."""
    ok = _state["vectordb_up"] or settings.chroma_host == ""
    return JSONResponse({"ready": ok, "warm": _state["warm"]}, status_code=200 if ok else 503)


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    return Response(_get_registry().to_prometheus_text(), media_type=CONTENT_TYPE)


@app.get("/config/public", tags=["System"])
async def public_config():
    return {"llm_provider": settings.llm_provider, "model": settings.llm_model,
            "embed": _embed_label(),
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS), "max_upload_mb": MAX_UPLOAD_MB,
            "grafana_url": settings.grafana_url.rstrip("/"),
            "public_access": bool(_public_role()),
            "version": app.version}


# ── System ────────────────────────────────────────────────────────────────

@app.get("/stats", tags=["System"])
async def stats(_: str = Depends(require_auth)):
    from ingestion.indexer import DocumentIndexer
    s = await asyncio.to_thread(lambda: DocumentIndexer().stats())
    _get_registry().indexed_chunks.set(s.get("total_chunks", 0))
    return s


@app.get("/token-usage", tags=["System"])
async def token_usage(_: str = Depends(require_auth)):
    """Process-wide token consumption and projected (cloud-equivalent) cost."""
    from agents.llm_client import get_total_tokens
    return get_total_tokens()


# ── Chat / query ──────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(req: QueryRequest, request: Request, _: str = Depends(require_auth)):
    t0 = time.perf_counter()
    gp = _get_guardrail_pipeline()
    input_check = gp.run_input(req.query, client_id=_client_id(request, req.user_id))
    if input_check.is_blocked:
        _get_registry().record_request(req.mode, "blocked")
        raise HTTPException(status_code=400, detail=_blocked_detail(input_check))
    safe_query = input_check.safe_text or req.query

    from agents import orchestrator
    from agents.circuit_breaker import CircuitBreakerOpen
    try:
        result = await asyncio.to_thread(orchestrator.run_query, safe_query, req.conversation_history,
                                         req.user_id, req.mode, req.top_k, req.doc_filter)
    except CircuitBreakerOpen:
        raise HTTPException(503, detail="The language model is temporarily unavailable. Try again shortly.")
    except Exception as exc:
        logger.error("query_pipeline_error", error=str(exc)[:200])
        raise HTTPException(503, detail=f"Query processing failed: {str(exc)[:200]}")

    if result.get("error"):
        _record(result, time.perf_counter() - t0)
        msg = str(result["error"])
        status = 503 if any(k in msg.lower() for k in ("connect", "circuit", "unavailable", "timed out")) else 500
        raise HTTPException(status, detail=msg[:300])

    result = _apply_output_guardrails(result)
    from agents.nodes import get_all_p95
    result["p95_latency_ms"] = get_all_p95()
    result["retrieved_chunks"] = [c.to_dict() if hasattr(c, "to_dict") else c
                                  for c in (result.get("retrieved_chunks") or [])]
    _record(result, time.perf_counter() - t0)
    return QueryResponse(**{k: v for k, v in result.items() if k in QueryResponse.model_fields})


@app.post("/query/stream", tags=["Query"])
async def query_stream(req: QueryRequest, request: Request, _: str = Depends(require_auth)):
    """
    Stream the answer token-by-token (Server-Sent Events). Event payloads:
      {"type":"meta",  "data":{mode, intent, model, sources, retrieval_metrics}}
      {"type":"token", "data":"..."}
      {"type":"final", "data":{full QueryResponse}}   ← authoritative final text
      {"type":"blocked"|"error", "data":"message"}
    """
    t0 = time.perf_counter()
    gp = _get_guardrail_pipeline()
    input_check = gp.run_input(req.query, client_id=_client_id(request, req.user_id))

    async def events() -> AsyncGenerator[str, None]:
        if input_check.is_blocked:
            _get_registry().record_request(req.mode, "blocked")
            yield _sse({"type": "blocked", "data": input_check.blocked_response,
                        "detail": _blocked_detail(input_check)})
            yield _sse({"type": "done"})
            return
        from agents import pipeline
        from agents.circuit_breaker import CircuitBreakerOpen
        from agents.llm_client import LLMError
        turn = None
        try:
            turn = await asyncio.to_thread(pipeline.prepare, input_check.safe_text or req.query,
                                           req.conversation_history, req.user_id, req.mode,
                                           req.top_k, req.doc_filter)
            yield _sse({"type": "meta", "data": {
                "mode": turn.mode, "intent": turn.intent, "route_reason": turn.route_reason,
                "model": turn.model if turn.mode != "instant" else "",
                "sources": [dict(c.to_dict(), index=i) for i, c in enumerate(turn.chunks, 1)],
                "retrieval_metrics": turn.retrieval_metrics,
            }})
            parts: list[str] = []
            async for tok in pipeline.stream(turn):
                parts.append(tok)
                yield _sse({"type": "token", "data": tok})
            result = await asyncio.to_thread(pipeline.finalize, turn, "".join(parts))
            result = _apply_output_guardrails(result)
            from agents.nodes import get_all_p95
            result["p95_latency_ms"] = get_all_p95()
            _record(result, time.perf_counter() - t0)
            yield _sse({"type": "final", "data": {k: v for k, v in result.items()
                                                  if not k.startswith("_")}})
        except asyncio.CancelledError:
            logger.info("stream_cancelled_by_client")
            raise
        except CircuitBreakerOpen:
            _get_registry().record_request(req.mode, "error")
            yield _sse({"type": "error", "data": "The language model is temporarily unavailable "
                                                 "(circuit open). Please retry in a minute."})
        except LLMError as exc:
            _get_registry().record_request(getattr(turn, "mode", req.mode), "error")
            msg = str(exc)
            if _config_hint() and "reach" in msg.lower():
                msg = "The language model is not configured for this server. " + _config_hint()
            yield _sse({"type": "error", "data": msg})
        except Exception as exc:
            logger.error("stream_error", error=str(exc)[:300])
            _get_registry().record_request(getattr(turn, "mode", req.mode), "error")
            msg = str(exc) or exc.__class__.__name__
            if "connect" in msg.lower():
                msg = (f"Cannot reach the language model at "
                       f"{settings.ollama_host if settings.llm_provider == 'ollama' else settings.openai_base_url}. "
                       f"Is it running?")
                if _config_hint():
                    msg = "The language model is not configured for this server. " + _config_hint()
            yield _sse({"type": "error", "data": msg[:400]})
        yield _sse({"type": "done"})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


# ── Documents ─────────────────────────────────────────────────────────────

_SAFE_NAME = re.compile(r"[^\w\-. ()\[\]]+", re.UNICODE)


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(settings.chunk_size, ge=128, le=2048),
    chunk_overlap: int = Query(settings.chunk_overlap, ge=0, le=512),
    _: str = Depends(require_writer),
):
    filename = Path(file.filename or "document.txt").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"Unsupported file type '{suffix}'. "
                                        f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    title = _SAFE_NAME.sub("_", Path(filename).stem).strip() or "document"
    tmp_dir = tempfile.mkdtemp(prefix="ecip_upload_")
    tmp_path = Path(tmp_dir) / f"{title}{suffix}"
    size = 0
    with open(tmp_path, "wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                fh.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(413, detail=f"File larger than {MAX_UPLOAD_MB} MB.")
            fh.write(chunk)
    t0 = time.perf_counter()
    reg = _get_registry()
    try:
        from ingestion.indexer import ingest_file
        result = await asyncio.to_thread(ingest_file, str(tmp_path), chunk_size, chunk_overlap,
                                         title, filename)
    except Exception as exc:
        reg.ingestion_errors.inc()
        logger.error("ingest_failed", file=filename, error=str(exc)[:300])
        raise HTTPException(422, detail=f"Failed to process document: {str(exc)[:300]}")
    finally:
        tmp_path.unlink(missing_ok=True)
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
    duration_ms = int((time.perf_counter() - t0) * 1000)
    reg.ingestion_total.inc()
    reg.ingestion_duration.observe(duration_ms / 1000.0)
    reg.ingested_chunks.inc(result.get("chunks_indexed", 0))
    if result.get("chunks_created", 0) == 0:
        raise HTTPException(422, detail="No text could be extracted from this file "
                                        "(scanned PDFs/images need Tesseract OCR installed).")
    asyncio.create_task(_refresh_health_gauges())
    return IngestResponse(file=filename, doc_title=result.get("doc_title", title),
                          pages=result.get("pages", 0), chunks_created=result.get("chunks_created", 0),
                          chunks_indexed=result.get("chunks_indexed", 0), duration_ms=duration_ms)


@app.get("/documents", tags=["Ingestion"])
async def list_documents(_: str = Depends(require_auth)):
    """List indexed documents (never 500s — returns an empty list if the index is unavailable)."""
    try:
        from ingestion.indexer import DocumentIndexer
        docs = await asyncio.to_thread(lambda: DocumentIndexer().list_documents())
        return {"documents": docs, "total": len(docs)}
    except Exception as exc:
        logger.warning("list_documents_unavailable", error=str(exc)[:120])
        return {"documents": [], "total": 0, "error": "Index not ready yet — try again shortly"}


@app.delete("/documents/{doc_title:path}", tags=["Ingestion"])
async def delete_document(doc_title: str, reason: str = Query(..., min_length=3),
                          _: str = Depends(require_admin)):
    from ingestion.indexer import DocumentIndexer
    n = await asyncio.to_thread(lambda: DocumentIndexer().delete_document(doc_title))
    if not n:
        raise HTTPException(404, detail=f"Document '{doc_title}' not found.")
    logger.warning("document_deleted", doc=doc_title, reason=reason, chunks=n)
    asyncio.create_task(_refresh_health_gauges())
    return {"status": "deleted", "doc_title": doc_title, "chunks_removed": n, "reason": reason}


# ── Connectors ────────────────────────────────────────────────────────────

@app.post("/connectors", tags=["Connectors"])
async def register_connector(req: ConnectorRegisterRequest, _: str = Depends(require_admin)):
    from connectors.registry import get_connector_registry
    try:
        get_connector_registry().register(req.name, req.connector_type, req.config)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    return {"status": "registered", "name": req.name, "type": req.connector_type}


@app.get("/connectors", tags=["Connectors"])
async def list_connectors(_: str = Depends(require_auth)):
    from connectors.registry import get_connector_registry
    reg = get_connector_registry()
    return {"connectors": reg.list_all(), "supported_types": reg.supported_types()}


@app.delete("/connectors/{name}", tags=["Connectors"])
async def remove_connector(name: str, _: str = Depends(require_admin)):
    from connectors.registry import get_connector_registry
    get_connector_registry().remove(name)
    return {"status": "removed", "name": name}


@app.post("/connectors/{name}/sync", tags=["Connectors"])
async def trigger_sync(name: str, req: ConnectorSyncRequest, _: str = Depends(require_writer)):
    from pipeline.worker import run_connector_sync
    try:
        job_id = run_connector_sync(name, force=req.force, max_workers=req.max_workers)
    except ValueError as exc:
        raise HTTPException(404, detail=str(exc))
    return {"job_id": job_id, "connector": name, "status": "started", "poll_url": f"/sync/jobs/{job_id}"}


@app.get("/sync/jobs", tags=["Connectors"])
async def list_sync_jobs(_: str = Depends(require_auth)):
    from pipeline.progress import get_progress_tracker
    return {"jobs": get_progress_tracker().list_all()}


@app.get("/sync/jobs/{job_id}", tags=["Connectors"])
async def get_sync_job(job_id: str, _: str = Depends(require_auth)):
    from pipeline.progress import get_progress_tracker
    job = get_progress_tracker().get(job_id)
    if not job:
        raise HTTPException(404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/sync/manifest/stats", tags=["Connectors"])
async def manifest_stats(_: str = Depends(require_auth)):
    from pipeline.sync_manifest import get_manifest
    return get_manifest().stats()


@app.post("/sync/manifest/reset-failed", tags=["Connectors"])
async def reset_failed_docs(_: str = Depends(require_admin)):
    from pipeline.sync_manifest import get_manifest
    get_manifest().reset_failed()
    return {"status": "ok", "message": "Failed documents marked for retry."}


# ── Approvals ─────────────────────────────────────────────────────────────

@app.get("/approvals", tags=["Approvals"])
async def list_approvals(_: str = Depends(require_admin)):
    from tools.approval_gate import get_approval_gate
    return {"pending": [r.to_dict() for r in get_approval_gate().list_pending()]}


@app.post("/approvals/{request_id}/decide", tags=["Approvals"])
async def decide_approval(request_id: str, req: ApprovalDecisionRequest, _: str = Depends(require_admin)):
    from tools.approval_gate import get_approval_gate
    try:
        return get_approval_gate().decide(request_id, req.approved, req.decided_by).to_dict()
    except ValueError as exc:
        raise HTTPException(404, detail=str(exc))


# ── User preferences ──────────────────────────────────────────────────────

@app.get("/users/{user_id}/preferences", tags=["Users"])
async def get_user_preferences(user_id: str, _: str = Depends(require_auth)):
    from dataclasses import asdict
    from memory.store import get_memory_store
    return asdict(get_memory_store().get(user_id))


@app.put("/users/{user_id}/preferences", tags=["Users"])
async def update_user_preferences(user_id: str, req: UserPreferencesRequest, _: str = Depends(require_auth)):
    from dataclasses import asdict
    from memory.store import get_memory_store
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    return asdict(get_memory_store().update(user_id, **updates))


# ── Tools ─────────────────────────────────────────────────────────────────

@app.get("/tools", tags=["Tools"])
async def list_tools(_: str = Depends(require_auth)):
    from tools.registry import get_registry
    return {"tools": get_registry().list_tools(), "tool_metrics": get_registry().get_metrics()}


@app.post("/tools/{tool_name}", tags=["Tools"])
async def call_tool(tool_name: str, params: dict, role: str = Depends(require_auth)):
    from tools.registry import get_registry
    result = await asyncio.to_thread(get_registry().call, tool_name, params, role)
    _get_registry().record_tool_call(tool_name, result.success)
    if result.requires_approval:
        return JSONResponse(status_code=202, content={"requires_approval": True, "error": result.error})
    if not result.success:
        raise HTTPException(400, detail=result.error)
    return {"data": result.data, "latency_ms": result.latency_ms}


# ── Evaluation ────────────────────────────────────────────────────────────

@app.post("/evaluate", tags=["Evaluation"])
async def run_evaluation(eval_dataset_path: str = Query(None), _: str = Depends(require_admin)):
    path = eval_dataset_path or str(settings.eval_dataset_path)
    try:
        from evaluation.ragas_eval import run_ragas_evaluation
        return await asyncio.to_thread(run_ragas_evaluation, path)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


# ── Circuit breakers ──────────────────────────────────────────────────────

@app.get("/circuit-breakers", tags=["System"])
async def circuit_breaker_status(_: str = Depends(require_auth)):
    stats = _cb_stats()
    _get_registry().set_circuit_breaker(any(s["state"] == "open" for s in stats))
    return {"circuit_breakers": stats}


@app.post("/circuit-breakers/{name}/reset", tags=["System"])
async def reset_circuit_breaker(name: str, _: str = Depends(require_admin)):
    _get_breaker(name).reset()
    return {"status": "reset", "name": name}


# ── Frontend (single-container deployments) ───────────────────────────────

if settings.serve_frontend and (FRONTEND_DIR / "index.html").exists():
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        svg = FRONTEND_DIR / "favicon.svg"
        if svg.exists():
            return FileResponse(svg, media_type="image/svg+xml")
        return Response(status_code=204)


def main():
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port,
                log_level=settings.log_level.lower(), proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
