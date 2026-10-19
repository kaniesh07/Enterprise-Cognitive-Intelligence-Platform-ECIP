"""
agents/llm_client.py
--------------------
One LLM client for the whole app.

* Providers: Ollama (native /api/chat) and any OpenAI-compatible endpoint
  (OpenAI, Groq, Together, vLLM, LM Studio, llama.cpp server, …).
* Pooled keep-alive HTTP connections (sync + async) — no new TCP/TLS
  handshake per call.
* Ollama `keep_alive` keeps the model resident in RAM/VRAM, so there is no
  multi-second model reload between requests.
* True token streaming (async generator) with per-request token accounting.
* Circuit breaker + retry on connection errors only (never on 4xx).
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Generator, Optional

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents.circuit_breaker import CircuitBreakerOpen, get_breaker  # noqa: F401 (re-export)
from config import settings

logger = structlog.get_logger()

PROVIDER = settings.llm_provider.lower().strip()
OLLAMA_HOST = settings.ollama_host.rstrip("/")
OPENAI_BASE = settings.openai_base_url.rstrip("/")
DEFAULT_MODEL = settings.llm_model
DEFAULT_TEMP = settings.temperature
DEFAULT_MAX_TOKENS = settings.max_tokens
EMBED_MODEL = settings.embed_model

TIMEOUT = httpx.Timeout(connect=5.0, read=settings.llm_timeout_s, write=30.0, pool=10.0)
_LIMITS = httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=120)
_RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError,
              httpx.ReadError, httpx.NetworkError)

# Price per 1M tokens (USD) — cost projection only; local Ollama is free.
_COST_PER_1M = {
    "llama3.1:8b": {"input": 0.06, "output": 0.06},
    "llama3.2:3b": {"input": 0.03, "output": 0.03},
    "phi3:mini": {"input": 0.01, "output": 0.01},
    "llama3.1:70b": {"input": 0.90, "output": 0.90},
    "mistral:7b": {"input": 0.06, "output": 0.06},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class LLMError(RuntimeError):
    """Raised for non-retryable LLM backend errors (bad model name, auth, …)."""


@dataclass
class LLMUsage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    projected_cost_usd: float

    def to_dict(self) -> dict:
        return {
            "model": self.model, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms, "projected_cost_usd": self.projected_cost_usd,
        }


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = _COST_PER_1M.get(model, {"input": 0.06, "output": 0.06})
    return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000


# ── Session usage (whole process) ─────────────────────────────────────────
_session_lock = threading.Lock()
_session_totals = {"prompt": 0, "completion": 0, "total": 0, "projected_cost_usd": 0.0, "calls": 0}


def get_session_usage() -> dict:
    with _session_lock:
        return dict(_session_totals)


def get_total_tokens() -> dict:
    """Process-wide token totals since start-up (used by /token-usage)."""
    return get_session_usage()


def summarize_usage(usages: list[LLMUsage]) -> dict:
    """Per-request token summary."""
    return {
        "prompt": sum(u.prompt_tokens for u in usages),
        "completion": sum(u.completion_tokens for u in usages),
        "total": sum(u.total_tokens for u in usages),
        "projected_cost_usd": round(sum(u.projected_cost_usd for u in usages), 8),
        "calls": len(usages),
    }


def _record_usage(model: str, prompt_tokens: int, completion_tokens: int, latency_s: float,
                  gen_s: float, usage_sink: Optional[list]) -> LLMUsage:
    usage = LLMUsage(model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
                     int(latency_s * 1000), _compute_cost(model, prompt_tokens, completion_tokens))
    with _session_lock:
        _session_totals["prompt"] += prompt_tokens
        _session_totals["completion"] += completion_tokens
        _session_totals["total"] += usage.total_tokens
        _session_totals["projected_cost_usd"] += usage.projected_cost_usd
        _session_totals["calls"] += 1
    if usage_sink is not None:
        usage_sink.append(usage)
    try:
        from monitoring.metrics import get_registry
        get_registry().observe_llm_call(model, latency_s, prompt_tokens, completion_tokens,
                                        ok=True, gen_seconds=gen_s,
                                        cost_usd=usage.projected_cost_usd)
    except Exception:  # metrics must never break a request
        pass
    logger.info("llm_call", **usage.to_dict())
    return usage


def _record_error(model: str):
    try:
        from monitoring.metrics import get_registry
        get_registry().observe_llm_call(model, 0.0, 0, 0, ok=False)
    except Exception:
        pass


# ── Pooled HTTP clients ───────────────────────────────────────────────────
_sync_client: Optional[httpx.Client] = None
_async_clients: dict[int, httpx.AsyncClient] = {}
_client_lock = threading.Lock()


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if PROVIDER == "openai" and settings.openai_api_key:
        h["Authorization"] = f"Bearer {settings.openai_api_key}"
    return h


def get_sync_client() -> httpx.Client:
    global _sync_client
    with _client_lock:
        if _sync_client is None or _sync_client.is_closed:
            _sync_client = httpx.Client(timeout=TIMEOUT, limits=_LIMITS, headers=_headers())
        return _sync_client


def get_async_client() -> httpx.AsyncClient:
    """One AsyncClient per running event loop (clients cannot be shared across loops)."""
    loop_id = id(asyncio.get_running_loop())
    client = _async_clients.get(loop_id)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=TIMEOUT, limits=_LIMITS, headers=_headers())
        _async_clients[loop_id] = client
    return client


async def aclose_clients():
    for c in list(_async_clients.values()):
        try:
            await c.aclose()
        except Exception:
            pass
    _async_clients.clear()


# ── Payload builders ──────────────────────────────────────────────────────

def _messages(messages: list[dict], system: Optional[str]) -> list[dict]:
    out = [{"role": "system", "content": system}] if system else []
    out.extend({"role": m.get("role", "user"), "content": str(m.get("content", ""))}
               for m in messages if m.get("content") is not None)
    return out


def _payload(messages, model, temperature, max_tokens, stream, json_mode=False) -> tuple[str, dict]:
    if PROVIDER == "openai":
        body = {"model": model, "messages": messages, "stream": stream,
                "temperature": temperature, "max_tokens": max_tokens}
        if stream:
            body["stream_options"] = {"include_usage": True}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if settings.reasoning_effort:          # reasoning models, e.g. openai/gpt-oss-20b on Groq
            body["reasoning_effort"] = settings.reasoning_effort
        return f"{OPENAI_BASE}/chat/completions", body
    body = {"model": model, "messages": messages, "stream": stream,
            "keep_alive": settings.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": settings.num_ctx}}
    if json_mode:
        body["format"] = "json"
    return f"{OLLAMA_HOST}/api/chat", body


def _raise_for_status(resp: httpx.Response, model: str):
    if resp.status_code >= 500:
        resp.raise_for_status()
    if resp.status_code >= 400:
        try:
            detail = resp.json()
            detail = detail.get("error") or detail
        except Exception:
            detail = resp.text[:300]
        if resp.status_code == 404 and PROVIDER == "ollama":
            raise LLMError(f"Model '{model}' is not available in Ollama. "
                           f"Pull it first: ollama pull {model}")
        if resp.status_code == 404 and "model" in str(detail).lower():
            raise LLMError(f"The model '{model}' is not available at {OPENAI_BASE} (it may have been "
                           f"retired). Set LLM_MODEL to a model your provider lists "
                           f"(for Groq: console.groq.com/docs/models) and restart.")
        raise LLMError(f"LLM backend returned HTTP {resp.status_code}: {detail}")


def _parse_full(body: dict) -> tuple[str, int, int]:
    if PROVIDER == "openai":
        content = body["choices"][0]["message"].get("content") or ""
        u = body.get("usage") or {}
        return content, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
    return (body.get("message", {}).get("content", ""),
            int(body.get("prompt_eval_count", 0) or 0), int(body.get("eval_count", 0) or 0))


# ── Non-streaming chat ────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
       retry=retry_if_exception_type(_RETRYABLE), reraise=True)
def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system: Optional[str] = None,
    json_mode: bool = False,
    usage_sink: Optional[list] = None,
) -> str:
    url, body = _payload(_messages(messages, system), model, temperature, max_tokens,
                         stream=False, json_mode=json_mode)
    breaker = get_breaker("ollama")
    breaker.allow()
    t0 = time.perf_counter()
    try:
        resp = get_sync_client().post(url, json=body)
        _raise_for_status(resp, model)
    except LLMError:
        _record_error(model)
        raise
    except Exception:
        breaker.record_failure()
        _record_error(model)
        raise
    breaker.record_success()
    content, p, c = _parse_full(resp.json())
    dt = time.perf_counter() - t0
    _record_usage(model, p, c, dt, dt, usage_sink)
    return content


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def chat_json(messages: list[dict], model: str = DEFAULT_MODEL, system: Optional[str] = None,
              usage_sink: Optional[list] = None) -> dict:
    text = _strip_fences(chat(messages, model=model, temperature=0.0, system=system,
                              usage_sink=usage_sink))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if 0 <= start < end:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("json_parse_failed", raw=text[:200])
        return {}


# ── Streaming chat ────────────────────────────────────────────────────────

def _parse_stream_line(line: str) -> tuple[str, Optional[dict], bool]:
    """Return (token, usage_or_None, done)."""
    if PROVIDER == "openai":
        if not line.startswith("data:"):
            return "", None, False
        data = line[5:].strip()
        if data == "[DONE]":
            return "", None, True
        obj = json.loads(data)
        usage = obj.get("usage")
        tok = ""
        if obj.get("choices"):
            tok = (obj["choices"][0].get("delta") or {}).get("content") or ""
        return tok, ({"p": usage.get("prompt_tokens", 0), "c": usage.get("completion_tokens", 0)}
                     if usage else None), False
    obj = json.loads(line)
    if obj.get("error"):
        raise LLMError(str(obj["error"]))
    tok = (obj.get("message") or {}).get("content", "")
    if obj.get("done"):
        return tok, {"p": obj.get("prompt_eval_count", 0) or 0,
                     "c": obj.get("eval_count", 0) or 0}, True
    return tok, None, False


async def chat_stream_async(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system: Optional[str] = None,
    usage_sink: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """Yield tokens as the model produces them."""
    url, body = _payload(_messages(messages, system), model, temperature, max_tokens, stream=True)
    breaker = get_breaker("ollama")
    breaker.allow()
    t0 = time.perf_counter()
    t_first: Optional[float] = None
    usage = {"p": 0, "c": 0}
    n_tokens = 0
    client = get_async_client()
    try:
        async with client.stream("POST", url, json=body) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                _raise_for_status(resp, model)
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    tok, u, done = _parse_stream_line(line)
                except json.JSONDecodeError:
                    continue
                if u:
                    usage = u
                if tok:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_tokens += 1
                    yield tok
                if done:
                    break
    except LLMError:
        _record_error(model)
        raise
    except (asyncio.CancelledError, GeneratorExit):
        # client disconnected / stop button — not a backend failure
        raise
    except Exception:
        breaker.record_failure()
        _record_error(model)
        raise
    breaker.record_success()
    t_end = time.perf_counter()
    completion = usage["c"] or n_tokens
    _record_usage(model, int(usage["p"]), int(completion), t_end - t0,
                  (t_end - t_first) if t_first else 0.0, usage_sink)


def chat_stream(messages: list[dict], model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMP,
                max_tokens: int = DEFAULT_MAX_TOKENS, system: Optional[str] = None,
                usage_sink: Optional[list] = None) -> Generator[str, None, None]:
    """Synchronous streaming (CLI use)."""
    url, body = _payload(_messages(messages, system), model, temperature, max_tokens, stream=True)
    t0 = time.perf_counter()
    usage = {"p": 0, "c": 0}
    n = 0
    with get_sync_client().stream("POST", url, json=body) as resp:
        if resp.status_code >= 400:
            resp.read()
            _raise_for_status(resp, model)
        for line in resp.iter_lines():
            if not line.strip():
                continue
            try:
                tok, u, done = _parse_stream_line(line)
            except json.JSONDecodeError:
                continue
            if u:
                usage = u
            if tok:
                n += 1
                yield tok
            if done:
                break
    dt = time.perf_counter() - t0
    _record_usage(model, int(usage["p"]), int(usage["c"] or n), dt, dt, usage_sink)


# ── Embeddings (delegates to retrieval.embeddings) ────────────────────────

def embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    from retrieval.embeddings import embed_texts
    return embed_texts([text])[0]


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    from retrieval.embeddings import embed_texts
    return embed_texts(texts)


# ── Health / warm-up ──────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Backend reachability (name kept for compatibility; works for both providers)."""
    try:
        if PROVIDER == "openai":
            r = httpx.get(f"{OPENAI_BASE}/models", headers=_headers(), timeout=3.0)
            return r.status_code < 500
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


async def ping_llm(timeout: float = 2.0) -> bool:
    try:
        client = get_async_client()
        if PROVIDER == "openai":
            r = await client.get(f"{OPENAI_BASE}/models", timeout=timeout)
            return r.status_code < 500
        r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def list_available_models() -> list[str]:
    try:
        if PROVIDER == "openai":
            r = httpx.get(f"{OPENAI_BASE}/models", headers=_headers(), timeout=5.0)
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


async def warmup(model: str = DEFAULT_MODEL) -> bool:
    """Load the model into memory ahead of the first user request (Ollama only)."""
    if PROVIDER != "ollama":
        return True
    try:
        client = get_async_client()
        r = await client.post(f"{OLLAMA_HOST}/api/chat",
                              json={"model": model, "messages": [], "keep_alive": settings.keep_alive},
                              timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=5.0))
        ok = r.status_code == 200
        logger.info("llm_warmup", model=model, ok=ok, status=r.status_code)
        return ok
    except Exception as exc:
        logger.warning("llm_warmup_failed", model=model, error=str(exc)[:120])
        return False
