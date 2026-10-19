"""
retrieval/embeddings.py
-----------------------
Pluggable text embeddings.

EMBED_PROVIDER:
  ollama  — Ollama /api/embed (batched; falls back to /api/embeddings on old servers)
  openai  — any OpenAI-compatible /embeddings endpoint
  onnx    — local all-MiniLM-L6-v2 style ONNX model + tokenizer.json (CPU, no server)
  hash    — deterministic feature hashing (no model at all — tests / emergencies)

Query embeddings are LRU-cached, so repeated or retried questions skip the call.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import httpx
import structlog

from config import settings

logger = structlog.get_logger()

PROVIDER = settings.embed_provider.lower().strip()
MODEL = settings.embed_model
_BATCH = 32
_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=10.0)

_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()


def _http() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None or _client.is_closed:
            headers = {}
            if PROVIDER == "openai" and settings.openai_api_key:
                headers["Authorization"] = f"Bearer {settings.openai_api_key}"
            _client = httpx.Client(timeout=_TIMEOUT, headers=headers,
                                   limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
        return _client


def signature() -> str:
    """Short identifier of the embedding space (used to name the vector collection)."""
    name = "hash512" if PROVIDER == "hash" else (
        Path(settings.onnx_model_dir).name if PROVIDER == "onnx" else MODEL)
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", f"{PROVIDER}-{name}").strip("-").lower()[:40]


def _nomic_prefix(texts: list[str], is_query: bool) -> list[str]:
    # nomic-embed-text is trained with task prefixes; using them noticeably improves recall.
    if "nomic" in MODEL.lower():
        p = "search_query: " if is_query else "search_document: "
        return [p + t for t in texts]
    return texts


# ── Providers ─────────────────────────────────────────────────────────────

def _ollama(texts: list[str]) -> list[list[float]]:
    host = settings.ollama_host.rstrip("/")
    client = _http()
    r = client.post(f"{host}/api/embed", json={"model": MODEL, "input": texts,
                                                "keep_alive": settings.keep_alive})
    if r.status_code == 404 and "model" not in r.text.lower():
        # very old Ollama without /api/embed
        out = []
        for t in texts:
            rr = client.post(f"{host}/api/embeddings", json={"model": MODEL, "prompt": t})
            rr.raise_for_status()
            out.append(rr.json()["embedding"])
        return out
    if r.status_code == 404:
        raise RuntimeError(f"Embedding model '{MODEL}' not found in Ollama. Run: ollama pull {MODEL}")
    r.raise_for_status()
    return r.json()["embeddings"]


def _openai(texts: list[str]) -> list[list[float]]:
    r = _http().post(f"{settings.openai_base_url.rstrip('/')}/embeddings",
                     json={"model": MODEL, "input": texts})
    r.raise_for_status()
    data = sorted(r.json()["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


class _OnnxEmbedder:
    def __init__(self, model_dir: str):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        onnx_path = d / "model.onnx" if (d / "model.onnx").exists() else d / "onnx" / "model.onnx"
        self.np = np
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def __call__(self, texts: list[str]) -> list[list[float]]:
        np = self.np
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feeds)[0]                     # (b, seq, dim)
        m = mask[..., None].astype(np.float32)
        pooled = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return pooled.astype(float).tolist()


_onnx: Optional[_OnnxEmbedder] = None
_onnx_lock = threading.Lock()


def _onnx_embed(texts: list[str]) -> list[list[float]]:
    global _onnx
    with _onnx_lock:
        if _onnx is None:
            _onnx = _OnnxEmbedder(settings.onnx_model_dir)
    return _onnx(texts)


_WORD = re.compile(r"[a-z0-9]+")


def _hash_embed(texts: list[str], dim: int = 512) -> list[list[float]]:
    out = []
    for t in texts:
        v = [0.0] * dim
        words = _WORD.findall(t.lower())
        feats = words + [a + "_" + b for a, b in zip(words, words[1:])]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % dim] += 1.0 if (h >> 63) == 0 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


_PROVIDERS = {"ollama": "_ollama", "openai": "_openai", "onnx": "_onnx_embed", "hash": "_hash_embed"}


def _embed(texts: list[str]) -> list[list[float]]:
    fn = globals().get(_PROVIDERS.get(PROVIDER, ""))   # looked up at call time (patchable)
    if fn is None:
        raise ValueError(f"Unknown EMBED_PROVIDER '{PROVIDER}'. Use one of {list(_PROVIDERS)}")
    t0 = time.perf_counter()
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        out.extend(fn(texts[i:i + _BATCH]))
    try:
        from monitoring.metrics import get_registry
        get_registry().observe_embedding(PROVIDER, time.perf_counter() - t0)
    except Exception:
        pass
    return out


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed documents/chunks."""
    if not texts:
        return []
    return _embed(_nomic_prefix([t if t.strip() else " " for t in texts], is_query=False))


@lru_cache(maxsize=2048)
def _embed_query_cached(query: str) -> tuple[float, ...]:
    return tuple(_embed(_nomic_prefix([query], is_query=True))[0])


def embed_query(query: str) -> list[float]:
    return list(_embed_query_cached(query.strip()))


def warmup() -> bool:
    try:
        embed_query("warm up")
        return True
    except Exception as exc:
        logger.warning("embedding_warmup_failed", provider=PROVIDER, error=str(exc)[:120])
        return False
