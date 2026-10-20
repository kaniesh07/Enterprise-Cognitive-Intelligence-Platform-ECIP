"""
config.py — Central configuration (pydantic-settings v2).

Every module reads settings from here so defaults are consistent between
local runs, Docker Compose and cloud deployments. Values come from
environment variables (or a .env file in the working directory).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=False, extra="ignore",
    )

    # ── LLM provider ──────────────────────────────────────────────────────
    # "ollama"  → local/self-hosted Ollama (default)
    # "openai"  → any OpenAI-compatible endpoint (OpenAI, Groq, Together,
    #             vLLM, LM Studio, llama.cpp server, Azure OpenAI proxy …)
    llm_provider:   str   = Field("ollama",                 validation_alias="LLM_PROVIDER")
    ollama_host:    str   = Field("http://localhost:11434", validation_alias="OLLAMA_HOST")
    openai_base_url: str  = Field("https://api.openai.com/v1", validation_alias="OPENAI_BASE_URL")
    openai_api_key: str   = Field("",                       validation_alias="OPENAI_API_KEY")
    llm_model:      str   = Field("llama3.2:3b",            validation_alias="LLM_MODEL")
    max_tokens:     int   = Field(1024,                     validation_alias="MAX_TOKENS")
    grafana_url:    str   = Field("",                       validation_alias="GRAFANA_URL")  # shown in the chat UI
    reasoning_effort: str = Field("",                       validation_alias="REASONING_EFFORT")  # low|medium|high
    temperature:    float = Field(0.3,                      validation_alias="TEMPERATURE")
    num_ctx:        int   = Field(4096,                     validation_alias="NUM_CTX")
    keep_alive:     str   = Field("30m",                    validation_alias="OLLAMA_KEEP_ALIVE")
    llm_timeout_s:  float = Field(180.0,                    validation_alias="LLM_TIMEOUT_S")
    history_turns:  int   = Field(6,                        validation_alias="HISTORY_TURNS")

    # ── Embeddings ────────────────────────────────────────────────────────
    # "ollama" | "openai" | "onnx" (local all-MiniLM ONNX dir) | "hash" (no model, testing)
    embed_provider: str = Field("ollama",           validation_alias="EMBED_PROVIDER")
    embed_model:    str = Field("nomic-embed-text", validation_alias="EMBED_MODEL")
    onnx_model_dir: str = Field("./models/minilm",  validation_alias="ONNX_MODEL_DIR")

    # ── Auth ──────────────────────────────────────────────────────────────
    api_keys: str = Field("ecip-dev-key:admin", validation_alias="API_KEYS")

    # ── Vector DB ─────────────────────────────────────────────────────────
    # Leave CHROMA_HOST empty to use an embedded, on-disk Chroma (no server needed).
    chroma_host:       str = Field("",               validation_alias="CHROMA_HOST")
    chroma_port:       int = Field(8000,             validation_alias="CHROMA_PORT")
    chroma_collection: str = Field("ecip_documents", validation_alias="CHROMA_COLLECTION")

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k_dense:   int   = Field(10,   validation_alias="TOP_K_DENSE")
    top_k_bm25:    int   = Field(10,   validation_alias="TOP_K_BM25")
    top_k_final:   int   = Field(4,    validation_alias="TOP_K_FINAL")
    chunk_size:    int   = Field(300,  validation_alias="CHUNK_SIZE")
    chunk_overlap: int   = Field(60,   validation_alias="CHUNK_OVERLAP")
    rerank_enabled: bool = Field(False, validation_alias="RERANK_ENABLED")
    # In "auto" mode a question is answered from documents only when the best
    # chunk clears these relevance bars; otherwise it is answered as normal chat.
    # Leave MIN_DENSE_SCORE unset to use a sensible default for the embedding model in use.
    min_dense_score: Optional[float] = Field(None, validation_alias="MIN_DENSE_SCORE")
    instant_smalltalk: bool = Field(True, validation_alias="INSTANT_SMALLTALK")
    min_lexical_overlap: float = Field(0.34, validation_alias="MIN_LEXICAL_OVERLAP")
    context_chars_per_chunk: int = Field(1400, validation_alias="CONTEXT_CHARS_PER_CHUNK")
    max_context_chars: int = Field(4500, validation_alias="MAX_CONTEXT_CHARS")

    # ── Storage ───────────────────────────────────────────────────────────
    # Only DATA_DIR normally needs setting; the other paths default to files inside it.
    data_dir:        Path = Field(Path("./data"), validation_alias="DATA_DIR")
    bm25_index_path: Optional[Path] = Field(None, validation_alias="BM25_INDEX_PATH")
    user_prefs_path: Optional[Path] = Field(None, validation_alias="USER_PREFS_PATH")
    manifest_path:   Optional[Path] = Field(None, validation_alias="MANIFEST_PATH")
    connector_store: Optional[Path] = Field(None, validation_alias="CONNECTOR_STORE_PATH")

    @model_validator(mode="after")
    def _derive_paths(self):
        proc = self.data_dir / "processed"
        self.bm25_index_path = self.bm25_index_path or proc / "bm25_index.pkl"
        self.user_prefs_path = self.user_prefs_path or proc / "user_prefs.json"
        self.manifest_path = self.manifest_path or proc / "sync_manifest.json"
        self.connector_store = self.connector_store or proc / "connectors.json"
        return self

    # ── Scale ingestion ───────────────────────────────────────────────────
    ingest_workers:    int = Field(4,   validation_alias="INGEST_WORKERS")
    max_doc_size_mb:   int = Field(100, validation_alias="MAX_DOC_SIZE_MB")
    celery_broker_url: str = Field("",  validation_alias="CELERY_BROKER_URL")
    use_docling:       bool = Field(False, validation_alias="USE_DOCLING")

    # ── Guardrails ────────────────────────────────────────────────────────
    guardrail_rate_limit:   bool  = Field(True, validation_alias="GUARDRAIL_RATE_LIMIT")
    faithfulness_threshold: float = Field(0.35, validation_alias="FAITHFULNESS_THRESHOLD")
    require_citations:      bool  = Field(True, validation_alias="REQUIRE_CITATIONS")
    redact_pii:             bool  = Field(True, validation_alias="REDACT_PII")
    rate_limit_rpm:         int   = Field(60,   validation_alias="RATE_LIMIT_RPM")
    rate_limit_rph:         int   = Field(1000, validation_alias="RATE_LIMIT_RPH")
    rate_limit_burst:       int   = Field(10,   validation_alias="RATE_LIMIT_BURST")
    # "heuristic" (fast, no extra LLM call) | "llm" (slower, second LLM pass) | "off"
    verifier_mode:          str   = Field("heuristic", validation_alias="VERIFIER_MODE")

    # ── API ───────────────────────────────────────────────────────────────
    api_host:     str = Field("0.0.0.0", validation_alias="API_HOST")
    api_port:     int = Field(8000,      validation_alias=AliasChoices("PORT", "API_PORT"))
    log_level:    str = Field("INFO",    validation_alias="LOG_LEVEL")
    cors_origins: str = Field("*",       validation_alias="CORS_ORIGINS")
    serve_frontend: bool = Field(True,   validation_alias="SERVE_FRONTEND")

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_dataset_path: Path = Field(Path("./evaluation/datasets/eval_set.json"),
                                    validation_alias="EVAL_DATASET_PATH")

    # Backwards-compatible aliases (older code / .env files used these names)
    @property
    def fast_model(self) -> str:
        return self.llm_model

    @property
    def strong_model(self) -> str:
        return self.llm_model

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir / "raw", self.data_dir / "processed", self.data_dir / "chroma",
            self.bm25_index_path.parent, Path("./evaluation/results"),
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
