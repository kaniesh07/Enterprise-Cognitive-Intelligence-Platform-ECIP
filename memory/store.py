"""
memory/store.py
----------------
Persistent user preference memory + per-session state.

Separates:
  - Workflow state   → AgentState TypedDict (in-request, ephemeral)
  - Session memory   → conversation history (request-scoped)
  - User preferences → persisted across sessions (this module)
  - Knowledge        → ChromaDB / BM25 (retrieval layer)

Production: swap JsonFileStore for Redis or PostgreSQL.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Optional

@dataclass
class UserPreferences:
    user_id:         str
    answer_format:   str  = "prose"        # "prose" | "bullets" | "table"
    language:        str  = "en"
    domain_focus:    str  = ""             # e.g. "legal", "finance", "HR"
    verbosity:       str  = "normal"       # "concise" | "normal" | "detailed"
    top_k_override:  Optional[int] = None  # user-specified retrieval depth
    created_at:      float = field(default_factory=time.time)
    updated_at:      float = field(default_factory=time.time)

    def to_prompt_hint(self) -> str:
        """Return a hint injected into the reasoning prompt."""
        parts = []
        if self.answer_format == "bullets":
            parts.append("Format your answer as bullet points.")
        elif self.answer_format == "table":
            parts.append("Format your answer as a markdown table where appropriate.")
        if self.domain_focus:
            parts.append(f"The user works in {self.domain_focus}. Prioritise relevant context.")
        if self.verbosity == "concise":
            parts.append("Be concise. Answer in 2-3 sentences maximum.")
        elif self.verbosity == "detailed":
            parts.append("Be thorough. Include all relevant details and sub-points.")
        return " ".join(parts)


class MemoryStore:
    """
    File-backed user preference store.
    Thread-safe via Lock.
    Production swap: replace _load/_save with Redis HGETALL/HMSET.
    """
    def __init__(self, store_path: str = "./data/processed/user_prefs.json"):
        self._path = Path(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
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

    def get(self, user_id: str) -> UserPreferences:
        with self._lock:
            raw = self._data.get(user_id)
            if raw:
                return UserPreferences(**{k: v for k, v in raw.items()
                                         if k in UserPreferences.__dataclass_fields__})
            return UserPreferences(user_id=user_id)

    def set(self, prefs: UserPreferences):
        with self._lock:
            prefs.updated_at = time.time()
            self._data[prefs.user_id] = asdict(prefs)
            self._save()

    def update(self, user_id: str, **kwargs) -> UserPreferences:
        prefs = self.get(user_id)
        for k, v in kwargs.items():
            if hasattr(prefs, k):
                setattr(prefs, k, v)
        self.set(prefs)
        return prefs

    def delete(self, user_id: str):
        with self._lock:
            self._data.pop(user_id, None)
            self._save()


_store: Optional[MemoryStore] = None

def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        from config import settings
        _store = MemoryStore(str(settings.user_prefs_path))
    return _store
