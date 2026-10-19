"""
connectors/base.py
-------------------
Abstract base class all source connectors must implement.
Documents NEVER move — connectors stream content in place.
"""
from __future__ import annotations
import re
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, Optional


class ConnectorStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    SKIPPED    = "skipped"   # already indexed, not modified


@dataclass
class ConnectorDocument:
    """A single document discovered by a connector."""
    source_id:    str              # unique ID within the source (path, object key, page ID)
    source_type:  str              # "s3", "sharepoint", "gdrive", "local", "confluence", "sql"
    title:        str
    content_bytes: Optional[bytes] = None    # streamed content (None until loaded)
    content_url:  Optional[str]   = None    # or a URL to fetch on demand
    mime_type:    str             = "application/pdf"
    last_modified: float          = field(default_factory=time.time)
    size_bytes:   int             = 0
    metadata:     dict            = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """SHA256 of source_id + last_modified — used for change detection.
        Web connector sets _fingerprint_override to use content hash instead."""
        if hasattr(self, '_fingerprint_override') and self._fingerprint_override:
            return self._fingerprint_override
        raw = f"{self.source_id}:{self.last_modified}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    @property
    def extension(self) -> str:
        # FIX: only treat a real-looking suffix as an extension ("v2.0 notes" or
        # "Page (docs/a.html)" used to yield bogus extensions like ".0 notes").
        if "." in self.title:
            ext = "." + self.title.rsplit(".", 1)[-1].lower()
            if re.fullmatch(r"\.[a-z0-9]{1,5}", ext):
                return ext
        ext_map = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "text/plain": ".txt",
            "text/html": ".html",
            "image/png": ".png",
            "image/jpeg": ".jpg",
        }
        return ext_map.get(self.mime_type, ".bin")


class BaseConnector(ABC):
    """
    All source connectors extend this class.
    The key contract: list() is a lazy generator — it does NOT download
    everything upfront. Content is loaded only when load() is called.
    """
    source_type: str = "base"

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def list(self) -> Generator[ConnectorDocument, None, None]:
        """
        Lazily yield ConnectorDocument objects from the source.
        Each document has metadata but content_bytes may be None.
        Never loads all documents into memory at once.
        """
        ...

    @abstractmethod
    def load(self, doc: ConnectorDocument) -> bytes:
        """
        Load the actual content bytes for a document.
        Called only when the document is being processed.
        """
        ...

    def test_connection(self) -> bool:
        """Verify the connector can reach its source. Return True if OK."""
        try:
            next(iter(self.list()))
            return True
        except StopIteration:
            return True   # empty source is still connected
        except Exception:
            return False

    def supports_incremental(self) -> bool:
        """Does this connector support last_modified-based incremental sync?"""
        return True
