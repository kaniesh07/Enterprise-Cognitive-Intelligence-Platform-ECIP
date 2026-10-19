"""
connectors/local_fs.py
-----------------------
Local filesystem / NFS / SMB share connector.
Supports millions of files via lazy os.walk — never loads all into memory.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Generator
from connectors.base import BaseConnector, ConnectorDocument

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".csv"}

class LocalFSConnector(BaseConnector):
    source_type = "local"

    def __init__(self, config: dict):
        """
        config keys:
          root_path  : str  — directory to scan (required)
          recursive  : bool — default True
          extensions : list — filter by extension, default SUPPORTED
          exclude_dirs: list — directory names to skip (e.g. [".git", "node_modules"])
        """
        super().__init__(config)
        self.root_path   = Path(config["root_path"])
        self.recursive   = config.get("recursive", True)
        self.extensions  = set(config.get("extensions", list(SUPPORTED)))
        self.exclude_dirs = set(config.get("exclude_dirs", [".git", "__pycache__", "node_modules"]))

    def list(self) -> Generator[ConnectorDocument, None, None]:
        """Lazily walk the directory tree — O(1) memory regardless of file count."""
        if self.recursive:
            for dirpath, dirnames, filenames in os.walk(self.root_path):
                # Prune excluded dirs in-place so os.walk skips them
                dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]
                for fname in filenames:
                    fpath = Path(dirpath) / fname
                    if fpath.suffix.lower() in self.extensions:
                        yield self._make_doc(fpath)
        else:
            for fpath in self.root_path.iterdir():
                if fpath.is_file() and fpath.suffix.lower() in self.extensions:
                    yield self._make_doc(fpath)

    def load(self, doc: ConnectorDocument) -> bytes:
        return Path(doc.source_id).read_bytes()

    def _make_doc(self, fpath: Path) -> ConnectorDocument:
        stat = fpath.stat()
        return ConnectorDocument(
            source_id     = str(fpath.resolve()),
            source_type   = self.source_type,
            title         = fpath.name,
            mime_type     = _mime(fpath.suffix),
            last_modified = stat.st_mtime,
            size_bytes    = stat.st_size,
            metadata      = {"root_path": str(self.root_path), "relative_path": str(fpath.relative_to(self.root_path))},
        )

def _mime(ext: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain", ".md": "text/plain",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".tiff": "image/tiff", ".csv": "text/csv",
    }.get(ext.lower(), "application/octet-stream")
