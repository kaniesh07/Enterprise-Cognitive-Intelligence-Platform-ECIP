"""
connectors/database.py
-----------------------
SQL database connector — indexes text columns from relational databases.
Supports PostgreSQL, MySQL, SQLite, MSSQL via SQLAlchemy.
Requires: sqlalchemy  (pip install sqlalchemy)
"""
from __future__ import annotations
import time
from typing import Generator
from connectors.base import BaseConnector, ConnectorDocument

class DatabaseConnector(BaseConnector):
    source_type = "sql"

    def __init__(self, config: dict):
        """
        config keys:
          connection_string : str  — SQLAlchemy URL, e.g.
                                     "postgresql://user:pass@host/db"
                                     "sqlite:///./data/knowledge.db"
          query             : str  — SQL query returning rows with at least:
                                     id (unique), title, content, updated_at (optional)
                                     e.g. "SELECT id, title, body AS content, updated_at FROM articles"
          id_column         : str  — column used as unique document ID (default "id")
          title_column      : str  — column used as document title (default "title")
          content_column    : str  — column with main text (default "content")
          updated_at_column : str  — for incremental sync (default "updated_at")
          batch_size        : int  — rows per fetch (default 500)
        """
        super().__init__(config)
        self._engine = None

    def _get_engine(self):
        if self._engine:
            return self._engine
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError("sqlalchemy required: pip install sqlalchemy")
        self._engine = create_engine(self.config["connection_string"], pool_pre_ping=True)
        return self._engine

    def list(self) -> Generator[ConnectorDocument, None, None]:
        from sqlalchemy import text
        id_col      = self.config.get("id_column", "id")
        title_col   = self.config.get("title_column", "title")
        content_col = self.config.get("content_column", "content")
        updated_col = self.config.get("updated_at_column", "updated_at")
        batch_size  = self.config.get("batch_size", 500)
        query       = self.config["query"]

        with self._get_engine().connect() as conn:
            offset = 0
            while True:
                paginated = f"{query} LIMIT {batch_size} OFFSET {offset}"
                rows = conn.execute(text(paginated)).mappings().all()
                if not rows:
                    break
                for row in rows:
                    row_dict = dict(row)
                    uid     = str(row_dict.get(id_col, ""))
                    title   = str(row_dict.get(title_col, uid))
                    content = str(row_dict.get(content_col, ""))
                    updated = row_dict.get(updated_col)
                    try:
                        mod_ts = updated.timestamp() if hasattr(updated, "timestamp") else float(updated or time.time())
                    except Exception:
                        mod_ts = time.time()
                    yield ConnectorDocument(
                        source_id     = f"sql://{uid}",
                        source_type   = self.source_type,
                        title         = title,
                        content_bytes = content.encode("utf-8"),  # content already in memory
                        mime_type     = "text/plain",
                        last_modified = mod_ts,
                        size_bytes    = len(content.encode("utf-8")),
                        metadata      = {"row_id": uid, "connection": self.config["connection_string"].split("@")[-1]},
                    )
                offset += batch_size

    def load(self, doc: ConnectorDocument) -> bytes:
        # Content already stored in content_bytes during list()
        return doc.content_bytes or b""

    def test_connection(self) -> bool:
        try:
            with self._get_engine().connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
