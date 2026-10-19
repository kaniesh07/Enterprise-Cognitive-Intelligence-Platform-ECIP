"""
connectors/confluence.py
-------------------------
Atlassian Confluence connector (Cloud and Server/Data Center).
Fetches pages as HTML and converts to text for indexing.
Requires: requests  (already in requirements)
"""
from __future__ import annotations
from typing import Generator
from connectors.base import BaseConnector, ConnectorDocument

class ConfluenceConnector(BaseConnector):
    source_type = "confluence"

    def __init__(self, config: dict):
        """
        config keys:
          base_url    : str  — e.g. "https://company.atlassian.net/wiki" (Cloud)
                               or "https://confluence.company.com" (Server)
          username    : str  — email for Cloud, username for Server
          api_token   : str  — API token (Cloud) or password (Server)
          space_keys  : list — Confluence space keys to index, e.g. ["HR", "ENG", "LEGAL"]
                               empty = all spaces
          include_attachments: bool — also index PDF/DOCX attachments (default False)
        """
        super().__init__(config)
        self.base_url = config["base_url"].rstrip("/")
        self.auth     = (config["username"], config["api_token"])

    def _get(self, path: str, params: dict = None):
        import requests
        resp = requests.get(
            f"{self.base_url}/rest/api/{path}",
            auth=self.auth,
            params=params or {},
            headers={"Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_spaces(self) -> list[str]:
        configured = self.config.get("space_keys", [])
        if configured:
            return configured
        data = self._get("space", {"limit": 250})
        return [s["key"] for s in data.get("results", [])]

    def list(self) -> Generator[ConnectorDocument, None, None]:
        space_keys = self._get_spaces()
        for space_key in space_keys:
            start = 0
            limit = 100
            while True:
                data = self._get("content", {
                    "type": "page", "spaceKey": space_key,
                    "expand": "version,metadata.properties",
                    "start": start, "limit": limit,
                })
                results = data.get("results", [])
                for page in results:
                    import time as _time
                    from datetime import datetime
                    mod_str = page.get("version", {}).get("when", "")
                    try:
                        mod_ts = datetime.fromisoformat(mod_str.rstrip("Z")).timestamp()
                    except Exception:
                        mod_ts = _time.time()
                    yield ConnectorDocument(
                        source_id     = f"confluence://{page['id']}",
                        source_type   = self.source_type,
                        title         = page["title"],
                        mime_type     = "text/html",
                        last_modified = mod_ts,
                        size_bytes    = 0,
                        metadata      = {
                            "page_id": page["id"],
                            "space_key": space_key,
                            "version": page.get("version", {}).get("number", 1),
                            "web_url": f"{self.base_url}/pages/{page['id']}",
                        },
                    )
                if len(results) < limit:
                    break
                start += limit

    def load(self, doc: ConnectorDocument) -> bytes:
        """Fetch page body as storage format HTML, convert to plain text."""
        page_id = doc.metadata["page_id"]
        data = self._get(f"content/{page_id}", {"expand": "body.storage"})
        html = data.get("body", {}).get("storage", {}).get("value", "")
        # Convert HTML to plain text (basic tag stripping)
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.encode("utf-8")

    def test_connection(self) -> bool:
        try:
            self._get("space", {"limit": 1})
            return True
        except Exception:
            return False
