"""
connectors/sharepoint.py
-------------------------
Microsoft SharePoint / OneDrive connector via Microsoft Graph API.
Requires: msal requests  (pip install msal requests)
Documents stay in SharePoint — only content bytes are streamed for indexing.
"""
from __future__ import annotations
from typing import Generator, Optional
from connectors.base import BaseConnector, ConnectorDocument

class SharePointConnector(BaseConnector):
    source_type = "sharepoint"

    def __init__(self, config: dict):
        """
        config keys:
          tenant_id     : str  — Azure AD tenant ID
          client_id     : str  — App registration client ID
          client_secret : str  — App registration client secret
          site_url      : str  — e.g. "https://company.sharepoint.com/sites/HR"
          drive_name    : str  — document library name, default "Documents"
          folder_path   : str  — subfolder within library, default "/"
          extensions    : list — file types to index
        """
        super().__init__(config)
        self._token: Optional[str] = None
        self._graph_base = "https://graph.microsoft.com/v1.0"

    def _get_token(self) -> str:
        if self._token:
            return self._token
        try:
            import msal
        except ImportError:
            raise ImportError("msal required: pip install msal")
        app = msal.ConfidentialClientApplication(
            client_id=self.config["client_id"],
            client_credential=self.config["client_secret"],
            authority=f"https://login.microsoftonline.com/{self.config['tenant_id']}",
        )
        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        if "access_token" not in result:
            raise RuntimeError(f"SharePoint auth failed: {result.get('error_description')}")
        self._token = result["access_token"]
        return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}", "Accept": "application/json"}

    def _get_site_id(self) -> str:
        import requests
        site_url = self.config["site_url"]
        # Extract hostname and site path from URL
        parts = site_url.replace("https://", "").split("/", 2)
        hostname = parts[0]
        site_path = "/" + "/".join(parts[1:]) if len(parts) > 1 else "/"
        url = f"{self._graph_base}/sites/{hostname}:{site_path}"
        resp = requests.get(url, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()["id"]

    def _iter_folder(self, drive_id: str, folder_id: str, exts: set) -> Generator[ConnectorDocument, None, None]:
        """Recursively iterate folder contents using Graph API pagination."""
        import requests
        url = f"{self._graph_base}/drives/{drive_id}/items/{folder_id}/children"
        while url:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("value", []):
                if "folder" in item:
                    # Recurse into subfolder
                    yield from self._iter_folder(drive_id, item["id"], exts)
                elif "file" in item:
                    name = item["name"]
                    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
                    if not exts or ext in exts:
                        import time as _time
                        from datetime import datetime
                        mod_str = item.get("lastModifiedDateTime", "")
                        try:
                            mod_ts = datetime.fromisoformat(mod_str.rstrip("Z")).timestamp()
                        except Exception:
                            mod_ts = _time.time()
                        yield ConnectorDocument(
                            source_id     = item["id"],
                            source_type   = self.source_type,
                            title         = name,
                            mime_type     = item.get("file", {}).get("mimeType", "application/octet-stream"),
                            last_modified = mod_ts,
                            size_bytes    = item.get("size", 0),
                            metadata      = {
                                "drive_id": drive_id,
                                "item_id": item["id"],
                                "web_url": item.get("webUrl", ""),
                                "site_url": self.config["site_url"],
                            },
                        )
            url = data.get("@odata.nextLink")  # follow pagination

    def list(self) -> Generator[ConnectorDocument, None, None]:
        import requests
        exts = set(self.config.get("extensions", [".pdf", ".docx", ".txt"]))
        site_id  = self._get_site_id()
        drive_name = self.config.get("drive_name", "Documents")
        url = f"{self._graph_base}/sites/{site_id}/drives"
        resp = requests.get(url, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        drive_id = None
        for d in resp.json().get("value", []):
            if d["name"] == drive_name:
                drive_id = d["id"]
                break
        if not drive_id:
            raise RuntimeError(f"Drive '{drive_name}' not found in SharePoint site.")
        # Get root folder or subfolder
        folder_path = self.config.get("folder_path", "/").strip("/")
        if folder_path:
            url = f"{self._graph_base}/drives/{drive_id}/root:/{folder_path}"
            resp = requests.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            folder_id = resp.json()["id"]
        else:
            folder_id = "root"
        yield from self._iter_folder(drive_id, folder_id, exts)

    def load(self, doc: ConnectorDocument) -> bytes:
        import requests
        drive_id = doc.metadata["drive_id"]
        item_id  = doc.metadata["item_id"]
        url = f"{self._graph_base}/drives/{drive_id}/items/{item_id}/content"
        resp = requests.get(url, headers=self._headers(), timeout=120, allow_redirects=True)
        resp.raise_for_status()
        return resp.content

    def test_connection(self) -> bool:
        try:
            self._get_site_id()
            return True
        except Exception:
            return False
