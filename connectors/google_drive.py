"""
connectors/google_drive.py
---------------------------
Google Drive connector via Google Drive API v3.
Requires: google-api-python-client google-auth  (pip install google-api-python-client google-auth)
Exports Google Docs/Sheets/Slides as PDF for indexing.
"""
from __future__ import annotations
from typing import Generator
from connectors.base import BaseConnector, ConnectorDocument

# Google MIME types → export format
_GOOGLE_EXPORT = {
    "application/vnd.google-apps.document":     ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet":  ("application/pdf", ".pdf"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
}
_NATIVE_MIME = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt", "text/csv": ".csv",
    "image/png": ".png", "image/jpeg": ".jpg",
}

class GoogleDriveConnector(BaseConnector):
    source_type = "gdrive"

    def __init__(self, config: dict):
        """
        config keys:
          service_account_json : str  — path to service account JSON key file
          folder_id            : str  — Google Drive folder ID to scan (optional, default = My Drive)
          shared_drive_id      : str  — Shared Drive ID (optional)
          recursive            : bool — default True
        """
        super().__init__(config)
        self._service = None

    def _get_service(self):
        if self._service:
            return self._service
        try:
            from googleapiclient.discovery import build
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError("Required: pip install google-api-python-client google-auth")
        creds = service_account.Credentials.from_service_account_file(
            self.config["service_account_json"],
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def _iter_folder(self, folder_id: str) -> Generator[ConnectorDocument, None, None]:
        svc = self._get_service()
        query = f"'{folder_id}' in parents and trashed=false"
        page_token = None
        while True:
            kwargs = dict(
                q=query, pageSize=100,
                fields="nextPageToken,files(id,name,mimeType,modifiedTime,size,webViewLink)",
            )
            if self.config.get("shared_drive_id"):
                kwargs.update(driveId=self.config["shared_drive_id"],
                              includeItemsFromAllDrives=True, supportsAllDrives=True,
                              corpora="drive")
            if page_token:
                kwargs["pageToken"] = page_token
            results = svc.files().list(**kwargs).execute()
            for f in results.get("files", []):
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    if self.config.get("recursive", True):
                        yield from self._iter_folder(f["id"])
                elif f["mimeType"] in _GOOGLE_EXPORT or f["mimeType"] in _NATIVE_MIME:
                    from datetime import datetime
                    try:
                        mod_ts = datetime.fromisoformat(f.get("modifiedTime","").rstrip("Z")).timestamp()
                    except Exception:
                        import time; mod_ts = time.time()
                    mime = f["mimeType"]
                    if mime in _GOOGLE_EXPORT:
                        export_mime, ext = _GOOGLE_EXPORT[mime]
                        actual_mime = export_mime
                    else:
                        ext = _NATIVE_MIME.get(mime, "")
                        actual_mime = mime
                    yield ConnectorDocument(
                        source_id     = f"gdrive://{f['id']}",
                        source_type   = self.source_type,
                        title         = f["name"] + (ext if not f["name"].endswith(ext) else ""),
                        mime_type     = actual_mime,
                        last_modified = mod_ts,
                        size_bytes    = int(f.get("size", 0) or 0),
                        metadata      = {"file_id": f["id"], "google_mime": mime,
                                         "web_url": f.get("webViewLink","")},
                    )
            page_token = results.get("nextPageToken")
            if not page_token:
                break

    def list(self) -> Generator[ConnectorDocument, None, None]:
        folder_id = self.config.get("folder_id", "root")
        yield from self._iter_folder(folder_id)

    def load(self, doc: ConnectorDocument) -> bytes:
        svc = self._get_service()
        file_id   = doc.metadata["file_id"]
        google_mime = doc.metadata["google_mime"]
        if google_mime in _GOOGLE_EXPORT:
            export_mime, _ = _GOOGLE_EXPORT[google_mime]
            return svc.files().export_media(fileId=file_id, mimeType=export_mime).execute()
        else:
            return svc.files().get_media(fileId=file_id).execute()

    def test_connection(self) -> bool:
        try:
            self._get_service().about().get(fields="user").execute()
            return True
        except Exception:
            return False
