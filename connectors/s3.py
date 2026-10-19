"""
connectors/s3.py
-----------------
AWS S3 / MinIO / Azure Blob (S3-compatible) connector.
Streams objects lazily — never downloads all at once.
Requires: boto3  (pip install boto3)
"""
from __future__ import annotations
from typing import Generator
from connectors.base import BaseConnector, ConnectorDocument

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".csv"}

class S3Connector(BaseConnector):
    source_type = "s3"

    def __init__(self, config: dict):
        """
        config keys:
          bucket_name   : str  (required)
          prefix        : str  — key prefix to scan, e.g. "documents/" (default "")
          region        : str  — AWS region (default "us-east-1")
          endpoint_url  : str  — for MinIO/custom S3-compatible (optional)
          aws_access_key: str  — if not using IAM role
          aws_secret_key: str
          extensions    : list — filter by extension (default SUPPORTED_EXTS)
        """
        super().__init__(config)
        self.bucket     = config["bucket_name"]
        self.prefix     = config.get("prefix", "")
        self.extensions = set(config.get("extensions", list(SUPPORTED_EXTS)))
        self._s3 = None  # lazy init

    def _client(self):
        if self._s3 is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 is required for S3Connector: pip install boto3")
            kwargs = dict(region_name=self.config.get("region", "us-east-1"))
            if self.config.get("endpoint_url"):
                kwargs["endpoint_url"] = self.config["endpoint_url"]
            if self.config.get("aws_access_key"):
                kwargs["aws_access_key_id"]     = self.config["aws_access_key"]
                kwargs["aws_secret_access_key"]  = self.config["aws_secret_key"]
            self._s3 = boto3.client("s3", **kwargs)
        return self._s3

    def list(self) -> Generator[ConnectorDocument, None, None]:
        """
        Use paginator — handles buckets with millions of objects efficiently.
        Each page returns up to 1000 objects; memory usage stays constant.
        """
        paginator = self._client().get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
                if ext not in self.extensions:
                    continue
                yield ConnectorDocument(
                    source_id     = f"s3://{self.bucket}/{key}",
                    source_type   = self.source_type,
                    title         = key.split("/")[-1],
                    mime_type     = _s3_mime(ext),
                    last_modified = obj["LastModified"].timestamp(),
                    size_bytes    = obj["Size"],
                    metadata      = {"bucket": self.bucket, "key": key, "etag": obj.get("ETag","")},
                )

    def load(self, doc: ConnectorDocument) -> bytes:
        key = doc.metadata["key"]
        resp = self._client().get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def test_connection(self) -> bool:
        try:
            self._client().head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False

def _s3_mime(ext: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain", ".md": "text/plain",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".csv": "text/csv",
    }.get(ext, "application/octet-stream")
