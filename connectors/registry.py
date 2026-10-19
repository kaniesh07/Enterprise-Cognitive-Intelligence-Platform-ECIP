"""
connectors/registry.py
-----------------------
Connector registry — stores named connector configurations and
instantiates the correct connector class on demand.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import structlog
from connectors.base import BaseConnector

logger = structlog.get_logger()

_CONNECTOR_MAP: dict[str, type] = {}

def _get_map() -> dict[str, type]:
    if not _CONNECTOR_MAP:
        from connectors.local_fs    import LocalFSConnector
        from connectors.s3          import S3Connector
        from connectors.sharepoint  import SharePointConnector
        from connectors.google_drive import GoogleDriveConnector
        from connectors.confluence  import ConfluenceConnector
        from connectors.database    import DatabaseConnector
        from connectors.web         import WebConnector
        _CONNECTOR_MAP.update({
            "local":      LocalFSConnector,
            "s3":         S3Connector,
            "sharepoint": SharePointConnector,
            "gdrive":     GoogleDriveConnector,
            "confluence": ConfluenceConnector,
            "sql":        DatabaseConnector,
            "web":        WebConnector,
        })
    return _CONNECTOR_MAP


class ConnectorRegistry:
    """
    Stores connector configs by name.
    Persists to JSON so connectors survive restarts.
    """
    def __init__(self, store_path: str = "./data/processed/connectors.json"):
        self._path = Path(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._configs: dict[str, dict] = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                return {}
        return {}

    def _save(self):
        self._path.write_text(json.dumps(self._configs, indent=2))

    def register(self, name: str, connector_type: str, config: dict) -> dict:
        """Register a new connector configuration."""
        if connector_type not in _get_map():
            raise ValueError(f"Unknown connector type '{connector_type}'. "
                             f"Valid types: {list(_get_map().keys())}")
        self._configs[name] = {"type": connector_type, "config": config, "enabled": True}
        self._save()
        logger.info("connector_registered", name=name, type=connector_type)
        return self._configs[name]

    def get(self, name: str) -> Optional[BaseConnector]:
        """Instantiate and return a connector by name."""
        entry = self._configs.get(name)
        if not entry:
            return None
        cls = _get_map().get(entry["type"])
        if not cls:
            raise ValueError(f"Connector type '{entry['type']}' not available.")
        return cls(entry["config"])

    def list_all(self) -> list[dict]:
        return [
            {"name": name, "type": entry["type"], "enabled": entry.get("enabled", True)}
            for name, entry in self._configs.items()
        ]

    def remove(self, name: str):
        self._configs.pop(name, None)
        self._save()

    def supported_types(self) -> list[str]:
        return list(_get_map().keys())


_registry: Optional[ConnectorRegistry] = None

def get_connector_registry() -> ConnectorRegistry:
    global _registry
    if _registry is None:
        _registry = ConnectorRegistry(
            str(__import__("config").settings.connector_store)
        )
    return _registry
