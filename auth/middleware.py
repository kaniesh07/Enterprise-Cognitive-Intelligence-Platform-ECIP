"""
auth/middleware.py
------------------
API-key authentication + role-based access control (RBAC).

Roles:  reader (chat/query) < writer (+ ingest) < admin (+ delete, connectors, eval)

Keys come from API_KEYS="key:role,key:role". For production, swap `_load_keys`
for a secrets-manager / database lookup.

Auth is enforced per-endpoint through the `require_*` FastAPI dependencies.
(The old global middleware also rejected CORS pre-flight OPTIONS requests and
static files, which broke browsers calling the API from another origin.)
"""
from __future__ import annotations

import hmac
import os
from functools import lru_cache
from typing import Optional

import structlog
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = structlog.get_logger()

_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_ROLE_WEIGHTS = {"reader": 1, "writer": 2, "admin": 3}


def _load_keys() -> dict[str, str]:
    raw = os.getenv("API_KEYS")
    if raw is None:
        try:
            from config import settings
            raw = settings.api_keys
        except Exception:
            raw = "ecip-dev-key:admin"
    store: dict[str, str] = {}
    quotes = "\"' \t\r\n"
    for entry in raw.strip(quotes).split(","):
        entry = entry.strip(quotes)
        if not entry:
            continue
        key, role = entry.rsplit(":", 1) if ":" in entry else (entry, "admin")  # bare key = admin
        key, role = key.strip(quotes), role.strip(quotes).lower()
        if key and role in _ROLE_WEIGHTS:
            store[key] = role
    try:
        import structlog
        log = structlog.get_logger()
        if store:
            log.info("api_keys_loaded", count=len(store), roles=sorted(set(store.values())))
        else:
            log.error("api_keys_none_valid",
                      hint="API_KEYS has no valid entries. Use the form key:role, e.g. mysecret:admin")
    except Exception:
        pass
    return store


@lru_cache(maxsize=1)
def _key_store() -> dict[str, str]:
    return _load_keys()


def _get_role(api_key: Optional[str]) -> Optional[str]:
    if not api_key:
        return None
    for key, role in _key_store().items():
        if hmac.compare_digest(key.encode(), api_key.encode()):   # constant-time
            return role
    return None


def _auth_failed():
    try:
        from monitoring.metrics import get_registry
        get_registry().auth_failures.inc()
    except Exception:
        pass


def _check(api_key: Optional[str], min_role: str) -> str:
    role = _get_role(api_key)
    if not role:
        _auth_failed()
        raise HTTPException(status_code=401, detail="Missing or invalid API key. Set the X-API-Key header.")
    if _ROLE_WEIGHTS.get(role, 0) < _ROLE_WEIGHTS.get(min_role, 99):
        raise HTTPException(status_code=403, detail=f"Role '{role}' insufficient. Required: '{min_role}'.")
    return role


async def require_auth(api_key: str = Security(_KEY_HEADER)) -> str:
    """Any valid key (reader/writer/admin)."""
    return _check(api_key, "reader")


async def require_writer(api_key: str = Security(_KEY_HEADER)) -> str:
    return _check(api_key, "writer")


async def require_admin(api_key: str = Security(_KEY_HEADER)) -> str:
    return _check(api_key, "admin")


# Optional global middleware (not installed by default — kept for compatibility).
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402


class APIKeyMiddleware(BaseHTTPMiddleware):
    _OPEN_PATHS = {"/", "/health", "/ready", "/docs", "/redoc", "/openapi.json", "/metrics",
                   "/favicon.ico"}
    _OPEN_PREFIXES = ("/static/", "/docs/")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (request.method != "OPTIONS" and path not in self._OPEN_PATHS
                and not path.startswith(self._OPEN_PREFIXES)):
            key = request.headers.get("X-API-Key")
            role = _get_role(key)
            if not role:
                from fastapi.responses import JSONResponse
                _auth_failed()
                return JSONResponse({"detail": "Missing or invalid API key."}, status_code=401)
            request.state.role = role
        return await call_next(request)
