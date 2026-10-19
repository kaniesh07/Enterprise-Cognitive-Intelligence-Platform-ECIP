"""Structured logging: JSON lines in containers/cloud (LOG_FORMAT=json), pretty console locally."""
from __future__ import annotations

import logging
import os
import sys

import structlog

_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    _configured = True
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    fmt = os.getenv("LOG_FORMAT", "json" if not sys.stderr.isatty() else "console").lower()
    logging.basicConfig(level=lvl, stream=sys.stdout, format="%(message)s")
    renderer = (structlog.processors.JSONRenderer() if fmt == "json"
                else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        cache_logger_on_first_use=True,
    )
    # Silence chatty libraries
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
