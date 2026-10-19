"""
tests/conftest.py
-----------------
Shared pytest fixtures. Mocks ALL external dependencies so every test
runs without live services: ChromaDB, Ollama, LangGraph, Redis, Celery,
boto3, msal, google-api, sqlalchemy.
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Complete list of modules to mock — covers original + all new deps
_HEAVY_MOCKS = [
    # ML / Vector / LLM
    "chromadb", "chromadb.config",
    "langgraph", "langgraph.graph",
    "langchain", "langchain_community",
    "langchain_ollama",
    "sentence_transformers",
    "ragas", "datasets",
    # Document parsing
    "docling", "docling.document_converter",
    "fitz", "pytesseract",
    # Cloud connectors
    "boto3", "boto3.session",
    "botocore", "botocore.exceptions",
    "msal",
    "googleapiclient", "googleapiclient.discovery",
    "google", "google.oauth2", "google.oauth2.service_account",
    # Database
    "sqlalchemy", "sqlalchemy.create_engine",
    # Job queue
    "celery", "celery.app",
    "redis",
    # Auth extras
    "jose", "passlib",
]


@pytest.fixture(autouse=True)
def mock_heavy_deps(monkeypatch):
    """Auto-applied to every test — mocks all external services."""
    mock_chroma = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 42
    mock_collection.get.return_value = {
        "ids": [], "metadatas": [], "documents": []
    }
    mock_collection.query.return_value = {
        "ids": [["chunk_001", "chunk_002"]],
        "documents": [["Doc text one", "Doc text two"]],
        "metadatas": [[
            {"doc_title": "SampleDoc", "source_path": "/tmp/doc.pdf",
             "page_number": 1, "layout_type": "text", "doc_type": "pdf"},
            {"doc_title": "SampleDoc", "source_path": "/tmp/doc.pdf",
             "page_number": 2, "layout_type": "text", "doc_type": "pdf"},
        ]],
        "distances": [[0.12, 0.28]],
    }
    mock_chroma.HttpClient.return_value.get_or_create_collection.return_value = mock_collection
    monkeypatch.setitem(sys.modules, "chromadb", mock_chroma)
    monkeypatch.setitem(sys.modules, "chromadb.config", MagicMock())

    for mod in _HEAVY_MOCKS:
        if mod not in ("chromadb", "chromadb.config"):
            sys.modules.setdefault(mod, MagicMock())

    yield mock_chroma


# ── Document fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_text_doc(tmp_path) -> Path:
    doc = tmp_path / "refund_policy.txt"
    doc.write_text(
        "REFUND POLICY\n\n"
        "All products may be returned within 30 days of purchase with a valid receipt. "
        "Items must be in their original condition and packaging. "
        "Electronics have a separate 15-day return window due to manufacturer policy. "
        "Gift cards, downloadable software, and personalised items are non-refundable. "
        "To initiate a return, contact customer support at returns@company.com "
        "or visit your nearest store with the original receipt."
    )
    return doc


@pytest.fixture
def safety_manual_doc(tmp_path) -> Path:
    doc = tmp_path / "safety_manual.txt"
    doc.write_text(
        "SAFETY MANUAL — Production Floor\n\n"
        "Personal Protective Equipment (PPE) is mandatory for all personnel. "
        "Steel-toed boots must be worn at all times on the production floor. "
        "Safety goggles are required when operating machinery. "
        "Hearing protection is mandatory in zones marked with the noise symbol. "
        "Emergency exits are located on the north and south walls. "
        "In case of fire, activate the nearest alarm and evacuate via the marked exits. "
        "Do not use lifts during an evacuation."
    )
    return doc


@pytest.fixture
def multi_page_doc(tmp_path) -> Path:
    doc = tmp_path / "employee_handbook.txt"
    content = "\n\n".join([
        f"SECTION {i}\n" + (f"Content for section {i} paragraph {j}. " * 5)
        for i in range(1, 6)
        for j in range(1, 4)
    ])
    doc.write_text(content)
    return doc
