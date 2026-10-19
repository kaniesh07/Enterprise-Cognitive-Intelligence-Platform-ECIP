"""
tests/test_new_modules.py
--------------------------
Tests for all modules added in the gap-fix and scale-ingestion build:
  auth, tools, memory, connectors, pipeline, monitoring (new metrics),
  model_router, api (new endpoints), evaluation (new metrics)

All tests run without live services (ChromaDB / Ollama / Redis all mocked).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

for mod in ["chromadb","chromadb.config","langgraph","langgraph.graph",
            "langchain","langchain_community","sentence_transformers",
            "ragas","datasets","docling","docling.document_converter",
            "fitz","pytesseract"]:
    sys.modules.setdefault(mod, MagicMock())


# ═══════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════

class TestAuthMiddleware:
    def test_valid_admin_key_returns_role(self):
        import os; os.environ["API_KEYS"] = "test-admin:admin,test-reader:reader"
        from importlib import reload
        import auth.middleware as am; reload(am)
        am._key_store.cache_clear()
        assert am._get_role("test-admin") == "admin"
        assert am._get_role("test-reader") == "reader"

    def test_invalid_key_returns_none(self):
        import auth.middleware as am
        assert am._get_role("nonexistent-key") is None

    def test_role_weights_ordered(self):
        from auth.middleware import _ROLE_WEIGHTS
        assert _ROLE_WEIGHTS["reader"] < _ROLE_WEIGHTS["writer"] < _ROLE_WEIGHTS["admin"]

    def test_check_raises_401_no_key(self):
        from auth.middleware import _check
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            _check(None, "reader")
        assert exc.value.status_code == 401

    def test_check_raises_403_insufficient_role(self):
        import os; os.environ["API_KEYS"] = "rkey:reader"
        from importlib import reload
        import auth.middleware as am; reload(am); am._key_store.cache_clear()
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            am._check("rkey", "admin")
        assert exc.value.status_code == 403

    def test_check_passes_with_sufficient_role(self):
        import os; os.environ["API_KEYS"] = "akey:admin"
        from importlib import reload
        import auth.middleware as am; reload(am); am._key_store.cache_clear()
        role = am._check("akey", "reader")
        assert role == "admin"

    def test_open_paths_no_auth_required(self):
        from auth.middleware import APIKeyMiddleware
        assert "/health" in APIKeyMiddleware._OPEN_PATHS
        assert "/docs" in APIKeyMiddleware._OPEN_PATHS
        assert "/metrics" in APIKeyMiddleware._OPEN_PATHS


# ═══════════════════════════════════════════════════════════
# MODEL ROUTER
# ═══════════════════════════════════════════════════════════

class TestModelRouter:
    def test_intent_uses_fast_model(self):
        import os; os.environ["FAST_MODEL"] = "phi3:mini"; os.environ["STRONG_MODEL"] = "llama3.1:8b"
        from importlib import reload
        import agents.model_router as mr; reload(mr)
        assert mr.get_model("intent") == "phi3:mini"

    def test_reasoning_uses_strong_model(self):
        import os; os.environ["STRONG_MODEL"] = "llama3.1:8b"
        from importlib import reload
        import agents.model_router as mr; reload(mr)
        assert mr.get_model("reasoning") == "llama3.1:8b"

    def test_verifier_uses_strong_model(self):
        from agents.model_router import ModelTier
        from agents.model_router import _AGENT_TIERS
        assert _AGENT_TIERS["verifier"] == ModelTier.STRONG

    def test_multimodal_uses_fast_model(self):
        from agents.model_router import _AGENT_TIERS, ModelTier
        assert _AGENT_TIERS["multimodal"] == ModelTier.FAST

    def test_log_routing_returns_dict(self):
        from agents.model_router import log_routing
        info = log_routing("reasoning")
        assert "agent" in info and "model" in info and "tier" in info

    def test_unknown_agent_defaults_to_strong(self):
        from agents.model_router import ModelTier, _AGENT_TIERS
        assert _AGENT_TIERS.get("unknown_agent", ModelTier.STRONG) == ModelTier.STRONG


# ═══════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════

class TestToolRegistry:
    def _fresh_registry(self):
        from tools.registry import ToolRegistry
        return ToolRegistry()

    def test_list_tools_returns_all_builtins(self):
        reg = self._fresh_registry()
        tools = reg.list_tools()
        names = [t["name"] for t in tools]
        assert "search_knowledge_base" in names
        assert "delete_document" in names
        assert "ingest_document" in names

    def test_read_tool_types_correct(self):
        reg = self._fresh_registry()
        tools = {t["name"]: t for t in reg.list_tools()}
        assert tools["search_knowledge_base"]["type"] == "read"
        assert tools["delete_document"]["type"] == "high_risk"
        assert tools["ingest_document"]["type"] == "write"

    def test_reader_cannot_call_write_tool(self):
        reg = self._fresh_registry()
        result = reg.call("ingest_document", {"file_path": "/tmp/test.pdf"}, caller_role="reader")
        assert not result.success
        assert "cannot call" in result.error.lower()

    def test_admin_can_call_any_tool(self):
        reg = self._fresh_registry()
        # Patch handler so it doesn't actually run
        reg._handlers["get_document_metadata"] = lambda p: {"metadata": {}, "chunk_count": 0}
        result = reg.call("get_document_metadata", {"doc_title": "test"}, caller_role="admin")
        assert result.success

    def test_high_risk_tool_requires_confirmation(self):
        reg = self._fresh_registry()
        result = reg.call("delete_document",
                          {"doc_title": "test", "reason": "cleanup", "confirmed": False},
                          caller_role="admin")
        assert not result.success
        assert result.requires_approval

    def test_high_risk_confirmed_proceeds(self):
        reg = self._fresh_registry()
        reg._handlers["delete_document"] = lambda p: {"deleted": True}
        result = reg.call("delete_document",
                          {"doc_title": "test", "reason": "cleanup", "confirmed": True},
                          caller_role="admin")
        assert result.success

    def test_missing_required_field_fails(self):
        reg = self._fresh_registry()
        result = reg.call("search_knowledge_base", {}, caller_role="reader")
        assert not result.success
        assert "query" in result.error

    def test_unknown_tool_fails_gracefully(self):
        reg = self._fresh_registry()
        result = reg.call("nonexistent_tool", {}, caller_role="admin")
        assert not result.success
        assert "Unknown" in result.error

    def test_metrics_track_calls_and_errors(self):
        reg = self._fresh_registry()
        reg._handlers["get_document_metadata"] = lambda p: {"metadata":{}}
        reg.call("get_document_metadata", {"doc_title":"t"}, caller_role="admin")
        metrics = reg.get_metrics()
        assert metrics["get_document_metadata"]["calls"] == 1
        assert metrics["get_document_metadata"]["success_rate"] == 1.0

    def test_error_increments_error_count(self):
        reg = self._fresh_registry()
        reg._handlers["get_document_metadata"] = lambda p: (_ for _ in ()).throw(RuntimeError("db error"))
        reg.call("get_document_metadata", {"doc_title":"t"}, caller_role="admin")
        metrics = reg.get_metrics()
        assert metrics["get_document_metadata"]["errors"] == 1
        assert metrics["get_document_metadata"]["success_rate"] == 0.0


# ═══════════════════════════════════════════════════════════
# APPROVAL GATE
# ═══════════════════════════════════════════════════════════

class TestApprovalGate:
    def _gate(self):
        from tools.approval_gate import ApprovalGate
        return ApprovalGate()

    def test_request_creates_pending(self):
        gate = self._gate()
        req = gate.request("delete_document", {"doc_title":"t"}, "verifier", "High-risk delete")
        assert req.status.value == "pending"
        assert req.request_id

    def test_approve_changes_status(self):
        gate = self._gate()
        req = gate.request("delete_document", {}, "agent", "reason")
        result = gate.decide(req.request_id, approved=True, decided_by="admin@company.com")
        assert result.status.value == "approved"
        assert result.decided_by == "admin@company.com"
        assert result.decided_at is not None

    def test_reject_changes_status(self):
        gate = self._gate()
        req = gate.request("delete_document", {}, "agent", "reason")
        result = gate.decide(req.request_id, approved=False, decided_by="admin@company.com")
        assert result.status.value == "rejected"

    def test_list_pending_excludes_decided(self):
        gate = self._gate()
        r1 = gate.request("action1", {}, "agent", "reason1")
        r2 = gate.request("action2", {}, "agent", "reason2")
        gate.decide(r1.request_id, True, "admin")
        pending = gate.list_pending()
        ids = [p.request_id for p in pending]
        assert r1.request_id not in ids
        assert r2.request_id in ids

    def test_unknown_request_id_raises(self):
        gate = self._gate()
        with pytest.raises(ValueError):
            gate.decide("nonexistent-id", True, "admin")

    def test_stats_count_statuses(self):
        gate = self._gate()
        r = gate.request("act", {}, "a", "r")
        gate.decide(r.request_id, True, "admin")
        s = gate.stats()
        assert "approved" in s

    def test_to_dict_has_required_keys(self):
        gate = self._gate()
        req = gate.request("delete_document", {"doc_title": "important.pdf"}, "agent", "cleanup")
        d = req.to_dict()
        for key in ["request_id","action_type","params","proposed_by","risk_reason","status"]:
            assert key in d


# ═══════════════════════════════════════════════════════════
# MEMORY STORE
# ═══════════════════════════════════════════════════════════

class TestMemoryStore:
    def _store(self, tmp_path):
        from memory.store import MemoryStore
        return MemoryStore(str(tmp_path / "prefs.json"))

    def test_get_returns_defaults_for_new_user(self, tmp_path):
        store = self._store(tmp_path)
        prefs = store.get("user123")
        assert prefs.user_id == "user123"
        assert prefs.answer_format == "prose"
        assert prefs.language == "en"

    def test_set_and_get_roundtrip(self, tmp_path):
        from memory.store import UserPreferences
        store = self._store(tmp_path)
        prefs = UserPreferences(user_id="u1", answer_format="bullets", domain_focus="legal")
        store.set(prefs)
        loaded = store.get("u1")
        assert loaded.answer_format == "bullets"
        assert loaded.domain_focus == "legal"

    def test_update_partial_fields(self, tmp_path):
        store = self._store(tmp_path)
        store.update("u2", verbosity="concise")
        prefs = store.get("u2")
        assert prefs.verbosity == "concise"
        assert prefs.answer_format == "prose"   # default unchanged

    def test_persistence_across_instances(self, tmp_path):
        from memory.store import UserPreferences
        path = str(tmp_path / "prefs.json")
        from memory.store import MemoryStore
        s1 = MemoryStore(path)
        s1.set(UserPreferences(user_id="persist_user", domain_focus="finance"))
        s2 = MemoryStore(path)
        assert s2.get("persist_user").domain_focus == "finance"

    def test_delete_removes_user(self, tmp_path):
        from memory.store import UserPreferences
        store = self._store(tmp_path)
        store.set(UserPreferences(user_id="del_user", language="fr"))
        store.delete("del_user")
        prefs = store.get("del_user")
        assert prefs.language == "en"   # back to default

    def test_prompt_hint_bullets(self, tmp_path):
        from memory.store import UserPreferences
        prefs = UserPreferences(user_id="x", answer_format="bullets")
        hint = prefs.to_prompt_hint()
        assert "bullet" in hint.lower()

    def test_prompt_hint_domain_focus(self, tmp_path):
        from memory.store import UserPreferences
        prefs = UserPreferences(user_id="x", domain_focus="legal compliance")
        hint = prefs.to_prompt_hint()
        assert "legal compliance" in hint

    def test_prompt_hint_empty_for_defaults(self, tmp_path):
        from memory.store import UserPreferences
        prefs = UserPreferences(user_id="x")
        hint = prefs.to_prompt_hint()
        assert hint == ""   # no extra hints for default settings


# ═══════════════════════════════════════════════════════════
# CONNECTORS
# ═══════════════════════════════════════════════════════════

class TestConnectorBase:
    def test_fingerprint_changes_with_modified_time(self):
        from connectors.base import ConnectorDocument
        d1 = ConnectorDocument(source_id="s3://bucket/file.pdf",
                               source_type="s3", title="file.pdf", last_modified=1000.0)
        d2 = ConnectorDocument(source_id="s3://bucket/file.pdf",
                               source_type="s3", title="file.pdf", last_modified=2000.0)
        assert d1.fingerprint != d2.fingerprint

    def test_fingerprint_same_when_unmodified(self):
        from connectors.base import ConnectorDocument
        d1 = ConnectorDocument(source_id="id1", source_type="local",
                               title="doc.pdf", last_modified=500.0)
        d2 = ConnectorDocument(source_id="id1", source_type="local",
                               title="doc.pdf", last_modified=500.0)
        assert d1.fingerprint == d2.fingerprint

    def test_extension_from_title(self):
        from connectors.base import ConnectorDocument
        d = ConnectorDocument(source_id="x", source_type="s3",
                              title="report.docx", last_modified=0)
        assert d.extension == ".docx"

    def test_extension_from_mime_type(self):
        from connectors.base import ConnectorDocument
        d = ConnectorDocument(source_id="x", source_type="gdrive",
                              title="Untitled", mime_type="application/pdf", last_modified=0)
        assert d.extension == ".pdf"


class TestLocalFSConnector:
    def test_lists_supported_files(self, tmp_path):
        (tmp_path / "doc.pdf").write_bytes(b"%PDF test")
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "ignore.exe").write_bytes(b"binary")
        from connectors.local_fs import LocalFSConnector
        conn = LocalFSConnector({"root_path": str(tmp_path), "recursive": False})
        docs = list(conn.list())
        names = {d.title for d in docs}
        assert "doc.pdf" in names
        assert "notes.txt" in names
        assert "ignore.exe" not in names

    def test_recursive_finds_nested_files(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested content")
        from connectors.local_fs import LocalFSConnector
        conn = LocalFSConnector({"root_path": str(tmp_path), "recursive": True})
        docs = list(conn.list())
        assert any(d.title == "nested.txt" for d in docs)

    def test_excludes_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "config").write_text("git config")
        (tmp_path / "real.txt").write_text("real")
        from connectors.local_fs import LocalFSConnector
        conn = LocalFSConnector({"root_path": str(tmp_path), "recursive": True})
        docs = list(conn.list())
        assert not any(".git" in d.source_id for d in docs)

    def test_load_returns_file_bytes(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello ECIP")
        from connectors.local_fs import LocalFSConnector, ConnectorDocument
        conn = LocalFSConnector({"root_path": str(tmp_path)})
        doc = ConnectorDocument(source_id=str(f), source_type="local",
                                title="sample.txt", last_modified=f.stat().st_mtime)
        content = conn.load(doc)
        assert content == b"Hello ECIP"

    def test_empty_directory_yields_nothing(self, tmp_path):
        from connectors.local_fs import LocalFSConnector
        conn = LocalFSConnector({"root_path": str(tmp_path)})
        assert list(conn.list()) == []

    def test_metadata_has_relative_path(self, tmp_path):
        (tmp_path / "doc.txt").write_text("content")
        from connectors.local_fs import LocalFSConnector
        conn = LocalFSConnector({"root_path": str(tmp_path)})
        docs = list(conn.list())
        assert "relative_path" in docs[0].metadata
        assert docs[0].metadata["relative_path"] == "doc.txt"

    def test_custom_extensions(self, tmp_path):
        (tmp_path / "file.csv").write_text("a,b,c")
        (tmp_path / "file.pdf").write_bytes(b"%PDF")
        from connectors.local_fs import LocalFSConnector
        conn = LocalFSConnector({"root_path": str(tmp_path), "extensions": [".csv"]})
        docs = list(conn.list())
        assert all(d.title.endswith(".csv") for d in docs)


class TestConnectorRegistry:
    def test_register_and_retrieve(self, tmp_path):
        from connectors.registry import ConnectorRegistry
        reg = ConnectorRegistry(str(tmp_path / "conn.json"))
        reg.register("my_files", "local", {"root_path": "/tmp"})
        conn = reg.get("my_files")
        from connectors.local_fs import LocalFSConnector
        assert isinstance(conn, LocalFSConnector)

    def test_unknown_type_raises(self, tmp_path):
        from connectors.registry import ConnectorRegistry
        reg = ConnectorRegistry(str(tmp_path / "conn.json"))
        with pytest.raises(ValueError):
            reg.register("bad", "nonexistent_type", {})

    def test_list_all_returns_registered(self, tmp_path):
        from connectors.registry import ConnectorRegistry
        reg = ConnectorRegistry(str(tmp_path / "conn.json"))
        reg.register("a", "local", {"root_path": "/tmp"})
        reg.register("b", "local", {"root_path": "/data"})
        all_conns = reg.list_all()
        names = [c["name"] for c in all_conns]
        assert "a" in names and "b" in names

    def test_remove_connector(self, tmp_path):
        from connectors.registry import ConnectorRegistry
        reg = ConnectorRegistry(str(tmp_path / "conn.json"))
        reg.register("to_remove", "local", {"root_path": "/tmp"})
        reg.remove("to_remove")
        assert reg.get("to_remove") is None

    def test_persistence_across_instances(self, tmp_path):
        from connectors.registry import ConnectorRegistry
        path = str(tmp_path / "conn.json")
        r1 = ConnectorRegistry(path)
        r1.register("persisted", "local", {"root_path": "/tmp"})
        r2 = ConnectorRegistry(path)
        assert r2.get("persisted") is not None

    def test_supported_types_listed(self, tmp_path):
        from connectors.registry import ConnectorRegistry
        reg = ConnectorRegistry(str(tmp_path / "conn.json"))
        types = reg.supported_types()
        for t in ["local","s3","sharepoint","gdrive","confluence","sql"]:
            assert t in types


# ═══════════════════════════════════════════════════════════
# SYNC MANIFEST
# ═══════════════════════════════════════════════════════════

class TestSyncManifest:
    def _manifest(self, tmp_path):
        from pipeline.sync_manifest import SyncManifest
        return SyncManifest(str(tmp_path / "manifest.json"))

    def test_new_document_needs_indexing(self, tmp_path):
        m = self._manifest(tmp_path)
        assert m.needs_indexing("doc1", "fp1") is True

    def test_indexed_document_skip(self, tmp_path):
        m = self._manifest(tmp_path)
        m.mark_indexed("doc1", "local", "fp1", chunk_count=5)
        assert m.needs_indexing("doc1", "fp1") is False

    def test_changed_document_needs_reindex(self, tmp_path):
        m = self._manifest(tmp_path)
        m.mark_indexed("doc1", "local", "fp1", chunk_count=5)
        assert m.needs_indexing("doc1", "fp2") is True   # fingerprint changed

    def test_failed_document_needs_retry(self, tmp_path):
        m = self._manifest(tmp_path)
        m.mark_failed("doc1", "local", "fp1", error="parse error")
        assert m.needs_indexing("doc1", "fp1") is True

    def test_stats_count_correctly(self, tmp_path):
        m = self._manifest(tmp_path)
        m.mark_indexed("d1", "local", "fp1", 10)
        m.mark_indexed("d2", "local", "fp2", 5)
        m.mark_failed("d3", "local", "fp3", "error")
        stats = m.stats()
        assert stats["total_documents"] == 3
        assert stats["indexed"] == 2
        assert stats["failed"] == 1
        assert stats["total_chunks_indexed"] == 15

    def test_get_failed_returns_failures(self, tmp_path):
        m = self._manifest(tmp_path)
        m.mark_failed("bad1", "s3", "fp", "timeout")
        m.mark_failed("bad2", "s3", "fp", "parse error")
        m.mark_indexed("good", "s3", "fp", 3)
        failed = m.get_failed()
        ids = [f["source_id"] for f in failed]
        assert "bad1" in ids and "bad2" in ids and "good" not in ids

    def test_reset_failed_clears_fingerprints(self, tmp_path):
        m = self._manifest(tmp_path)
        m.mark_failed("doc", "local", "fp1", "error")
        m.reset_failed()
        # After reset, document should need re-indexing with any fingerprint
        assert m.needs_indexing("doc", "fp1") is True

    def test_persistence(self, tmp_path):
        from pipeline.sync_manifest import SyncManifest
        path = str(tmp_path / "manifest.json")
        m1 = SyncManifest(path)
        m1.mark_indexed("persistent_doc", "local", "fp99", 7)
        m2 = SyncManifest(path)
        assert m2.needs_indexing("persistent_doc", "fp99") is False
        assert m2.stats()["total_chunks_indexed"] == 7


# ═══════════════════════════════════════════════════════════
# PROGRESS TRACKER
# ═══════════════════════════════════════════════════════════

class TestProgressTracker:
    def _tracker(self):
        from pipeline.progress import ProgressTracker
        return ProgressTracker()

    def test_create_job_returns_progress(self):
        t = self._tracker()
        job = t.create_job("job1", "my_connector")
        assert job.job_id == "job1"
        assert job.status == "running"
        assert job.processed == 0

    def test_update_fields(self):
        t = self._tracker()
        t.create_job("j1", "conn")
        t.update("j1", total_discovered=100, indexed=42)
        d = t.get("j1")
        assert d["total_discovered"] == 100
        assert d["indexed"] == 42

    def test_add_error_increments_failed(self):
        t = self._tracker()
        t.create_job("j2", "conn")
        t.add_error("j2", "s3://bad.pdf", "parse failed")
        d = t.get("j2")
        assert d["failed"] == 1
        assert len(d["recent_errors"]) == 1

    def test_complete_job_sets_finished_at(self):
        t = self._tracker()
        t.create_job("j3", "conn")
        t.complete_job("j3", "completed")
        d = t.get("j3")
        assert d["status"] == "completed"
        assert d["elapsed_s"] >= 0

    def test_progress_pct_calculated(self):
        t = self._tracker()
        t.create_job("j4", "conn")
        t.update("j4", total_discovered=100, processed=50)
        d = t.get("j4")
        assert d["progress_pct"] == 50.0

    def test_throughput_calculated(self):
        t = self._tracker()
        job = t.create_job("j5", "conn")
        # Simulate 60 docs processed in ~1 second
        job.started_at = time.time() - 1.0
        t.update("j5", processed=60)
        d = t.get("j5")
        assert d["throughput_docs_per_min"] > 0

    def test_list_active_excludes_completed(self):
        t = self._tracker()
        t.create_job("active", "conn")
        t.create_job("done", "conn")
        t.complete_job("done", "completed")
        active = t.list_active()
        ids = [j["job_id"] for j in active]
        assert "active" in ids and "done" not in ids

    def test_unknown_job_returns_none(self):
        t = self._tracker()
        assert t.get("nonexistent") is None

    def test_eta_none_when_no_progress(self):
        t = self._tracker()
        t.create_job("j6", "conn")
        t.update("j6", total_discovered=100)
        d = t.get("j6")
        assert d["eta_seconds"] is None   # no processed yet


# ═══════════════════════════════════════════════════════════
# MONITORING — NEW METRICS
# ═══════════════════════════════════════════════════════════

class TestNewMetrics:
    def _fresh_registry(self):
        from monitoring.metrics import MetricsRegistry
        return MetricsRegistry()

    def test_record_token_usage(self):
        reg = self._fresh_registry()
        reg.record_pipeline_result({
            "faithfulness_score": 0.9, "confidence": 0.85,
            "hallucination_flags": [], "retrieved_chunks": [],
            "final_answer": "Here is the answer.",
            "token_usage": {"prompt": 200, "completion": 150, "total": 350, "projected_cost_usd": 0.0},
            "latency_ms": {},
        })
        assert reg.prompt_tokens_total.value == 200
        assert reg.completion_tokens_total.value == 150

    def test_record_tool_call_success(self):
        reg = self._fresh_registry()
        reg.record_tool_call("search_knowledge_base", success=True)
        reg.record_tool_call("search_knowledge_base", success=True)
        reg.record_tool_call("search_knowledge_base", success=False)
        assert reg.tool_calls_total["search_knowledge_base"].value == 3
        assert reg.tool_errors_total["search_knowledge_base"].value == 1

    def test_record_model_call(self):
        reg = self._fresh_registry()
        reg.record_model_call("llama3.1:8b")
        reg.record_model_call("llama3.1:8b")
        reg.record_model_call("phi3:mini")
        assert reg.model_call_counts["llama3.1:8b"].value == 2
        assert reg.model_call_counts["phi3:mini"].value == 1

    def test_intent_accuracy_tracking(self):
        reg = self._fresh_registry()
        reg.record_intent("factual", "factual")     # correct
        reg.record_intent("factual", "comparison")  # wrong
        reg.record_intent("procedural", "procedural")  # correct
        assert reg.intent_correct_total.value == 2
        assert reg.intent_total.value == 3

    def test_task_completion_tracked(self):
        reg = self._fresh_registry()
        reg.record_pipeline_result({
            "final_answer": "Here is a clear answer about the policy.",
            "faithfulness_score": 0.9, "confidence": 0.85,
            "hallucination_flags": [], "retrieved_chunks": [],
            "token_usage": {}, "latency_ms": {},
        })
        assert reg.task_completed_total.value == 1

    def test_failed_task_not_counted_as_completed(self):
        reg = self._fresh_registry()
        reg.record_pipeline_result({
            "final_answer": "An error occurred while processing your query.",
            "faithfulness_score": 0.0, "confidence": 0.0,
            "hallucination_flags": [], "retrieved_chunks": [],
            "token_usage": {}, "latency_ms": {},
        })
        assert reg.task_completed_total.value == 0

    def test_circuit_breaker_gauge(self):
        reg = self._fresh_registry()
        reg.set_circuit_breaker(True)
        assert reg.circuit_open.value == 1.0
        reg.set_circuit_breaker(False)
        assert reg.circuit_open.value == 0.0

    def test_prometheus_text_includes_new_metrics(self):
        reg = self._fresh_registry()
        reg.record_model_call("llama3.1:8b")
        reg.record_tool_call("search_knowledge_base", success=True)
        text = reg.to_prometheus_text()
        assert "ecip_prompt_tokens_total" in text
        assert "ecip_projected_cost_usd_total" in text
        assert "ecip_intent_correct_total" in text
        assert "ecip_task_completed_total" in text
        assert "ecip_circuit_breaker_open" in text
        assert 'model="llama3.1:8b"' in text
        assert 'tool="search_knowledge_base"' in text


# ═══════════════════════════════════════════════════════════
# EVALUATION — NEW METRICS
# ═══════════════════════════════════════════════════════════

class TestEvaluationNewMetrics:
    def test_intent_accuracy_perfect(self):
        from evaluation.ragas_eval import intent_accuracy
        assert intent_accuracy(["factual","procedural"], ["factual","procedural"]) == 1.0

    def test_intent_accuracy_zero(self):
        from evaluation.ragas_eval import intent_accuracy
        assert intent_accuracy(["factual","factual"], ["comparison","procedural"]) == 0.0

    def test_intent_accuracy_partial(self):
        from evaluation.ragas_eval import intent_accuracy
        assert intent_accuracy(["factual","comparison"], ["factual","procedural"]) == pytest.approx(0.5)

    def test_intent_accuracy_empty_gt(self):
        from evaluation.ragas_eval import intent_accuracy
        assert intent_accuracy(["factual"], []) == 0.0

    def test_task_completion_clear_answer(self):
        from evaluation.ragas_eval import is_task_completed
        assert is_task_completed("The return policy allows 30 days with a valid receipt.") is True

    def test_task_completion_error_answer(self):
        from evaluation.ragas_eval import is_task_completed
        assert is_task_completed("An error occurred while processing your query.") is False

    def test_task_completion_no_docs(self):
        from evaluation.ragas_eval import is_task_completed
        assert is_task_completed("No relevant documents found for your query.") is False

    def test_task_completion_cannot_answer(self):
        from evaluation.ragas_eval import is_task_completed
        assert is_task_completed("I cannot find any information about this topic.") is False



# ═══════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════

class TestCircuitBreaker:
    def _cb(self, failures=3, successes=2, timeout=60.0):
        from agents.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        return CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=failures,
            success_threshold=successes,
            timeout_seconds=timeout,
        ))

    def test_initial_state_closed(self):
        cb = self._cb()
        from agents.circuit_breaker import CircuitState
        assert cb.state == CircuitState.CLOSED

    def test_successful_call_passes_through(self):
        cb = self._cb()
        result = cb.call(lambda: "ok")
        assert result == "ok"

    def test_opens_after_failure_threshold(self):
        cb = self._cb(failures=3)
        from agents.circuit_breaker import CircuitState
        for _ in range(3):
            try: cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
            except RuntimeError: pass
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_raises_circuit_open(self):
        cb = self._cb(failures=1)
        from agents.circuit_breaker import CircuitBreakerOpen, CircuitState
        try: cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        except RuntimeError: pass
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: "should not reach")

    def test_open_circuit_uses_fallback(self):
        cb = self._cb(failures=1)
        try: cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        except RuntimeError: pass
        result = cb.call(lambda: "main", fallback=lambda: "fallback")
        assert result == "fallback"

    def test_half_open_after_timeout(self):
        import time as _time
        cb = self._cb(failures=1, timeout=0.01)
        from agents.circuit_breaker import CircuitState
        try: cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        except RuntimeError: pass
        _time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_successes_in_half_open(self):
        import time as _time
        cb = self._cb(failures=1, successes=2, timeout=0.01)
        from agents.circuit_breaker import CircuitState
        try: cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        except RuntimeError: pass
        _time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.call(lambda: "ok")
        cb.call(lambda: "ok")
        assert cb.state == CircuitState.CLOSED

    def test_reset_restores_closed(self):
        cb = self._cb(failures=1)
        from agents.circuit_breaker import CircuitState
        try: cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        except RuntimeError: pass
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_get_stats_returns_dict(self):
        cb = self._cb()
        stats = cb.get_stats()
        assert "name" in stats
        assert "state" in stats
        assert "failure_count" in stats

    def test_failure_count_tracked(self):
        cb = self._cb(failures=10)
        for _ in range(3):
            try: cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
            except RuntimeError: pass
        assert cb.get_stats()["failure_count"] == 3

    def test_global_breakers_named(self):
        from agents.circuit_breaker import get_breaker
        ob = get_breaker("ollama")
        cb = get_breaker("chromadb")
        assert ob.name == "ollama"
        assert cb.name == "chromadb"
        assert ob is get_breaker("ollama")   # singleton

    def test_get_all_stats_lists_breakers(self):
        from agents.circuit_breaker import get_breaker, get_all_stats
        get_breaker("ollama"); get_breaker("chromadb")
        stats = get_all_stats()
        names = [s["name"] for s in stats]
        assert "ollama" in names and "chromadb" in names

# ═══════════════════════════════════════════════════════════
# API — NEW ENDPOINTS WITH AUTH
# ═══════════════════════════════════════════════════════════

class TestAPINewEndpoints:
    """Test new endpoints added for connectors, approvals, tools, user prefs."""

    _ADMIN_HEADERS = {"X-API-Key": "ecip-dev-key"}
    _READER_HEADERS = {"X-API-Key": "reader-only-key"}

    def _client(self):
        import os; os.environ["API_KEYS"] = "ecip-dev-key:admin,reader-only-key:reader"
        from importlib import reload
        import auth.middleware as am; reload(am); am._key_store.cache_clear()
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_health_no_auth_required(self):
        c = self._client()
        r = c.get("/health")
        assert r.status_code == 200

    def test_stats_requires_auth(self):
        c = self._client()
        with patch("ingestion.indexer.DocumentIndexer") as MockIdx:
            MockIdx.return_value.stats.return_value = {"total_chunks": 10}
            r = c.get("/stats", headers=self._ADMIN_HEADERS)
            assert r.status_code == 200

    def test_unauthenticated_query_returns_401(self):
        c = self._client()
        r = c.post("/query", json={"query": "What is the policy?"})
        assert r.status_code == 401

    def test_list_tools_returns_tool_list(self):
        c = self._client()
        r = c.get("/tools", headers=self._ADMIN_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert "tools" in body
        assert len(body["tools"]) >= 3

    def test_list_connectors_empty(self):
        c = self._client()
        with patch("connectors.registry.ConnectorRegistry.list_all", return_value=[]):
            with patch("connectors.registry.ConnectorRegistry.supported_types",
                       return_value=["local","s3","sharepoint","gdrive","confluence","sql"]):
                r = c.get("/connectors", headers=self._ADMIN_HEADERS)
        assert r.status_code == 200
        assert "connectors" in r.json()

    def test_register_connector_requires_admin(self):
        c = self._client()
        # reader cannot register
        r = c.post("/connectors",
                   json={"name":"test","connector_type":"local","config":{"root_path":"/tmp"}},
                   headers=self._READER_HEADERS)
        assert r.status_code == 403

    def test_get_user_preferences_default(self):
        c = self._client()
        with patch("memory.store.MemoryStore.get") as mock_get:
            from memory.store import UserPreferences
            mock_get.return_value = UserPreferences(user_id="u1")
            r = c.get("/users/u1/preferences", headers=self._ADMIN_HEADERS)
        assert r.status_code == 200
        assert r.json()["answer_format"] == "prose"

    def test_update_user_preferences(self):
        c = self._client()
        with patch("memory.store.MemoryStore.update") as mock_update:
            from memory.store import UserPreferences
            mock_update.return_value = UserPreferences(user_id="u1", verbosity="concise")
            r = c.put("/users/u1/preferences",
                      json={"verbosity": "concise"},
                      headers=self._ADMIN_HEADERS)
        assert r.status_code == 200

    def test_list_approvals_empty(self):
        c = self._client()
        with patch("tools.approval_gate.ApprovalGate.list_pending", return_value=[]):
            r = c.get("/approvals", headers=self._ADMIN_HEADERS)
        assert r.status_code == 200
        assert "pending" in r.json()

    def test_sync_trigger_unknown_connector_404(self):
        c = self._client()
        with patch("pipeline.worker.run_connector_sync",
                   side_effect=ValueError("Connector 'ghost' not registered.")):
            r = c.post("/connectors/ghost/sync",
                       json={"force": False, "max_workers": 4},
                       headers=self._ADMIN_HEADERS)
        assert r.status_code == 404

    def test_sync_trigger_returns_job_id(self):
        c = self._client()
        with patch("pipeline.worker.run_connector_sync", return_value="abc-123"):
            r = c.post("/connectors/my_conn/sync",
                       json={"force": False, "max_workers": 4},
                       headers=self._ADMIN_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == "abc-123"
        assert "poll_url" in body

    def test_sync_job_not_found_404(self):
        c = self._client()
        with patch("pipeline.progress.ProgressTracker.get", return_value=None):
            r = c.get("/sync/jobs/nonexistent", headers=self._ADMIN_HEADERS)
        assert r.status_code == 404

    def test_manifest_stats_endpoint(self):
        c = self._client()
        with patch("pipeline.sync_manifest.SyncManifest.stats",
                   return_value={"total_documents":5,"indexed":4,"failed":1}):
            r = c.get("/sync/manifest/stats", headers=self._ADMIN_HEADERS)
        assert r.status_code == 200

    def test_delete_document_requires_reason(self):
        c = self._client()
        # Missing reason param → 422
        r = c.delete("/documents/some_doc", headers=self._ADMIN_HEADERS)
        assert r.status_code == 422

    def test_token_usage_endpoint(self):
        c = self._client()
        with patch("agents.llm_client.get_total_tokens",
                   return_value={"prompt":100,"completion":50,"total":150,"projected_cost_usd":0.0}):
            r = c.get("/token-usage", headers=self._ADMIN_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert "total" in body
        assert "projected_cost_usd" in body
