"""
tools/registry.py
-----------------
Formal tool registry separating READ and WRITE tools.
Every write tool requires deterministic application-code validation
before execution — the model can only SUGGEST, not execute directly.

Tool design principles (from production agentic AI):
  1. Clear name, description, input schema, output schema
  2. READ tools: safe, no approval needed
  3. WRITE tools: require role check + deterministic validation
  4. Human approval gate for HIGH-RISK writes
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional
import structlog

logger = structlog.get_logger()

class ToolType(str, Enum):
    READ       = "read"        # safe, no side effects
    WRITE      = "write"       # modifies state, requires validation
    HIGH_RISK  = "high_risk"   # requires human approval before execution

@dataclass
class ToolSchema:
    name:        str
    description: str
    tool_type:   ToolType
    input_fields:  list[dict]   # [{"name": "query", "type": "str", "required": True}]
    output_fields: list[dict]
    required_role: str = "reader"  # minimum role to call this tool

@dataclass
class ToolResult:
    success:     bool
    data:        Any = None
    error:       Optional[str] = None
    tool_name:   str = ""
    latency_ms:  int = 0
    requires_approval: bool = False   # True for HIGH_RISK tools

class ToolRegistry:
    """
    Central registry for all tools the agent pipeline can call.
    Enforces read/write separation and role-based access.
    """
    def __init__(self):
        self._tools:     dict[str, ToolSchema]  = {}
        self._handlers:  dict[str, Callable]    = {}
        self._call_count:  dict[str, int]  = {}
        self._error_count: dict[str, int]  = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register all built-in ECIP tools with their schemas."""
        self.register(
            ToolSchema(
                name="search_knowledge_base",
                description="Search the indexed enterprise knowledge base with a natural language query.",
                tool_type=ToolType.READ,
                input_fields=[
                    {"name": "query",    "type": "str",           "required": True},
                    {"name": "top_k",    "type": "int",           "required": False},
                    {"name": "filter_doc","type": "Optional[str]","required": False},
                ],
                output_fields=[
                    {"name": "chunks",   "type": "list[dict]"},
                    {"name": "count",    "type": "int"},
                ],
                required_role="reader",
            ),
            handler=self._handle_search,
        )
        self.register(
            ToolSchema(
                name="get_document_metadata",
                description="Get metadata about a specific indexed document.",
                tool_type=ToolType.READ,
                input_fields=[{"name": "doc_title", "type": "str", "required": True}],
                output_fields=[{"name": "metadata", "type": "dict"}],
                required_role="reader",
            ),
            handler=self._handle_get_metadata,
        )
        self.register(
            ToolSchema(
                name="ingest_document",
                description="Index a new document into the knowledge base.",
                tool_type=ToolType.WRITE,
                input_fields=[
                    {"name": "file_path", "type": "str", "required": True},
                    {"name": "chunk_size","type": "int", "required": False},
                ],
                output_fields=[{"name": "chunks_indexed", "type": "int"}],
                required_role="writer",
            ),
            handler=self._handle_ingest,
        )
        self.register(
            ToolSchema(
                name="delete_document",
                description="Remove a document and all its chunks from the knowledge base.",
                tool_type=ToolType.HIGH_RISK,
                input_fields=[
                    {"name": "doc_title", "type": "str",  "required": True},
                    {"name": "reason",    "type": "str",  "required": True},
                    {"name": "confirmed", "type": "bool", "required": True},
                ],
                output_fields=[{"name": "deleted_chunks", "type": "int"}],
                required_role="admin",
            ),
            handler=self._handle_delete,
        )

    def register(self, schema: ToolSchema, handler: Callable):
        self._tools[schema.name]    = schema
        self._handlers[schema.name] = handler
        self._call_count[schema.name]  = 0
        self._error_count[schema.name] = 0

    def call(self, tool_name: str, params: dict, caller_role: str = "reader") -> ToolResult:
        """
        Execute a tool with deterministic validation.
        Write tools validate params in application code before execution.
        High-risk tools require confirmed=True in params.
        """
        if tool_name not in self._tools:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}", tool_name=tool_name)

        schema = self._tools[tool_name]

        # Role check — deterministic, never delegated to LLM
        from auth.middleware import _ROLE_WEIGHTS
        if _ROLE_WEIGHTS.get(caller_role, 0) < _ROLE_WEIGHTS.get(schema.required_role, 99):
            return ToolResult(
                success=False,
                error=f"Role '{caller_role}' cannot call '{tool_name}'. Required: '{schema.required_role}'.",
                tool_name=tool_name,
            )

        # HIGH_RISK: require explicit confirmation
        if schema.tool_type == ToolType.HIGH_RISK:
            if not params.get("confirmed", False):
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    requires_approval=True,
                    error="This is a high-risk action. Set confirmed=True after human approval.",
                )

        # Required field validation
        for f in schema.input_fields:
            if f.get("required") and f["name"] not in params:
                return ToolResult(
                    success=False,
                    error=f"Missing required field: {f['name']}",
                    tool_name=tool_name,
                )

        t0 = time.perf_counter()
        try:
            self._call_count[tool_name] += 1
            result = self._handlers[tool_name](params)
            latency = int((time.perf_counter() - t0) * 1000)
            logger.info("tool_call_success", tool=tool_name, latency_ms=latency)
            return ToolResult(success=True, data=result, tool_name=tool_name, latency_ms=latency)
        except Exception as exc:
            self._error_count[tool_name] += 1
            logger.error("tool_call_error", tool=tool_name, error=str(exc))
            return ToolResult(success=False, error=str(exc), tool_name=tool_name)

    def get_metrics(self) -> dict:
        return {
            name: {
                "calls": self._call_count[name],
                "errors": self._error_count[name],
                "success_rate": (
                    round(1 - self._error_count[name] / self._call_count[name], 3)
                    if self._call_count[name] > 0 else 1.0
                ),
            }
            for name in self._tools
        }

    def list_tools(self) -> list[dict]:
        return [
            {"name": s.name, "type": s.tool_type.value,
             "description": s.description, "required_role": s.required_role}
            for s in self._tools.values()
        ]

    # ── Handlers ─────────────────────────────────────────────────────────

    def _handle_search(self, params: dict) -> dict:
        from retrieval.retriever import HybridRetriever
        chunks = HybridRetriever().retrieve(
            params["query"],
            top_k_final=params.get("top_k", 5),
            filter_doc=params.get("filter_doc"),
        )
        return {"chunks": [c.to_dict(snippet_chars=1000) for c in chunks], "count": len(chunks)}

    def _handle_get_metadata(self, params: dict) -> dict:
        from ingestion.indexer import DocumentIndexer
        idx = DocumentIndexer()
        results = idx.collection.get(
            where={"doc_title": params["doc_title"]},
            include=["metadatas"],
        )
        metas = results.get("metadatas", [])
        return {"metadata": metas[0] if metas else {}, "chunk_count": len(metas)}

    def _handle_ingest(self, params: dict) -> dict:
        from ingestion.indexer import ingest_file
        return ingest_file(params["file_path"], params.get("chunk_size", 512))

    def _handle_delete(self, params: dict) -> dict:
        from ingestion.indexer import DocumentIndexer
        idx = DocumentIndexer()
        idx.delete_document(params["doc_title"])
        return {"deleted": True, "reason": params.get("reason", "")}


_registry: Optional[ToolRegistry] = None

def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
