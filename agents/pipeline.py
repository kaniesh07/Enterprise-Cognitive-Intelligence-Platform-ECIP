"""
agents/pipeline.py — the chat engine.

One request = route → (retrieve) → ONE streamed LLM call → verify → assemble.

Routing (no LLM call, a few milliseconds):
  * small talk ("hi", "thanks")            → instant reply
  * mode="chat" or nothing indexed          → general assistant (ChatGPT-style)
  * mode="docs"                             → always answer from documents
  * mode="auto" (default)                   → retrieve; answer from documents only if
                                              the best chunks are actually relevant,
                                              otherwise answer as general chat.

The previous design made three sequential LLM calls per question (intent,
reasoning, verifier) and the streaming endpoint generated every answer twice.
Here the intent is classified heuristically, grounding is verified with a fast
lexical check (VERIFIER_MODE=llm restores an LLM verifier), and the answer is
streamed straight from the model.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, AsyncGenerator, Optional

import structlog

from agents import llm_client as llm
from agents.model_router import get_model
from config import settings
from retrieval.retriever import HybridRetriever, RetrievedChunk

logger = structlog.get_logger()

PROMPT_VERSION = "v3.0"

# ── Small talk ────────────────────────────────────────────────────────────

_SMALLTALK = re.compile(
    r"^\s*(hi+|hey+|hello+|hiya|yo|thanks|thank you|thx|ty|ok|okay|bye|goodbye|see you|"
    r"good\s*(morning|evening|afternoon|night)|how are you( doing)?|what'?s up|sup|"
    r"who are you|what can you do|got it|cool|great|nice|sounds good|perfect|awesome)"
    r"(\s+(there|man|bro|buddy|friend|claude|assistant|ecip))?\s*[!?.]*\s*$",
    re.I,
)
_SMALLTALK_REPLIES = {
    "hi": "Hi! 👋 How can I help you today? You can ask me anything, or upload documents and ask questions about them.",
    "hello": "Hello! What can I help you with today?",
    "hey": "Hey! What's on your mind?",
    "thanks": "You're welcome! Anything else I can help with?",
    "thank you": "You're welcome! Anything else I can help with?",
    "bye": "Goodbye! Come back any time.",
    "how are you": "I'm doing well, thanks for asking! What can I help you with?",
    "who are you": ("I'm ECIP Assistant — a private AI assistant. I can chat about almost anything, "
                    "and when you upload documents I answer questions about them with citations."),
    "what can you do": ("I can answer general questions, explain concepts, help with writing and code, "
                        "and — once you upload PDFs, Word files, text, CSVs or images — answer questions "
                        "grounded in those documents with page-level citations."),
}


def smalltalk_reply(query: str) -> Optional[str]:
    q = query.strip()
    if len(q) > 40 or not _SMALLTALK.match(q):
        return None
    key = re.sub(r"[!?.\s]+$", "", q.lower())
    key = re.sub(r"\s+(there|man|bro|buddy|friend|claude|assistant|ecip)$", "", key)
    if re.match(r"^h(i+|ey+|ello+)$|^hiya$|^yo$", key):
        key = "hi"
    if key.startswith("good"):
        return f"{key.capitalize()}! How can I help you today?"
    if key in ("thx", "ty"):
        key = "thanks"
    if key in ("goodbye", "see you"):
        key = "bye"
    if key.startswith("how are you"):
        key = "how are you"
    return _SMALLTALK_REPLIES.get(key, "Sure! What would you like to do next?")


# ── Heuristic intent (replaces an LLM call) ───────────────────────────────

_INTENT_RULES = [
    ("summarisation", re.compile(r"\b(summari[sz]e|summary|tl;?dr|overview|key points|gist)\b", re.I)),
    ("comparison", re.compile(r"\b(compare|comparison|difference|differ|vs\.?|versus|better than)\b", re.I)),
    ("procedural", re.compile(r"\b(how (do|can|to|should)|steps?|procedure|process for|guide|set ?up|install)\b", re.I)),
    ("analytical", re.compile(r"\b(why|analy[sz]e|impact|evaluate|implications?|reason)\b", re.I)),
    ("creative", re.compile(r"\b(write|draft|compose|poem|story|email|essay|rewrite|translate)\b", re.I)),
    ("coding", re.compile(r"\b(code|function|python|javascript|sql|bug|error|regex|script|api)\b", re.I)),
]


def classify_intent(query: str) -> str:
    for name, rx in _INTENT_RULES:
        if rx.search(query):
            return name
    return "factual"


# ── Relevance gate helpers ────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset((
    "a an the of to in on for and or is are was were be been being it its this that these those with as at by "
    "from what which who whom how why when where do does did can could should would will shall may might "
    "i you he she we they me my your our their about into than then there here please tell explain give "
    "show list describe define mean means meaning any some all much many more most also just like get got "
    "make know need want use using used have has had not no yes so if but out up down over under "
    # meta words models use when talking about the context itself
    "source sources document documents mention mentioned mentions provide provided provides state "
    "states stated according given based answer question information context page".split()))
_DOC_CUES = re.compile(r"\b(documents?|docs?|pdfs?|files?|upload(ed)?|attachments?|according to the|"
                       r"in the (text|paper|slides|contract|sheet|report|policy|handbook|manual))\b", re.I)
_FOLLOWUP = re.compile(r"^\s*(and|also|what about|how about|why|more|elaborate|explain (that|it|more)|"
                       r"continue|go on|tell me more)\b|\b(it|its|that|this|they|them|those|he|she|his|her)\b",
                       re.I)

_DEFAULT_DENSE = {"nomic": 0.60, "minilm": 0.45, "mxbai": 0.60, "bge": 0.62,
                  "text-embedding-3": 0.40, "hash": 0.30}


def _stem(w: str) -> str:
    """Tiny suffix stripper so 'expenses'/'expense', 'submitted'/'submit' match."""
    for suf in ("ations", "ation", "ments", "ment", "ingly", "ings", "ing", "edly", "ies", "ied",
                "es", "ed", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if len(w) > 3 and w[-1] == w[-2]:       # submitt → submit
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):      # expense → expens (matches "expenses")
        w = w[:-1]
    return w


def _content_tokens(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def dense_threshold() -> float:
    if settings.min_dense_score is not None:
        return settings.min_dense_score
    name = (settings.embed_model if settings.embed_provider != "onnx"
            else settings.onnx_model_dir).lower()
    if settings.embed_provider == "hash":
        return _DEFAULT_DENSE["hash"]
    for key, val in _DEFAULT_DENSE.items():
        if key in name:
            return val
    return 0.5


def lexical_overlap(query: str, chunks: list[RetrievedChunk]) -> float:
    # Bare numbers ("what is 12 x 12?") must not make a question look document-related.
    q = {t for t in _content_tokens(query) if not t.replace(".", "").isdigit()}
    if not q:
        return 0.0
    best = 0.0
    for c in chunks[:3]:
        ct = _content_tokens(c.text + " " + c.doc_title)
        best = max(best, len(q & ct) / len(q))
    return best


def _prune_chunks(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop clearly off-topic chunks: fewer prompt tokens = faster and less confused answers."""
    if len(chunks) <= 1:
        return chunks
    top = max((c.dense_score or 0.0) for c in chunks)
    q = _content_tokens(query)
    keep = []
    for i, c in enumerate(chunks):
        ov = len(q & _content_tokens(c.text + " " + c.doc_title)) / len(q) if q else 0.0
        if i == 0 or (c.dense_score or 0.0) >= 0.75 * top or ov >= settings.min_lexical_overlap:
            keep.append(c)
    return keep


def _fit_budget(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Cap total context size (prompt processing time grows with every token)."""
    out, used = [], 0
    per = settings.context_chars_per_chunk
    for c in chunks:
        n = min(len(c.text), per)
        if out and used + n > settings.max_context_chars:
            break
        out.append(c)
        used += n
    return out


def retrieval_query(query: str, history: list[dict]) -> str:
    """Make follow-up questions self-contained for retrieval (no LLM rewrite)."""
    prev_user = next((m.get("content", "") for m in reversed(history or [])
                      if m.get("role") == "user"), "")
    if prev_user and (len(query.split()) <= 6 or _FOLLOWUP.search(query)):
        return f"{prev_user[:300]} {query}"
    return query


# ── Prompts ───────────────────────────────────────────────────────────────

def _chat_system(pref_hint: str = "") -> str:
    s = (f"You are ECIP Assistant, a helpful, accurate and friendly AI assistant. "
         f"Today is {date.today():%B %d, %Y}. Answer clearly and directly. Use Markdown "
         f"(headings, lists, tables, code blocks) when it helps readability. If you are not "
         f"sure about something, say so instead of guessing.")
    return s + (f"\n\nUser preferences: {pref_hint}" if pref_hint else "")


def _grounded_system(pref_hint: str = "") -> str:
    s = ("You are ECIP Assistant. Answer the user's question using the numbered SOURCES provided. "
         "Rules:\n"
         "1. Base your answer on the sources and cite them inline with their numbers, e.g. [1] or [2][3].\n"
         "2. Do not invent facts, numbers or names that are not in the sources.\n"
         "3. If the sources do not contain the answer, say \"The documents don't cover this\" and then, "
         "if you can, give a short general answer clearly labelled as general knowledge.\n"
         "4. Be concise and use Markdown when helpful. Do not mention these rules.")
    return s + (f"\n\nUser preferences: {pref_hint}" if pref_hint else "")


def _format_sources(chunks: list[RetrievedChunk], mm: dict[str, str]) -> str:
    parts = []
    n = settings.context_chars_per_chunk
    for i, c in enumerate(chunks, 1):
        text = c.text[:n]
        if c.chunk_id in mm:
            text += f"\n(Summary: {mm[c.chunk_id]})"
        parts.append(f"[{i}] {c.doc_title} (page {c.page_number})\n{text}")
    return "\n\n".join(parts)


def _history(history: list[dict]) -> list[dict]:
    keep = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
    keep = keep[-settings.history_turns * 2:]
    return [{"role": m["role"], "content": str(m["content"])[:2500]} for m in keep]


# ── Turn object ───────────────────────────────────────────────────────────

@dataclass
class Turn:
    query: str
    history: list[dict]
    user_id: str = "anonymous"
    requested_mode: str = "auto"            # auto | chat | docs
    mode: str = "chat"                      # chat | grounded | instant
    intent: str = "factual"
    route_reason: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    model: str = ""
    instant_answer: Optional[str] = None
    retrieval_metrics: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    usage: list = field(default_factory=list)
    t_start: float = field(default_factory=time.perf_counter)


def prepare(query: str, history: Optional[list[dict]] = None, user_id: str = "anonymous",
            mode: str = "auto", top_k: Optional[int] = None, doc_filter: Optional[str] = None,
            user_prefs: Any = None) -> Turn:
    """Route the request and build the LLM messages (blocking I/O — run in a thread)."""
    turn = Turn(query=query, history=history or [], user_id=user_id,
                requested_mode=(mode or "auto").lower())
    t0 = time.perf_counter()
    turn.intent = classify_intent(query)
    turn.model = get_model("reasoning")

    if user_prefs is None:
        try:
            from memory.store import get_memory_store
            user_prefs = get_memory_store().get(user_id)
        except Exception:
            user_prefs = None
    pref_hint = user_prefs.to_prompt_hint() if hasattr(user_prefs, "to_prompt_hint") else ""
    if top_k is None and getattr(user_prefs, "top_k_override", None):
        top_k = user_prefs.top_k_override
    top_k = top_k or settings.top_k_final

    # 1 ─ small talk
    if settings.instant_smalltalk and turn.requested_mode != "docs":
        reply = smalltalk_reply(query)
        if reply:
            turn.mode, turn.intent, turn.instant_answer = "instant", "conversational", reply
            turn.route_reason = "smalltalk"
            turn.latency_ms["route"] = int((time.perf_counter() - t0) * 1000)
            return turn

    # 2 ─ retrieval (unless the user explicitly asked for plain chat)
    from ingestion.indexer import index_is_empty
    empty = index_is_empty()
    if turn.requested_mode != "chat" and not empty:
        tr = time.perf_counter()
        rq = retrieval_query(query, turn.history)
        try:
            chunks = HybridRetriever().retrieve(rq, top_k_final=top_k, filter_doc=doc_filter)
        except Exception as exc:
            logger.error("retrieval_failed", error=str(exc)[:200])
            chunks = []
            turn.retrieval_metrics["error"] = str(exc)[:200]
        turn.latency_ms["retrieval"] = int((time.perf_counter() - tr) * 1000)

        top_dense = max((c.dense_score or 0.0) for c in chunks) if chunks else 0.0
        overlap = lexical_overlap(query if len(query.split()) > 2 else rq, chunks)
        thr = dense_threshold()
        relevant = bool(chunks) and (top_dense >= thr or overlap >= settings.min_lexical_overlap
                                     or bool(_DOC_CUES.search(query)) or doc_filter is not None)
        turn.retrieval_metrics.update({
            "returned": len(chunks), "top_dense_score": round(top_dense, 4),
            "lexical_overlap": round(overlap, 3), "dense_threshold": thr,
            "retrieval_query": rq if rq != query else "",
        })
        try:
            from monitoring.metrics import get_registry
            if chunks:
                get_registry().retrieval_top_score.observe(max(0.0, min(1.0, top_dense)))
        except Exception:
            pass

        if turn.requested_mode == "docs" or relevant:
            if chunks:
                turn.mode, turn.chunks = "grounded", _fit_budget(_prune_chunks(query, chunks))
                turn.route_reason = "docs_mode" if turn.requested_mode == "docs" else "relevant_docs"
            else:
                turn.mode = "instant"
                turn.instant_answer = ("I couldn't find anything relevant in the indexed documents. "
                                       "Try rephrasing, or switch to **Chat** mode for a general answer.")
                turn.route_reason = "docs_mode_no_hits"
        else:
            turn.mode, turn.route_reason = "chat", "docs_not_relevant"
    elif turn.requested_mode == "docs" and empty:
        turn.mode = "instant"
        turn.instant_answer = ("No documents are indexed yet. Upload a file with the 📎 button, "
                               "or switch to **Chat** mode to ask general questions.")
        turn.route_reason = "docs_mode_empty_index"
    else:
        turn.mode, turn.route_reason = "chat", ("chat_mode" if turn.requested_mode == "chat"
                                                else "empty_index")

    # 3 ─ messages
    hist = _history(turn.history)
    if turn.mode == "grounded":
        mm = _multimodal_summaries(turn) if settings_multimodal() else {}
        sources = _format_sources(turn.chunks, mm)
        # Sources go in the system message (after the fixed rules) and the user turn is
        # just the question — small models otherwise tend to echo the sources back.
        turn.messages = [{"role": "system",
                          "content": f"{_grounded_system(pref_hint)}\n\nSOURCES:\n{sources}"},
                         *hist, {"role": "user", "content": query}]
    elif turn.mode == "chat":
        turn.messages = [{"role": "system", "content": _chat_system(pref_hint)}, *hist,
                         {"role": "user", "content": query}]
    turn.latency_ms["route"] = int((time.perf_counter() - t0) * 1000) - turn.latency_ms.get("retrieval", 0)
    return turn


def settings_multimodal() -> bool:
    import os
    return os.getenv("MULTIMODAL_SUMMARIES", "false").lower() == "true"


def _multimodal_summaries(turn: Turn) -> dict[str, str]:
    """Optional: summarise table/figure chunks (one extra LLM call each — off by default)."""
    out = {}
    for c in turn.chunks:
        if c.layout_type in ("table", "figure", "mixed"):
            try:
                out[c.chunk_id] = llm.chat(
                    [{"role": "user", "content": f"Summarise this table/figure in 2 sentences:\n{c.text[:800]}"}],
                    model=get_model("multimodal"), temperature=0.0, max_tokens=120, usage_sink=turn.usage)
            except Exception as exc:
                logger.warning("multimodal_failed", error=str(exc)[:80])
    return out


# ── Verification & assembly ───────────────────────────────────────────────

_THINK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CITE = re.compile(r"\[(\d{1,2})\]")


def clean_answer(text: str) -> str:
    text = _THINK.sub("", text or "")
    text = re.sub(r"^\s*<answer>|</answer>\s*$", "", text.strip(), flags=re.I)
    return text.strip()


def heuristic_grounding(answer: str, chunks: list[RetrievedChunk]) -> tuple[float, list[str]]:
    """Share of answer sentences whose content words are mostly found in the sources."""
    source_tokens: set[str] = set()
    for c in chunks:
        source_tokens |= _content_tokens(c.text + " " + c.doc_title)
    sentences = [s.strip() for s in _SENT_SPLIT.split(_CITE.sub("", answer)) if s.strip()]
    scored, unsupported = 0, []
    total = 0
    for s in sentences:
        toks = _content_tokens(s)
        if len(toks) < 3:            # headings, "Sure!", list markers …
            continue
        if re.search(r"documents? don'?t cover|general knowledge|not (mentioned|covered|found) in",
                     s, re.I):
            continue
        total += 1
        if len(toks & source_tokens) / len(toks) >= 0.45:
            scored += 1
        else:
            unsupported.append(s[:160])
    if total == 0:
        return 1.0, []
    return round(scored / total, 3), unsupported[:3]


def _llm_verify(answer: str, turn: Turn) -> tuple[float, list[str]]:
    context = "\n".join(f"- {c.text[:500]}" for c in turn.chunks)
    res = llm.chat_json([
        {"role": "system", "content": "Fact-check the ANSWER against the CONTEXT. Return ONLY JSON: "
                                      '{"faithfulness_score": 0.0-1.0, "unsupported_claims": ["..."]}'},
        {"role": "user", "content": f"ANSWER:\n{answer}\n\nCONTEXT:\n{context}"},
    ], model=get_model("verifier"), usage_sink=turn.usage)
    return float(res.get("faithfulness_score", 0.8)), list(res.get("unsupported_claims", []))[:3]


def finalize(turn: Turn, raw_answer: str) -> dict:
    """Verify grounding, build citations and the response payload."""
    tv = time.perf_counter()
    answer = clean_answer(raw_answer) if turn.mode != "instant" else (turn.instant_answer or "")
    faith, flags = 1.0, []
    citations: list[str] = []
    sources: list[dict] = []

    if turn.mode == "grounded":
        vm = settings.verifier_mode.lower()
        if vm == "llm":
            try:
                faith, flags = _llm_verify(answer, turn)
            except Exception as exc:
                logger.warning("llm_verifier_failed", error=str(exc)[:100])
                faith, flags = heuristic_grounding(answer, turn.chunks)
        elif vm != "off":
            faith, flags = heuristic_grounding(answer, turn.chunks)
        used = {int(n) for n in _CITE.findall(answer) if 0 < int(n) <= len(turn.chunks)}
        for i, c in enumerate(turn.chunks, 1):
            src = c.to_dict()
            src.update({"index": i, "cited": i in used})
            sources.append(src)
        cited = [s for s in sources if s["cited"]] or sources
        seen = set()
        for s in cited:
            label = f"[{s['index']}] {s['doc_title']}, p.{s['page_number']}"
            if label not in seen:
                seen.add(label)
                citations.append(label)
        # Low grounding is reported via faithfulness_score / hallucination_flags (the UI
        # shows a warning) rather than by editing the answer text after it was streamed.
    turn.latency_ms["verify"] = int((time.perf_counter() - tv) * 1000)
    turn.latency_ms["total"] = int((time.perf_counter() - turn.t_start) * 1000)

    top = max((c.dense_score or 0.0) for c in turn.chunks) if turn.chunks else 0.0
    confidence = 1.0 if turn.mode != "grounded" else round(0.7 * faith + 0.3 * max(0.0, min(1.0, top)), 3)

    return {
        "final_answer": answer,
        "mode": turn.mode,
        "route_reason": turn.route_reason,
        "intent": turn.intent,
        "citations": citations,
        "sources": sources,
        "confidence": confidence,
        "faithfulness_score": faith,
        "hallucination_flags": flags if turn.mode == "grounded" else [],
        "reasoning_steps": [],
        "latency_ms": dict(turn.latency_ms),
        "retrieval_metrics": turn.retrieval_metrics,
        "retrieved_chunks": [c.to_dict(snippet_chars=200) for c in turn.chunks],
        "token_usage": llm.summarize_usage(turn.usage),
        "model": turn.model if turn.mode != "instant" else "",
        "prompt_version": PROMPT_VERSION,
        "_llm_metrics_recorded": True,
    }


# ── Generation ────────────────────────────────────────────────────────────

def generate(turn: Turn) -> str:
    """Blocking, non-streaming generation."""
    if turn.mode == "instant":
        return turn.instant_answer or ""
    tg = time.perf_counter()
    temp = 0.2 if turn.mode == "grounded" else settings.temperature
    out = llm.chat(turn.messages, model=turn.model, temperature=temp, usage_sink=turn.usage)
    turn.latency_ms["llm"] = int((time.perf_counter() - tg) * 1000)
    return out


async def stream(turn: Turn) -> AsyncGenerator[str, None]:
    """Yield answer tokens as they are generated."""
    if turn.mode == "instant":
        turn.latency_ms["ttft"] = int((time.perf_counter() - turn.t_start) * 1000)
        yield turn.instant_answer or ""
        return
    tg = time.perf_counter()
    temp = 0.2 if turn.mode == "grounded" else settings.temperature
    first = True
    in_think = False
    async for tok in llm.chat_stream_async(turn.messages, model=turn.model, temperature=temp,
                                           usage_sink=turn.usage):
        # Hide <think>…</think> blocks of reasoning models from the live stream
        if "<think>" in tok:
            in_think = True
        if in_think:
            if "</think>" in tok:
                in_think = False
                tok = tok.split("</think>", 1)[1]
            else:
                continue
        if first and tok.strip():
            first = False
            now = time.perf_counter()
            turn.latency_ms["ttft"] = int((now - turn.t_start) * 1000)
            try:
                from monitoring.metrics import get_registry
                get_registry().observe_ttft(turn.mode, now - turn.t_start)
            except Exception:
                pass
        if tok:
            yield tok
    turn.latency_ms["llm"] = int((time.perf_counter() - tg) * 1000)


def run(query: str, history: Optional[list[dict]] = None, user_id: str = "anonymous",
        mode: str = "auto", top_k: Optional[int] = None, doc_filter: Optional[str] = None) -> dict:
    """Synchronous end-to-end helper (CLI, evaluation, tests)."""
    turn = prepare(query, history, user_id, mode, top_k, doc_filter)
    raw = generate(turn)
    return finalize(turn, raw)


async def run_async(query: str, history: Optional[list[dict]] = None, user_id: str = "anonymous",
                    mode: str = "auto", top_k: Optional[int] = None,
                    doc_filter: Optional[str] = None) -> dict:
    turn = await asyncio.to_thread(prepare, query, history, user_id, mode, top_k, doc_filter)
    parts = [t async for t in stream(turn)]
    return await asyncio.to_thread(finalize, turn, "".join(parts))
