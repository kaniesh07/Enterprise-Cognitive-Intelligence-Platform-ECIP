"""
ingestion/chunker.py
--------------------
Converts ParsedDocument pages into overlapping, semantically-aware
text chunks. Each chunk carries full provenance metadata for citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

import structlog

from ingestion.parser import ParsedDocument, ParsedPage

logger = structlog.get_logger()


@dataclass
class DocumentChunk:
    chunk_id: str           # "{doc_title}__p{page}__c{idx}"
    text: str
    source_path: str
    doc_title: str
    page_number: int
    chunk_index: int        # within the document
    char_start: int
    char_end: int
    layout_type: str        # "text" | "table" | "figure" | "mixed"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_path": self.source_path,
            "doc_title": self.doc_title,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "layout_type": self.layout_type,
            **self.metadata,
        }


# ── Token counting (tiktoken or char-based fallback) ─────────────────────

_ENCODER = None
_ENCODER_TRIED = False


def _get_encoder():
    """Load tiktoken once. tiktoken downloads its vocab on first use, so on an
    offline machine it would retry (and stall) on every call — try only once."""
    global _ENCODER, _ENCODER_TRIED
    if not _ENCODER_TRIED:
        _ENCODER_TRIED = True
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
    return _ENCODER


def _count_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4) if text else 0  # rough char→token ratio


# ── Sentence splitter ─────────────────────────────────────────────────────

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


_MAX_SENT_CHARS = 1200   # longer "sentences" (lists, logs, tables-as-text) are split further


def _split_long(s: str) -> list[str]:
    if len(s) <= _MAX_SENT_CHARS:
        return [s]
    lines = [l.strip() for l in s.split("\n") if l.strip()]
    if len(lines) > 1:
        out: list[str] = []
        for l in lines:
            out.extend(_split_long(l))
        return out
    words, out, cur = s.split(), [], []
    size = 0
    for w in words:
        if size + len(w) + 1 > _MAX_SENT_CHARS and cur:
            out.append(" ".join(cur))
            cur, size = [], 0
        cur.append(w)
        size += len(w) + 1
    if cur:
        out.append(" ".join(cur))
    return out


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
HEADING_MARK = "\x00H\x00"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving paragraph boundaries.
    Markdown headings are prefixed with HEADING_MARK so the chunker can start a new
    chunk at section boundaries."""
    text = "\n".join((f"\n\n{HEADING_MARK}{l}\n\n" if _HEADING.match(l) else l)
                     for l in text.split("\n"))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        parts = _SENT_RE.split(para)
        for p in parts:
            if p.strip():
                sentences.extend(_split_long(p.strip()))
    return [s for s in sentences if s and s != HEADING_MARK]


# ── Table serialiser ──────────────────────────────────────────────────────

def _table_to_text(table: list[list[str]]) -> str:
    """Serialise a table (list of rows) to markdown for embedding."""
    if not table:
        return ""
    header = table[0]
    rows = table[1:]
    sep = " | ".join("---" for _ in header)
    lines = [" | ".join(header), sep]
    lines += [" | ".join(row) for row in rows]
    return "\n".join(lines)


# ── Core chunker ──────────────────────────────────────────────────────────

class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = 512,      # target tokens per chunk
        chunk_overlap: int = 64,    # overlap tokens between chunks
        min_chunk_size: int = 10,   # discard shorter chunks (tokens)
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Auto-scale min_chunk_size: never larger than 20% of chunk_size
        self.min_chunk_size = min(min_chunk_size, max(5, chunk_size // 5))

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        global_idx = 0

        for page in doc.pages:
            page_chunks = list(self._chunk_page(doc, page, global_idx))
            chunks.extend(page_chunks)
            global_idx += len(page_chunks)

        logger.info(
            "chunking_complete",
            doc=doc.title,
            pages=len(doc.pages),
            chunks=len(chunks),
        )
        return chunks

    def _chunk_page(
        self, doc: ParsedDocument, page: ParsedPage, start_idx: int
    ) -> Iterator[DocumentChunk]:
        idx = start_idx

        # 1. Emit table chunks verbatim (tables should not be split mid-row)
        for table in page.tables:
            table_text = _table_to_text(table)
            if table_text.strip():
                yield self._make_chunk(doc, page, table_text, idx, "table")
                idx += 1

        # 2. Emit figure captions as small context chunks
        for caption in page.image_captions:
            if caption.strip():
                yield self._make_chunk(doc, page, caption.strip(), idx, "figure")
                idx += 1

        # 3. Chunk main text using sliding window over sentences
        if page.text.strip():
            sentences = _split_sentences(page.text)
            current: list[str] = []
            current_tokens = 0

            for sent in sentences:
                is_heading = sent.startswith(HEADING_MARK)
                if is_heading:
                    sent = sent[len(HEADING_MARK):]
                sent_tokens = _count_tokens(sent)

                # New section: flush the current chunk (if it has some substance) without overlap
                if is_heading and current and current_tokens >= max(self.min_chunk_size, 40):
                    text = " ".join(current)
                    if _count_tokens(text) >= self.min_chunk_size:
                        yield self._make_chunk(doc, page, text, idx, page.layout_type)
                        idx += 1
                    current, current_tokens = [], 0

                if current_tokens + sent_tokens > self.chunk_size and current:
                    text = " ".join(current)
                    if _count_tokens(text) >= self.min_chunk_size:
                        yield self._make_chunk(doc, page, text, idx, page.layout_type)
                        idx += 1

                    # Overlap: keep last N tokens worth of sentences
                    overlap_sentences = []
                    overlap_tokens = 0
                    for s in reversed(current):
                        t = _count_tokens(s)
                        if overlap_tokens + t > self.chunk_overlap:
                            break
                        overlap_sentences.insert(0, s)
                        overlap_tokens += t
                    current = overlap_sentences

                    current_tokens = sum(_count_tokens(s) for s in current)

                current.append(sent)
                current_tokens += sent_tokens

            # Emit final chunk (always emit if any text remains)
            if current:
                text = " ".join(current)
                if text.strip():
                    yield self._make_chunk(doc, page, text.strip(), idx, page.layout_type)

    def _make_chunk(
        self,
        doc: ParsedDocument,
        page: ParsedPage,
        text: str,
        idx: int,
        layout_type: str,
    ) -> DocumentChunk:
        chunk_id = f"{doc.title}__p{page.page_number:03d}__c{idx:04d}"
        return DocumentChunk(
            chunk_id=chunk_id,
            text=text.strip(),
            source_path=doc.source_path,
            doc_title=doc.title,
            page_number=page.page_number,
            chunk_index=idx,
            char_start=0,    # populated by caller if needed
            char_end=len(text),
            layout_type=layout_type,
            metadata={
                "doc_type": doc.doc_type,
                "num_tokens": _count_tokens(text),
            },
        )


# ── Convenience function ──────────────────────────────────────────────────

_default_chunker = SemanticChunker()


def chunk_document(doc: ParsedDocument, **kwargs) -> list[DocumentChunk]:
    chunker = SemanticChunker(**kwargs) if kwargs else _default_chunker
    return chunker.chunk_document(doc)
