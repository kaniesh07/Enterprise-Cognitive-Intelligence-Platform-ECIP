"""
ingestion/parser.py
--------------------
Robust multi-layer document parser. Every optional dependency is
imported inside its function so a missing package never crashes the app.

Parse chain (tries each in order, uses first that succeeds):
  PDF/DOCX/image → Docling (IBM, layout-aware)
  PDF            → PyPDF  + PyMuPDF+Tesseract OCR for scanned pages
  DOCX           → python-docx
  HTML/HTM       → BeautifulSoup4 (falls back to regex if bs4 absent)
  CSV            → stdlib csv
  Image          → Pillow + Tesseract
  Anything else  → plaintext (always succeeds)
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class ParsedPage:
    page_number:    int
    text:           str
    tables:         list[list[list[str]]] = field(default_factory=list)
    image_captions: list[str]             = field(default_factory=list)
    layout_type:    str                   = "text"  # text|table|figure|mixed


@dataclass
class ParsedDocument:
    source_path: str
    doc_type:    str
    title:       str
    pages:       list[ParsedPage]
    metadata:    dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


# ── Helpers ────────────────────────────────────────────────────────────────

def _single(path: Path, text: str, doc_type: str,
            parser: str, layout: str = "text") -> ParsedDocument:
    """Convenience: wrap a single block of text as a one-page document."""
    return ParsedDocument(
        source_path=str(path), doc_type=doc_type, title=path.stem,
        pages=[ParsedPage(page_number=1, text=text, layout_type=layout)],
        metadata={"parser": parser},
    )


# ── Layer 1: Docling ──────────────────────────────────────────────────────

def _parse_with_docling(path: Path) -> Optional[ParsedDocument]:
    """
    Use Docling's stable public API: DocumentConverter → export_to_markdown().
    Avoids iterating doc.pages whose structure changed between Docling versions.
    """
    try:
        from docling.document_converter import DocumentConverter

        result   = DocumentConverter().convert(str(path))
        markdown = result.document.export_to_markdown()

        if not markdown.strip():
            return None

        # Split on Docling page markers when present
        import re
        parts = re.split(r"<!--\s*page\s*\d+\s*-->", markdown, flags=re.I)
        pages = [
            ParsedPage(page_number=i, text=part.strip())
            for i, part in enumerate(parts, 1)
            if part.strip()
        ] or [ParsedPage(page_number=1, text=markdown.strip())]

        return ParsedDocument(
            source_path=str(path),
            doc_type=path.suffix.lstrip(".").lower(),
            title=path.stem,
            pages=pages,
            metadata={"parser": "docling", "num_pages": len(pages)},
        )
    except Exception as exc:
        logger.warning("docling_failed", path=str(path), error=str(exc)[:120])
        return None


# ── Layer 2: PyPDF (text-layer PDFs) ─────────────────────────────────────

def _parse_pdf_pypdf(path: Path) -> Optional[ParsedDocument]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages  = []
        for i, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if not text:
                text = _ocr_pdf_page(path, i - 1)   # scanned page → OCR
            pages.append(ParsedPage(page_number=i, text=text))

        if not any(p.text for p in pages):
            return None

        return ParsedDocument(
            source_path=str(path), doc_type="pdf", title=path.stem,
            pages=pages, metadata={"parser": "pypdf"},
        )
    except Exception as exc:
        logger.warning("pypdf_failed", path=str(path), error=str(exc)[:120])
        return None


def _ocr_pdf_page(pdf_path: Path, page_index: int) -> str:
    """Render one PDF page as image and OCR it. Returns '' if unavailable."""
    try:
        import fitz          # PyMuPDF — optional
        import pytesseract
        from PIL import Image

        doc = fitz.open(str(pdf_path))
        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)
    except ImportError:
        return ""
    except Exception as exc:
        logger.warning("ocr_page_failed", page=page_index, error=str(exc)[:80])
        return ""


# ── Layer 3: python-docx ──────────────────────────────────────────────────

def _parse_docx(path: Path) -> Optional[ParsedDocument]:
    try:
        from docx import Document   # package: python-docx, import name: docx

        doc    = Document(str(path))
        text   = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        tables = [
            [[cell.text for cell in row.cells] for row in tbl.rows]
            for tbl in doc.tables
        ]
        layout = "mixed" if tables else "text"
        return ParsedDocument(
            source_path=str(path), doc_type="docx", title=path.stem,
            pages=[ParsedPage(page_number=1, text=text,
                              tables=tables, layout_type=layout)],
            metadata={"parser": "python-docx"},
        )
    except ImportError:
        logger.warning("python_docx_missing", path=str(path))
        return None
    except Exception as exc:
        logger.warning("docx_failed", path=str(path), error=str(exc)[:120])
        return None


# ── Layer 4: HTML ─────────────────────────────────────────────────────────

def _parse_html(path: Path) -> Optional[ParsedDocument]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "head"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        except ImportError:
            import re
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s+", " ", text).strip()
        return _single(path, text, "html", "beautifulsoup") if text.strip() else None
    except Exception as exc:
        logger.warning("html_failed", path=str(path), error=str(exc)[:120])
        return None


# ── Layer 5: CSV ──────────────────────────────────────────────────────────

def _parse_csv(path: Path) -> Optional[ParsedDocument]:
    try:
        raw     = path.read_text(encoding="utf-8", errors="ignore")
        rows    = list(csv.reader(io.StringIO(raw)))
        if not rows:
            return None
        headers = rows[0]
        lines   = []
        for row in rows[1:]:
            pairs = " | ".join(f"{h}: {v}" for h, v in zip(headers, row) if v.strip())
            if pairs:
                lines.append(pairs)

        return ParsedDocument(
            source_path=str(path), doc_type="csv", title=path.stem,
            # Rows are emitted once as "header: value" lines (best for the LLM);
            # emitting the table as well produced duplicate chunks.
            pages=[ParsedPage(page_number=1, text="\n\n".join(lines), layout_type="table")],
            metadata={"parser": "csv", "rows": len(rows), "cols": len(headers)},
        )
    except Exception as exc:
        logger.warning("csv_failed", path=str(path), error=str(exc)[:120])
        return None


# ── Layer 6: Image OCR ────────────────────────────────────────────────────

def _parse_image(path: Path) -> Optional[ParsedDocument]:
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(path))
        return _single(path, text, "image", "pytesseract", "figure") if text.strip() else None
    except ImportError:
        logger.warning("pytesseract_missing", path=str(path))
        return None
    except Exception as exc:
        logger.warning("image_ocr_failed", path=str(path), error=str(exc)[:120])
        return None


# ── Layer 7: Plaintext — always works ────────────────────────────────────

def _parse_plaintext(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _single(path, text, path.suffix.lstrip(".").lower() or "text", "plaintext")


# ── Router ────────────────────────────────────────────────────────────────

def _parse_by_type(path: Path) -> ParsedDocument:
    """Try specialised parsers in order, always fall back to plaintext."""
    suffix = path.suffix.lower()
    parsers = {
        ".pdf":  [_parse_pdf_pypdf],
        ".docx": [_parse_docx],
        ".doc":  [_parse_docx],
        ".html": [_parse_html],
        ".htm":  [_parse_html],
        ".csv":  [_parse_csv],
        ".png":  [_parse_image],
        ".jpg":  [_parse_image],
        ".jpeg": [_parse_image],
        ".tiff": [_parse_image],
        ".bmp":  [_parse_image],
        ".webp": [_parse_image],
    }
    for fn in parsers.get(suffix, []):
        result = fn(path)
        if result:
            return result
    return _parse_plaintext(path)


# ── Public API ────────────────────────────────────────────────────────────

def parse_document(path: str | Path) -> ParsedDocument:
    """
    Parse any supported document. Never raises (returns plaintext on all failures).

    Supported: PDF DOCX TXT MD HTML CSV PNG JPG JPEG TIFF BMP WEBP
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    logger.info("parsing_document", path=str(path), size_kb=path.stat().st_size // 1024)

    # Docling (layout-aware, but slow and downloads large models on first use)
    # is opt-in via USE_DOCLING=true and only used for rich formats.
    doc = None
    if (os.getenv("USE_DOCLING", "false").lower() == "true"
            and path.suffix.lower() in {".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".tiff"}):
        doc = _parse_with_docling(path)
    # Fall back to type-specific parsers
    if doc is None:
        doc = _parse_by_type(path)

    logger.info("parsing_complete", title=doc.title, pages=len(doc.pages),
                parser=doc.metadata.get("parser"), chars=len(doc.full_text))
    return doc
