#!/usr/bin/env python3
"""
scripts/ingest_bulk.py
-----------------------
Bulk-ingest a directory of documents into the ECIP knowledge base.

Usage:
  python scripts/ingest_bulk.py --dir ./my_documents
  python scripts/ingest_bulk.py --dir ./docs --chunk-size 256 --chunk-overlap 32
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from ingestion.indexer import ingest_file

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff"}

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest documents into ECIP")
    parser.add_argument("--dir", required=True, help="Directory containing documents")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirs")
    args = parser.parse_args()

    doc_dir = Path(args.dir)
    if not doc_dir.exists():
        console.print(f"[red]Directory not found: {doc_dir}[/red]")
        sys.exit(1)

    pattern = "**/*" if args.recursive else "*"
    files = [f for f in doc_dir.glob(pattern) if f.is_file() and f.suffix.lower() in SUPPORTED]

    if not files:
        console.print(f"[yellow]No supported files found in {doc_dir}[/yellow]")
        sys.exit(0)

    console.print("\n[bold]ECIP Bulk Ingestion[/bold]")
    console.print(f"Found [cyan]{len(files)}[/cyan] files in [cyan]{doc_dir}[/cyan]\n")

    results = []
    errors = []

    for f in tqdm(files, desc="Indexing"):
        try:
            result = ingest_file(str(f), args.chunk_size, args.chunk_overlap)
            results.append(result)
        except Exception as exc:
            errors.append({"file": str(f), "error": str(exc)})

    # Summary table
    table = Table(title="Ingestion Results", show_header=True)
    table.add_column("Document", style="cyan", no_wrap=True)
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Indexed", justify="right", style="green")

    for r in results:
        table.add_row(
            Path(r["file"]).name,
            str(r["pages"]),
            str(r["chunks_created"]),
            str(r["chunks_indexed"]),
        )

    console.print(table)

    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/red]")
        for e in errors:
            console.print(f"  ✗ {e['file']}: {e['error']}")

    total_indexed = sum(r["chunks_indexed"] for r in results)
    console.print(f"\n✓ [green]{total_indexed} total chunks indexed[/green] across {len(results)} documents.")


if __name__ == "__main__":
    main()
