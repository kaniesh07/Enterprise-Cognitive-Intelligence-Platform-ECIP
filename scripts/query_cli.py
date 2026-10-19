#!/usr/bin/env python3
"""
scripts/query_cli.py
--------------------
Interactive terminal client for ECIP.
Supports single queries, interactive mode, and eval mode.

Usage:
  python scripts/query_cli.py
  python scripts/query_cli.py --query "What is the return policy?"
  python scripts/query_cli.py --api http://localhost:8000
  python scripts/query_cli.py --eval ./evaluation/datasets/eval_set.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()
import os
API_BASE = os.getenv("ECIP_API", "http://localhost:8000")
# FIX: every endpoint except /health needs X-API-Key — the CLI never sent it (all calls 401'd).
HEADERS = {"X-API-Key": os.getenv("ECIP_API_KEY", "ecip-dev-key")}
MODE = os.getenv("ECIP_MODE", "auto")


# ── API helpers ───────────────────────────────────────────────────────────

def check_health(base: str) -> dict:
    try:
        r = httpx.get(f"{base}/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


def query_api(base: str, question: str, history: list[dict], top_k: int = 5) -> dict:
    r = httpx.post(
        f"{base}/query",
        json={"query": question, "conversation_history": history, "top_k": top_k, "mode": MODE},
        headers=HEADERS, timeout=300,
    )
    r.raise_for_status()
    return r.json()


def ingest_file(base: str, path: str) -> dict:
    with open(path, "rb") as f:
        r = httpx.post(
            f"{base}/ingest",
            files={"file": (Path(path).name, f)},
            headers=HEADERS, timeout=600,
        )
    r.raise_for_status()
    return r.json()


def list_docs(base: str) -> dict:
    r = httpx.get(f"{base}/documents", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Display helpers ───────────────────────────────────────────────────────

def print_health(h: dict):
    status_color = "green" if h.get("status") == "healthy" else "red"
    console.print(
        Panel(
            f"[{status_color}]Status:[/{status_color}] {h.get('status', '?')}\n"
            f"Ollama: {h.get('ollama','?')}  •  ChromaDB: {h.get('chromadb','?')}",
            title="[bold]ECIP Health[/bold]",
            border_style="dim",
        )
    )


def print_response(result: dict, question: str):
    console.print()

    # Main answer
    answer = result.get("final_answer", "No answer returned.")
    console.print(Panel(Markdown(answer), title="[bold cyan]Answer[/bold cyan]", border_style="cyan"))

    # Metadata row
    faith = result.get("faithfulness_score", 0) * 100
    conf = result.get("confidence", 0) * 100
    intent = result.get("intent", "?")
    total_ms = result.get("latency_ms", {}).get("total", "?")
    flags = result.get("hallucination_flags", [])

    faith_color = "green" if faith >= 70 else "yellow" if faith >= 50 else "red"
    meta = (
        f"[{faith_color}]Faithfulness {faith:.0f}%[/{faith_color}]  "
        f"[blue]Confidence {conf:.0f}%[/blue]  "
        f"[magenta]Intent: {intent}[/magenta]  "
        f"[dim]{total_ms}ms[/dim]"
    )
    console.print(meta)

    if flags:
        console.print(f"[yellow]⚠ Unverified claims:[/yellow] {' · '.join(flags[:3])}")

    # Citations
    citations = result.get("citations", [])
    if citations:
        console.print()
        console.print("[dim]Sources:[/dim]")
        for c in citations:
            console.print(f"  [dim]•[/dim] {c}")

    # Reasoning (collapsed)
    steps = result.get("reasoning_steps", [])
    if steps and "--verbose" in sys.argv:
        console.print()
        console.print("[dim]Reasoning:[/dim]")
        for i, s in enumerate(steps, 1):
            console.print(f"  [dim]{i}.[/dim] {s}")

    # Retrieved chunks
    chunks = result.get("retrieved_chunks", [])
    if chunks and "--sources" in sys.argv:
        console.print()
        t = Table(title="Retrieved Chunks", show_header=True, header_style="bold dim")
        t.add_column("Doc", style="cyan", no_wrap=True)
        t.add_column("Page", justify="right")
        t.add_column("Score", justify="right")
        t.add_column("Snippet")
        for ch in chunks:
            t.add_row(
                ch.get("doc_title", "?")[:30],
                str(ch.get("page_number", "?")),
                f"{ch.get('score', 0):.3f}",
                ch.get("snippet", "")[:60] + "…",
            )
        console.print(t)

    console.print()


def print_docs(data: dict):
    docs = data.get("documents", [])
    if not docs:
        console.print("[dim]No documents indexed.[/dim]")
        return
    t = Table(title=f"Indexed Documents ({data.get('total',0)})", show_header=True)
    t.add_column("Title", style="cyan")
    t.add_column("Type")
    t.add_column("Chunks", justify="right")
    for d in docs:
        t.add_row(d["doc_title"], d.get("doc_type", "?"), str(d.get("chunk_count", "?")))
    console.print(t)


# ── Evaluation mode ───────────────────────────────────────────────────────

def run_eval(base: str, dataset_path: str):
    console.print(f"[bold]Running evaluation on:[/bold] {dataset_path}")
    r = httpx.post(f"{base}/evaluate", params={"eval_dataset_path": dataset_path},
                   headers=HEADERS, timeout=1800)
    r.raise_for_status()
    results = r.json()

    t = Table(title="Evaluation Results", show_header=True)
    t.add_column("Metric")
    t.add_column("Score", justify="right")

    ragas = results.get("ragas_metrics", {})
    for k, v in ragas.items():
        t.add_row(f"RAGAS {k}", f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v))

    retrieval = results.get("retrieval_metrics", {})
    for k, v in retrieval.items():
        t.add_row(k, f"{float(v):.4f}")

    console.print(t)
    console.print(f"\n[dim]Samples: {results.get('num_samples')}  •  Time: {results.get('evaluation_time_s')}s[/dim]")


# ── Interactive REPL ──────────────────────────────────────────────────────

HELP_TEXT = """
[bold]ECIP Interactive Query CLI[/bold]

Commands:
  [cyan]/docs[/cyan]           List indexed documents
  [cyan]/ingest <path>[/cyan]  Upload and index a document
  [cyan]/eval[/cyan]           Run evaluation suite
  [cyan]/health[/cyan]         Check service health
  [cyan]/history[/cyan]        Show conversation history
  [cyan]/clear[/cyan]          Clear conversation history
  [cyan]/verbose[/cyan]        Toggle reasoning steps display
  [cyan]/sources[/cyan]        Toggle source chunks display
  [cyan]/help[/cyan]           Show this help
  [cyan]/quit[/cyan]           Exit

Any other input is sent as a query.
"""

def interactive_mode(base: str, top_k: int):
    console.print(Panel(HELP_TEXT.strip(), border_style="dim"))

    health = check_health(base)
    print_health(health)
    if health.get("status") != "healthy":
        console.print("[yellow]Warning: some services may be unavailable.[/yellow]")

    history: list[dict] = []
    verbose = False
    show_sources = False

    while True:
        try:
            console.print("[dim]───────────────────────────────────────────────[/dim]")
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            args = user_input.split()[1:]

            if cmd == "/quit":
                console.print("[dim]Goodbye.[/dim]")
                break
            elif cmd == "/help":
                console.print(Panel(HELP_TEXT.strip(), border_style="dim"))
            elif cmd == "/health":
                print_health(check_health(base))
            elif cmd == "/docs":
                print_docs(list_docs(base))
            elif cmd == "/ingest":
                if not args:
                    console.print("[red]Usage: /ingest <file_path>[/red]")
                else:
                    path = " ".join(args)
                    with console.status(f"Ingesting {path}…"):
                        result = ingest_file(base, path)
                    console.print(f"[green]✓[/green] {result['doc_title']} — {result['chunks_indexed']} chunks indexed in {result['duration_ms']}ms")
            elif cmd == "/eval":
                run_eval(base, "./evaluation/datasets/eval_set.json")
            elif cmd == "/history":
                if not history:
                    console.print("[dim]No conversation history.[/dim]")
                else:
                    for turn in history:
                        role_color = "cyan" if turn["role"] == "user" else "green"
                        console.print(f"[{role_color}]{turn['role'].upper()}:[/{role_color}] {turn['content'][:120]}…")
            elif cmd == "/clear":
                history = []
                console.print("[dim]Conversation cleared.[/dim]")
            elif cmd == "/verbose":
                verbose = not verbose
                sys.argv = [a for a in sys.argv if a != "--verbose"]
                if verbose:
                    sys.argv.append("--verbose")
                console.print(f"[dim]Verbose mode {'on' if verbose else 'off'}[/dim]")
            elif cmd == "/sources":
                show_sources = not show_sources
                sys.argv = [a for a in sys.argv if a != "--sources"]
                if show_sources:
                    sys.argv.append("--sources")
                console.print(f"[dim]Sources display {'on' if show_sources else 'off'}[/dim]")
            else:
                console.print(f"[red]Unknown command: {cmd}. Type /help for commands.[/red]")
            continue

        # Query
        with console.status("[dim]Thinking…[/dim]"):
            try:
                result = query_api(base, user_input, history, top_k=top_k)
            except httpx.HTTPStatusError as e:
                console.print(f"[red]API error {e.response.status_code}: {e.response.text[:200]}[/red]")
                continue
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue

        print_response(result, user_input)

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result.get("final_answer", "")})
        if len(history) > 20:
            history = history[-20:]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECIP Interactive Query CLI")
    parser.add_argument("--api", default=API_BASE, help="API base URL")
    parser.add_argument("--query", "-q", help="Single query (non-interactive)")
    parser.add_argument("--eval", help="Run evaluation on a dataset JSON file")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--api-key", default=None, help="X-API-Key (or set ECIP_API_KEY)")
    parser.add_argument("--mode", choices=["auto", "chat", "docs"], default=None)
    parser.add_argument("--verbose", action="store_true", help="Show reasoning steps")
    parser.add_argument("--sources", action="store_true", help="Show retrieved chunks")
    args = parser.parse_args()
    global MODE
    if args.api_key:
        HEADERS["X-API-Key"] = args.api_key
    if args.mode:
        MODE = args.mode

    if args.verbose:
        sys.argv.append("--verbose")
    if args.sources:
        sys.argv.append("--sources")

    if args.eval:
        run_eval(args.api, args.eval)
    elif args.query:
        with console.status("[dim]Querying…[/dim]"):
            result = query_api(args.api, args.query, [], top_k=args.top_k)
        print_response(result, args.query)
    else:
        interactive_mode(args.api, args.top_k)


if __name__ == "__main__":
    main()
