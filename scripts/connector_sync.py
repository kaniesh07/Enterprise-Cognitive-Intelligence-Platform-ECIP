#!/usr/bin/env python3
"""
scripts/connector_sync.py
--------------------------
CLI for managing source connectors and triggering sync jobs.

Usage:
  python scripts/connector_sync.py list-types
  python scripts/connector_sync.py register --name my_s3 --type s3 --config config.json
  python scripts/connector_sync.py list
  python scripts/connector_sync.py sync --name my_s3 [--force] [--workers 8]
  python scripts/connector_sync.py status --job-id <uuid>
  python scripts/connector_sync.py test --name my_s3
  python scripts/connector_sync.py manifest-stats
  python scripts/connector_sync.py retry-failed
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

console = Console()


def cmd_list_types(_args):
    """Show all available connector types."""
    from connectors.registry import ConnectorRegistry
    reg = ConnectorRegistry()
    t = Table(title="Supported Connector Types", show_header=True)
    t.add_column("Type", style="cyan")
    t.add_column("Description")
    t.add_column("Required Package")
    descriptions = {
        "local":      ("Local filesystem / NFS / SMB share",  "built-in"),
        "s3":         ("AWS S3 / MinIO / Azure Blob (S3-compatible)", "boto3"),
        "sharepoint": ("Microsoft SharePoint / OneDrive",     "msal requests"),
        "gdrive":     ("Google Drive",                         "google-api-python-client google-auth"),
        "confluence": ("Atlassian Confluence Cloud or Server", "requests"),
        "sql":        ("SQL databases (PostgreSQL/MySQL/SQLite/MSSQL)", "sqlalchemy"),
        "web":        ("Websites / web pages (crawl + scrape)", "requests beautifulsoup4"),
    }
    for t_type in reg.supported_types():
        desc, pkg = descriptions.get(t_type, ("", ""))
        t.add_row(t_type, desc, pkg)
    console.print(t)


def cmd_register(args):
    """Register a new connector from a JSON config file or inline JSON."""
    from connectors.registry import get_connector_registry
    config = {}
    if args.config:
        p = Path(args.config)
        if p.exists():
            config = json.loads(p.read_text())
        else:
            config = json.loads(args.config)  # treat as inline JSON
    if args.config_kv:
        for kv in args.config_kv:
            k, v = kv.split("=", 1)
            config[k] = v

    reg = get_connector_registry()
    reg.register(args.name, args.type, config)
    console.print(f"[green]✓[/green] Registered connector [cyan]{args.name}[/cyan] (type={args.type})")


def cmd_list(_args):
    """List all registered connectors."""
    from connectors.registry import get_connector_registry
    conns = get_connector_registry().list_all()
    if not conns:
        console.print("[dim]No connectors registered. Use 'register' to add one.[/dim]")
        return
    t = Table(title=f"Registered Connectors ({len(conns)})", show_header=True)
    t.add_column("Name", style="cyan"); t.add_column("Type"); t.add_column("Enabled")
    for c in conns:
        t.add_row(c["name"], c["type"], "✓" if c.get("enabled") else "✗")
    console.print(t)


def cmd_test(args):
    """Test connection to a registered connector."""
    from connectors.registry import get_connector_registry
    conn = get_connector_registry().get(args.name)
    if not conn:
        console.print(f"[red]Connector '{args.name}' not found.[/red]"); return
    with console.status(f"Testing connection to [cyan]{args.name}[/cyan]…"):
        ok = conn.test_connection()
    if ok:
        console.print(f"[green]✓[/green] Connection to [cyan]{args.name}[/cyan] is working.")
    else:
        console.print(f"[red]✗[/red] Connection to [cyan]{args.name}[/cyan] failed.")


def cmd_sync(args):
    """Trigger a sync job and watch progress live."""
    from pipeline.worker import run_connector_sync
    from pipeline.progress import get_progress_tracker

    with console.status(f"Starting sync for [cyan]{args.name}[/cyan]…"):
        try:
            job_id = run_connector_sync(
                args.name, force=args.force, max_workers=args.workers
            )
        except ValueError as e:
            console.print(f"[red]{e}[/red]"); return

    console.print(f"[green]Sync started[/green] — Job ID: [yellow]{job_id}[/yellow]")
    console.print("Watching progress (Ctrl+C to detach, sync continues in background)…\n")

    tracker = get_progress_tracker()
    try:
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                job = tracker.get(job_id)
                if not job:
                    break
                live.update(_render_progress(job))
                if job["status"] in ("completed","failed","cancelled"):
                    time.sleep(0.5)
                    break
                time.sleep(1)
    except KeyboardInterrupt:
        console.print(f"\n[dim]Detached. Poll with: python scripts/connector_sync.py status --job-id {job_id}[/dim]")

    final = tracker.get(job_id)
    if final:
        _print_final_summary(final)


def _render_progress(job: dict) -> Panel:
    pct   = job.get("progress_pct", 0)
    bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    eta   = job.get("eta_seconds")
    eta_s = f"{int(eta)}s" if eta else "calculating…"
    status_color = {"running":"cyan","completed":"green","failed":"red"}.get(job["status"],"white")
    content = (
        f"[{status_color}]Status:[/{status_color}] {job['status']}  "
        f"[dim]Job:[/dim] {job['job_id'][:8]}…\n"
        f"[{status_color}]{bar}[/{status_color}] {pct:.1f}%\n\n"
        f"Discovered: {job['total_discovered']:,}  "
        f"Indexed: [green]{job['indexed']:,}[/green]  "
        f"Skipped: [dim]{job['skipped']:,}[/dim]  "
        f"Failed: [red]{job['failed']:,}[/red]\n"
        f"Chunks: {job['total_chunks']:,}  "
        f"Throughput: {job['throughput_docs_per_min']:.0f} docs/min  "
        f"ETA: {eta_s}\n"
        f"Current: [dim]{job.get('current_doc','')[:60]}[/dim]"
    )
    return Panel(content, title=f"[bold]Syncing: {job['connector_name']}[/bold]", border_style=status_color)


def _print_final_summary(job: dict):
    color = "green" if job["status"] == "completed" else "red"
    console.print()
    console.print(Panel(
        f"[{color}]{job['status'].upper()}[/{color}]\n"
        f"Indexed:  [green]{job['indexed']:,}[/green] documents  "
        f"({job['total_chunks']:,} chunks)\n"
        f"Skipped:  {job['skipped']:,}  (unchanged)\n"
        f"Failed:   [red]{job['failed']:,}[/red]\n"
        f"Duration: {job['elapsed_s']:.1f}s  "
        f"Avg throughput: {job['throughput_docs_per_min']:.0f} docs/min",
        title="Sync Complete",
        border_style=color,
    ))
    if job.get("recent_errors"):
        console.print("[red]Recent errors:[/red]")
        for e in job["recent_errors"][:5]:
            console.print(f"  [red]✗[/red] {e['source_id']}: {e['error']}")


def cmd_status(args):
    """Poll a running sync job's progress."""
    from pipeline.progress import get_progress_tracker
    job = get_progress_tracker().get(args.job_id)
    if not job:
        console.print(f"[red]Job '{args.job_id}' not found.[/red]"); return
    console.print(_render_progress(job))


def cmd_manifest_stats(_args):
    """Show sync manifest statistics."""
    from pipeline.sync_manifest import get_manifest
    stats = get_manifest().stats()
    t = Table(title="Sync Manifest Statistics", show_header=True)
    t.add_column("Metric"); t.add_column("Value", justify="right", style="cyan")
    for k, v in stats.items():
        t.add_row(k.replace("_", " ").title(), f"{v:,}")
    console.print(t)


def cmd_retry_failed(_args):
    """Mark all failed documents for retry on next sync."""
    from pipeline.sync_manifest import get_manifest
    m = get_manifest()
    failed = m.get_failed()
    if not failed:
        console.print("[dim]No failed documents.[/dim]"); return
    m.reset_failed()
    console.print(f"[green]✓[/green] {len(failed)} failed document(s) marked for retry.")


def main():
    parser = argparse.ArgumentParser(
        description="ECIP Connector Sync CLI — index millions of documents from any source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-types", help="Show all supported connector types")
    sub.add_parser("list",       help="List registered connectors")
    sub.add_parser("manifest-stats", help="Show sync manifest statistics")
    sub.add_parser("retry-failed",   help="Mark failed docs for retry")

    r = sub.add_parser("register", help="Register a new connector")
    r.add_argument("--name",   required=True)
    r.add_argument("--type",   required=True, dest="type",
                   choices=["local","s3","sharepoint","gdrive","confluence","sql","web"])
    r.add_argument("--config", help="Path to JSON config file OR inline JSON string")
    r.add_argument("--set",    nargs="*", dest="config_kv", metavar="KEY=VALUE",
                   help="Additional config key=value pairs")

    te = sub.add_parser("test", help="Test a connector's connection")
    te.add_argument("--name", required=True)

    s = sub.add_parser("sync", help="Trigger a sync job")
    s.add_argument("--name",    required=True)
    s.add_argument("--force",   action="store_true", help="Re-index even unchanged docs")
    s.add_argument("--workers", type=int, default=8, help="Parallel workers (default 8)")

    st = sub.add_parser("status", help="Check a sync job's progress")
    st.add_argument("--job-id", required=True)

    args = parser.parse_args()
    cmds = {
        "list-types":     cmd_list_types,
        "register":       cmd_register,
        "list":           cmd_list,
        "test":           cmd_test,
        "sync":           cmd_sync,
        "status":         cmd_status,
        "manifest-stats": cmd_manifest_stats,
        "retry-failed":   cmd_retry_failed,
    }
    cmds[args.command](args)

if __name__ == "__main__":
    main()
