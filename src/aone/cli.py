"""Entry point for the Aone CLI."""

from __future__ import annotations

import os

# Silence the HuggingFace "Loading weights" tqdm bar and other library
# noise. The CLI owns stdout — we don't want sentence-transformers
# bleeding progress bars into the user's terminal during `aone ask`.
# Must be set before any HF / transformers import; the rest of the
# imports below are first-touch for HF, so this is the right place.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# LiteLLM emits warnings at import time for AWS Bedrock / SageMaker
# preload failures that we genuinely don't care about (no botocore
# installed because Aone doesn't use AWS). Suppress at module level.
os.environ.setdefault("LITELLM_LOG", "ERROR")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from aone.agent.graph import ask as agent_ask
from aone.agent.graph import build_agent
from aone.config import ConfigError, load_config
from aone.gmail.auth import GmailAuthError, get_service
from aone.llm.client import LLMClient
from aone.llm.embeddings import get_embedder
from aone.storage.cache import DEFAULT_CACHE_PATH, EmailCache
from aone.storage.vector import (
    DEFAULT_INDEX_PATH,
    VectorIndex,
    VectorIndexError,
)
from aone.sync import perform_sync

app = typer.Typer(
    name="aone",
    help="Conversational operations agent. Processes Gmail messages and answers business questions in natural language.",
    no_args_is_help=True,
    add_completion=False,
)

_console = Console()


@app.command()
def sync(
    limit: int = typer.Option(
        500, "--limit", "-n", help="Maximum number of messages to sync."
    ),
    query: str | None = typer.Option(
        None,
        "--query",
        "-q",
        help='Optional Gmail search query (web UI syntax, e.g. "is:unread").',
    ),
) -> None:
    """Sync the last N messages from Gmail into the local cache."""
    try:
        config = load_config()
    except ConfigError as exc:
        _console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _console.print("[bold]Connecting to Gmail[/bold] (browser opens on first run)…")
    try:
        service = get_service()
    except GmailAuthError as exc:
        _console.print(f"[red]Gmail authentication failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    cache = EmailCache.load_or_create(DEFAULT_CACHE_PATH)
    _console.print(
        f"  Cache: {len(cache)} messages already at "
        f"[dim]{DEFAULT_CACHE_PATH}[/dim]"
    )

    embedder = get_embedder(config.embedding_provider, config.embedding_model)
    index = _load_or_rebuild_index(embedder, cache)

    _console.print(
        f"  Listing up to {limit} message(s) from Gmail"
        + (f' matching [dim]"{query}"[/dim]' if query else "")
        + "…"
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=_console,
    ) as progress:
        task = progress.add_task("Fetching", total=None)

        def _on_fetch(email: object) -> None:
            progress.advance(task)

        def _on_error(message_id: str, exc: Exception) -> None:
            _console.print(
                f"  [yellow]skip[/yellow] {message_id}: {exc}"
            )

        result = perform_sync(
            service=service,
            cache=cache,
            index=index,
            limit=limit,
            query=query,
            on_fetch=_on_fetch,
            on_error=_on_error,
        )
        progress.update(task, total=result.fetched, completed=result.fetched)

    _console.print(
        f"\n  Listed: [bold]{result.listed}[/bold] "
        f"· already cached: {result.already_cached} "
        f"· fetched: [green]{result.fetched}[/green]"
        + (f" · failed: [red]{result.failed}[/red]" if result.failed else "")
    )

    if result.fetched > 0:
        cache.save()
        index.save(DEFAULT_INDEX_PATH)
        _console.print(
            f"  Cache: {result.cache_size} at "
            f"[dim]{DEFAULT_CACHE_PATH}[/dim]"
        )
        _console.print(
            f"  Index: {result.index_size} vectors at "
            f"[dim]{DEFAULT_INDEX_PATH}[/dim]"
        )

    _console.print("[bold green]Done.[/bold green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question for the agent."),
    show_metadata: bool = typer.Option(
        True,
        "--metadata/--no-metadata",
        help="Show intent, tools used, citations, and token cost after the answer.",
    ),
) -> None:
    """Ask the agent a question about your emails."""
    try:
        config = load_config()
    except ConfigError as exc:
        _console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not DEFAULT_CACHE_PATH.exists():
        _console.print(
            "[yellow]No cache yet.[/yellow] Run [bold]`aone sync`[/bold] first "
            "to load messages from Gmail."
        )
        raise typer.Exit(code=1)

    cache = EmailCache.load(DEFAULT_CACHE_PATH)
    if len(cache) == 0:
        _console.print(
            "[yellow]Cache is empty.[/yellow] Run [bold]`aone sync`[/bold] to load "
            "messages from Gmail."
        )
        raise typer.Exit(code=1)

    embedder = get_embedder(config.embedding_provider, config.embedding_model)
    try:
        index = VectorIndex.load(embedder, DEFAULT_INDEX_PATH)
    except (FileNotFoundError, VectorIndexError) as exc:
        _console.print(
            f"[red]Index unavailable:[/red] {exc}\n"
            "Run [bold]`aone sync`[/bold] to rebuild it."
        )
        raise typer.Exit(code=1) from exc

    llm = LLMClient(config=config)
    agent = build_agent(cache, index, llm)

    with _console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots"):
        response = agent_ask(agent, question)

    _console.print()
    _console.print(
        Panel(
            response.text,
            title=f"[bold]Q:[/bold] {question}",
            title_align="left",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if show_metadata:
        tools = ", ".join(response.tools_used) or "(none)"
        citations = (
            ", ".join(response.citations[:5]) + ("…" if len(response.citations) > 5 else "")
            if response.citations
            else "(none)"
        )
        model = response.model or "(no LLM call)"
        _console.print(
            f"  [dim]intent:[/dim]    {response.intent.value}\n"
            f"  [dim]tools:[/dim]     {tools}\n"
            f"  [dim]citations:[/dim] {citations}\n"
            f"  [dim]model:[/dim]     {model} "
            f"[dim]({response.total_tokens} tokens)[/dim]"
        )


@app.command()
def stats(
    top: int = typer.Option(5, "--top", help="How many top senders to list."),
) -> None:
    """Show local cache statistics."""
    if not DEFAULT_CACHE_PATH.exists():
        _console.print(
            "[yellow]No cache yet.[/yellow] Run [bold]`aone sync`[/bold] first."
        )
        raise typer.Exit(code=1)

    cache = EmailCache.load(DEFAULT_CACHE_PATH)
    info = cache.stats(top_n=top)

    _console.print(
        f"[bold]Aone cache[/bold] [dim]({DEFAULT_CACHE_PATH})[/dim]"
    )
    _console.print(f"  Messages:  [bold]{info.email_count}[/bold]")

    if info.disk_size_bytes is not None:
        _console.print(f"  Disk size: {_format_bytes(info.disk_size_bytes)}")

    if info.earliest_internal_date and info.latest_internal_date:
        _console.print(
            f"  Date range: {_format_ms(info.earliest_internal_date)} "
            f"→ {_format_ms(info.latest_internal_date)}"
        )

    if info.top_senders:
        _console.print(f"\n[bold]Top {len(info.top_senders)} sender(s)[/bold]")
        max_count_width = len(str(info.top_senders[0][1]))
        for address, count in info.top_senders:
            _console.print(f"  [cyan]{count:>{max_count_width}}[/cyan]  {address}")
    else:
        _console.print("\n[dim]No senders to rank (cache empty).[/dim]")


@app.command()
def evals() -> None:
    """Run the RAGAS evaluation suite."""
    raise NotImplementedError("AONE-504 — pending in Sprint 5")


def _format_bytes(num: int) -> str:
    """Human-friendly file size: 5,265 B → '5.1 KB'."""
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}" if unit != "B" else f"{num} {unit}"
        num = num // 1024 if unit == "B" else num / 1024  # type: ignore[assignment]
    return f"{num:.1f} TB"


def _format_ms(ms: int) -> str:
    """Format a ms-epoch timestamp as YYYY-MM-DD."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _load_or_rebuild_index(embedder: object, cache: EmailCache) -> VectorIndex:
    """Load the on-disk index. If it's missing or incompatible (e.g. the
    user switched embedding provider), rebuild it from the cache so the
    sync still runs cleanly."""
    try:
        index = VectorIndex.load(embedder, DEFAULT_INDEX_PATH)  # type: ignore[arg-type]
        _console.print(
            f"  Index: {len(index)} vectors at "
            f"[dim]{DEFAULT_INDEX_PATH}[/dim]"
        )
        return index
    except FileNotFoundError:
        _console.print(
            f"  Index: starting fresh at [dim]{DEFAULT_INDEX_PATH}[/dim]"
        )
        return VectorIndex(embedder)  # type: ignore[arg-type]
    except VectorIndexError as exc:
        _console.print(
            f"  [yellow]Existing index incompatible ({exc}); "
            f"re-embedding the {len(cache)} cached message(s)[/yellow]"
        )
        index = VectorIndex(embedder)  # type: ignore[arg-type]
        if len(cache) > 0:
            index.add_many(list(cache))
        return index


if __name__ == "__main__":
    app()
