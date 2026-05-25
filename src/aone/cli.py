"""Entry point for the Aone CLI."""

from __future__ import annotations

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
def stats() -> None:
    """Show local cache statistics."""
    raise NotImplementedError("AONE-503 — pending in Sprint 5")


@app.command()
def evals() -> None:
    """Run the RAGAS evaluation suite."""
    raise NotImplementedError("AONE-504 — pending in Sprint 5")


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
