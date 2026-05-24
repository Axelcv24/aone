"""Entry point for the Aone CLI."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="aone",
    help="Conversational operations agent. Processes Gmail messages and answers business questions in natural language.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def sync(limit: int = typer.Option(500, help="Number of messages to sync.")) -> None:
    """Sync the last N messages from Gmail into the local cache."""
    raise NotImplementedError("AONE-501 — pending in Sprint 5")


@app.command()
def ask(question: str = typer.Argument(..., help="Question for the agent.")) -> None:
    """Ask the agent a question about your emails."""
    raise NotImplementedError("AONE-502 — pending in Sprint 5")


@app.command()
def stats() -> None:
    """Show local cache statistics."""
    raise NotImplementedError("AONE-503 — pending in Sprint 5")


@app.command()
def evals() -> None:
    """Run the RAGAS evaluation suite."""
    raise NotImplementedError("AONE-504 — pending in Sprint 5")


if __name__ == "__main__":
    app()
