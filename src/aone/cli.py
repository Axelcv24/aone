"""Entry point del CLI de Aone."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="aone",
    help="Agente conversacional de operaciones. Procesa correos de Gmail y responde preguntas de negocio en lenguaje natural.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def sync(limit: int = typer.Option(500, help="Número de correos a sincronizar.")) -> None:
    """Sincroniza los últimos N correos desde Gmail al cache local."""
    raise NotImplementedError("AONE-501 — pendiente en Sprint 5")


@app.command()
def ask(question: str = typer.Argument(..., help="Pregunta para el agente.")) -> None:
    """Pregunta al agente sobre tus correos."""
    raise NotImplementedError("AONE-502 — pendiente en Sprint 5")


@app.command()
def stats() -> None:
    """Muestra estadísticas del cache local."""
    raise NotImplementedError("AONE-503 — pendiente en Sprint 5")


@app.command()
def evals() -> None:
    """Corre la suite de evaluaciones RAGAS."""
    raise NotImplementedError("AONE-504 — pendiente en Sprint 5")


if __name__ == "__main__":
    app()
