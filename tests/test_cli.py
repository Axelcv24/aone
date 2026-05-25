"""CLI wiring tests for ``aone ask`` and ``aone sync`` (AONE-501, AONE-502).

The unit tests for the underlying logic live in ``tests/test_sync.py``
and ``tests/agent/test_graph.py``. Here we just verify the Typer
plumbing — error paths, exit codes, message strings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from aone import cli as cli_module
from aone.agent.intents import Intent
from aone.agent.respond import AgentResponse
from aone.cli import app
from aone.gmail.types import Email
from aone.storage.cache import EmailCache


runner = CliRunner()


def _email(id_: str = "m") -> Email:
    return Email(
        id=id_,
        thread_id=f"t-{id_}",
        from_="alice@x.com",
        to=["axel@example.com"],
        subject="s",
        body_text="b",
        body_html="<p>b</p>",
        body_clean="b",
        snippet="b",
        internal_date=1_700_000_000_000,
        labels=["INBOX"],
    )


def _populated_cache(tmp_path: Path) -> Path:
    cache = EmailCache()
    cache.add(_email())
    path = tmp_path / "cache.pkl"
    cache.save(path)
    return path


def _redirect_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    """Redirect the CLI's default cache + index paths into a tmp dir."""
    cache_path = tmp_path / "cache.pkl"
    index_path = tmp_path / "index.faiss"
    monkeypatch.setattr(cli_module, "DEFAULT_CACHE_PATH", cache_path)
    monkeypatch.setattr(cli_module, "DEFAULT_INDEX_PATH", index_path)
    return cache_path, index_path


# ─── aone --help ─────────────────────────────────────────────────────


def test_help_lists_all_four_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("sync", "ask", "stats", "evals"):
        assert cmd in result.stdout


# ─── aone ask: empty-cache short-circuit ─────────────────────────────


def test_ask_without_cache_file_exits_with_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)

    # Make load_config succeed so we exercise the cache-existence check.
    monkeypatch.setattr(
        cli_module, "load_config", lambda: _fake_config()
    )

    result = runner.invoke(app, ["ask", "anything"])

    assert result.exit_code == 1
    assert "No cache yet" in result.stdout
    assert "aone sync" in result.stdout


def test_ask_with_empty_cache_exits_with_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path, _ = _redirect_paths(monkeypatch, tmp_path)
    EmailCache().save(cache_path)  # empty but file exists
    monkeypatch.setattr(cli_module, "load_config", lambda: _fake_config())

    result = runner.invoke(app, ["ask", "anything"])

    assert result.exit_code == 1
    assert "Cache is empty" in result.stdout
    assert "aone sync" in result.stdout


def test_ask_with_bad_config_exits_with_red_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aone.config import ConfigError

    def _raise() -> None:
        raise ConfigError("GROQ_API_KEY is missing")

    monkeypatch.setattr(cli_module, "load_config", _raise)

    result = runner.invoke(app, ["ask", "anything"])

    assert result.exit_code == 1
    assert "Configuration error" in result.stdout
    assert "GROQ_API_KEY" in result.stdout


# ─── aone ask: happy path with mocked agent ──────────────────────────


def test_ask_happy_path_prints_response_text_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path, _ = _redirect_paths(monkeypatch, tmp_path)
    _populated_cache(tmp_path)  # reuses cache_path

    monkeypatch.setattr(cli_module, "load_config", lambda: _fake_config())

    # Bypass embedder loading (don't actually pull sentence-transformers).
    monkeypatch.setattr(
        cli_module,
        "get_embedder",
        lambda provider, model: MagicMock(provider_name=provider, model_name=model, dims=4),
    )
    # Bypass index load — return a stub.
    monkeypatch.setattr(
        cli_module.VectorIndex,
        "load",
        classmethod(lambda cls, embedder, path: MagicMock(_ids=["m"])),
    )
    # Bypass LLM client construction.
    monkeypatch.setattr(cli_module, "LLMClient", lambda config: MagicMock())
    # Skip the real graph; have build_agent return a sentinel and patch
    # agent_ask to return a canned response.
    sentinel = object()
    monkeypatch.setattr(cli_module, "build_agent", lambda c, i, llm: sentinel)

    canned = AgentResponse(
        text="Acme owes you USD 3,450.00 across 3 invoices.",
        intent=Intent.AGGREGATE_AMOUNTS,
        tools_used=["search_emails", "aggregate_amounts"],
        citations=["msg-acme-1024", "msg-acme-1031", "msg-acme-1042"],
        model="groq/llama-3.3-70b-versatile",
        total_tokens=812,
    )
    monkeypatch.setattr(
        cli_module,
        "agent_ask",
        lambda agent, question: canned if agent is sentinel else None,
    )

    result = runner.invoke(app, ["ask", "¿cuánto me debe Acme?"])

    assert result.exit_code == 0
    # The answer panel content shows up
    assert "USD 3,450.00" in result.stdout
    # Metadata block (default --metadata)
    assert "aggregate_amounts" in result.stdout
    assert "msg-acme-1024" in result.stdout
    assert "812 tokens" in result.stdout
    assert "groq/llama-3.3-70b-versatile" in result.stdout


# ─── aone stats ──────────────────────────────────────────────────────


def test_stats_without_cache_exits_with_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 1
    assert "No cache yet" in result.stdout
    assert "aone sync" in result.stdout


def test_stats_shows_counts_and_top_senders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path, _ = _redirect_paths(monkeypatch, tmp_path)
    cache = EmailCache()
    # Three senders with distinct counts so the ranking is deterministic.
    cache.add(_email("a"))  # alice@x.com (1)
    second = _email("b")
    cache.add(
        Email(  # type: ignore[name-defined]
            id="c",
            thread_id="t-c",
            from_="bob@y.com",
            to=["axel@example.com"],
            subject="s",
            body_text="b",
            body_html="<p>b</p>",
            body_clean="b",
            snippet="b",
            internal_date=1_700_000_100_000,
            labels=["INBOX"],
        )
    )
    cache.add(
        Email(  # type: ignore[name-defined]
            id="d",
            thread_id="t-d",
            from_="bob@y.com",
            to=["axel@example.com"],
            subject="s",
            body_text="b",
            body_html="<p>b</p>",
            body_clean="b",
            snippet="b",
            internal_date=1_700_000_200_000,
            labels=["INBOX"],
        )
    )
    cache.add(second)
    cache.save(cache_path)

    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 0
    assert "Messages:" in result.stdout
    assert "4" in result.stdout  # 4 messages
    # bob@y.com should be the top sender (2 messages)
    assert "bob@y.com" in result.stdout
    assert "alice@x.com" in result.stdout


def test_stats_respects_top_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path, _ = _redirect_paths(monkeypatch, tmp_path)
    cache = EmailCache()
    # Create 4 different senders so we can verify --top=2 caps the list.
    for i in range(4):
        cache.add(
            Email(  # type: ignore[name-defined]
                id=f"m{i}",
                thread_id=f"t-{i}",
                from_=f"user{i}@x.com",
                to=["axel@example.com"],
                subject="s",
                body_text="b",
                body_html="<p>b</p>",
                body_clean="b",
                snippet="b",
                internal_date=1_700_000_000_000 + i,
                labels=["INBOX"],
            )
        )
    cache.save(cache_path)

    result = runner.invoke(app, ["stats", "--top", "2"])

    assert result.exit_code == 0
    # Two of the four addresses appear in the rendered ranking section;
    # we can't easily assert which two given the equal counts, but the
    # header should reflect the cap.
    assert "Top 2 sender(s)" in result.stdout


def test_ask_no_metadata_flag_hides_intent_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path, _ = _redirect_paths(monkeypatch, tmp_path)
    _populated_cache(tmp_path)

    monkeypatch.setattr(cli_module, "load_config", lambda: _fake_config())
    monkeypatch.setattr(
        cli_module,
        "get_embedder",
        lambda p, m: MagicMock(provider_name=p, model_name=m, dims=4),
    )
    monkeypatch.setattr(
        cli_module.VectorIndex,
        "load",
        classmethod(lambda cls, e, p: MagicMock()),
    )
    monkeypatch.setattr(cli_module, "LLMClient", lambda config: MagicMock())
    monkeypatch.setattr(cli_module, "build_agent", lambda c, i, llm: object())
    monkeypatch.setattr(
        cli_module,
        "agent_ask",
        lambda a, q: AgentResponse(
            text="Plain answer.",
            intent=Intent.GENERAL_QA,
            tools_used=[],
            citations=[],
            model="m",
            total_tokens=0,
        ),
    )

    result = runner.invoke(app, ["ask", "--no-metadata", "anything"])

    assert result.exit_code == 0
    assert "Plain answer." in result.stdout
    assert "intent:" not in result.stdout
    assert "tools:" not in result.stdout


# ─── Helpers ─────────────────────────────────────────────────────────


def _fake_config():
    """Minimal Config-shaped object the CLI uses (only the fields it touches)."""
    from aone.config import Config

    return Config(
        groq_api_key="k",
        model_generation="groq/llama-3.3-70b-versatile",
        model_classification="groq/llama-3.1-8b-instant",
        embedding_provider="local",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        sync_limit=500,
        langfuse_public_key=None,
        langfuse_secret_key=None,
        langfuse_host="https://cloud.langfuse.com",
        anthropic_api_key=None,
        openai_api_key=None,
        gemini_api_key=None,
    )
