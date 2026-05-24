"""Tests for ``aone.config``."""

from __future__ import annotations

from pathlib import Path

import pytest

from aone.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_MODEL_CLASSIFICATION,
    DEFAULT_MODEL_GENERATION,
    DEFAULT_SYNC_LIMIT,
    ConfigError,
    load_config,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Clear relevant env vars so each test starts from a clean state."""
    for var in (
        "GROQ_API_KEY",
        "AONE_MODEL_GENERATION",
        "AONE_MODEL_CLASSIFICATION",
        "AONE_EMBEDDING_PROVIDER",
        "AONE_EMBEDDING_MODEL",
        "AONE_SYNC_LIMIT",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _empty_env_file(tmp_path: Path) -> Path:
    f = tmp_path / ".env"
    f.write_text("")
    return f


def test_load_config_fails_without_groq_key(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        load_config(env_file=_empty_env_file(tmp_path))


def test_load_config_uses_defaults_when_only_groq_set(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clean_env.setenv("GROQ_API_KEY", "fake-groq-key")
    cfg = load_config(env_file=_empty_env_file(tmp_path))

    assert cfg.groq_api_key == "fake-groq-key"
    assert cfg.model_generation == DEFAULT_MODEL_GENERATION
    assert cfg.model_classification == DEFAULT_MODEL_CLASSIFICATION
    assert cfg.embedding_provider == DEFAULT_EMBEDDING_PROVIDER
    assert cfg.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert cfg.sync_limit == DEFAULT_SYNC_LIMIT
    assert cfg.langfuse_enabled is False


def test_load_config_overrides_defaults_via_env(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clean_env.setenv("GROQ_API_KEY", "k")
    clean_env.setenv("AONE_MODEL_GENERATION", "anthropic/claude-haiku-4-5")
    clean_env.setenv("AONE_EMBEDDING_PROVIDER", "litellm")
    clean_env.setenv("AONE_SYNC_LIMIT", "100")

    cfg = load_config(env_file=_empty_env_file(tmp_path))

    assert cfg.model_generation == "anthropic/claude-haiku-4-5"
    assert cfg.embedding_provider == "litellm"
    assert cfg.sync_limit == 100


def test_langfuse_enabled_when_all_three_present(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clean_env.setenv("GROQ_API_KEY", "k")
    clean_env.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    clean_env.setenv("LANGFUSE_SECRET_KEY", "sk")
    clean_env.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    cfg = load_config(env_file=_empty_env_file(tmp_path))
    assert cfg.langfuse_enabled is True


def test_langfuse_disabled_when_partial(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("GROQ_API_KEY", "k")
    clean_env.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    # Missing LANGFUSE_SECRET_KEY

    cfg = load_config(env_file=_empty_env_file(tmp_path))
    assert cfg.langfuse_enabled is False


def test_load_config_invalid_int(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("GROQ_API_KEY", "k")
    clean_env.setenv("AONE_SYNC_LIMIT", "not-an-int")

    with pytest.raises(ConfigError, match="AONE_SYNC_LIMIT"):
        load_config(env_file=_empty_env_file(tmp_path))


def test_load_config_reads_dotenv_file(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=from-file\nAONE_SYNC_LIMIT=42\n")

    cfg = load_config(env_file=env_file)
    assert cfg.groq_api_key == "from-file"
    assert cfg.sync_limit == 42
