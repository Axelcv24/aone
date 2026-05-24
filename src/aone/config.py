"""Provider-agnostic configuration (ADR-005).

Models and providers come from environment variables — never hardcoded.
Changing ``AONE_MODEL_GENERATION`` or ``AONE_EMBEDDING_PROVIDER`` reconfigures
the whole app without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL_GENERATION = "groq/llama-3.3-70b-versatile"
DEFAULT_MODEL_CLASSIFICATION = "groq/llama-3.1-8b-instant"
DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"
DEFAULT_SYNC_LIMIT = 500


class ConfigError(RuntimeError):
    """Human-readable configuration error."""


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of the configuration loaded from the environment."""

    groq_api_key: str
    model_generation: str
    model_classification: str
    embedding_provider: str
    embedding_model: str
    sync_limit: int
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str
    anthropic_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None

    @property
    def langfuse_enabled(self) -> bool:
        """True if all three Langfuse pieces (public key, secret key, host) are set."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key and self.langfuse_host)


def _require(name: str, env: dict[str, str]) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Required environment variable {name!r} is not set.\n"
            f"Copy .env.example to .env and fill in the required values. "
            f"See README.md → 'Quickstart'."
        )
    return value


def _parse_int(name: str, env: dict[str, str], default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Variable {name!r} must be an integer, got: {raw!r}"
        ) from exc


def load_config(env_file: Path | None = None) -> Config:
    """Load configuration from ``.env`` (or already-set environment variables).

    Args:
        env_file: path to the .env file. Defaults to ``<repo>/.env``.

    Returns:
        Immutable ``Config``.

    Raises:
        ConfigError: when a required variable is missing or a value is invalid.
    """
    if env_file is None:
        env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    env: dict[str, str] = dict(os.environ)

    return Config(
        groq_api_key=_require("GROQ_API_KEY", env),
        model_generation=env.get("AONE_MODEL_GENERATION", DEFAULT_MODEL_GENERATION),
        model_classification=env.get("AONE_MODEL_CLASSIFICATION", DEFAULT_MODEL_CLASSIFICATION),
        embedding_provider=env.get("AONE_EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
        embedding_model=env.get("AONE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        sync_limit=_parse_int("AONE_SYNC_LIMIT", env, DEFAULT_SYNC_LIMIT),
        langfuse_public_key=env.get("LANGFUSE_PUBLIC_KEY") or None,
        langfuse_secret_key=env.get("LANGFUSE_SECRET_KEY") or None,
        langfuse_host=env.get("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
        openai_api_key=env.get("OPENAI_API_KEY") or None,
        gemini_api_key=env.get("GEMINI_API_KEY") or None,
    )
