"""Tests for ``aone.observability.tracing`` (AONE-505)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aone.config import Config
from aone.observability import tracing


def _config(*, with_langfuse: bool = True) -> Config:
    return Config(
        groq_api_key="k",
        model_generation="groq/llama-3.3-70b-versatile",
        model_classification="groq/llama-3.1-8b-instant",
        embedding_provider="local",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        sync_limit=500,
        langfuse_public_key="pk-lf-xxx" if with_langfuse else None,
        langfuse_secret_key="sk-lf-yyy" if with_langfuse else None,
        langfuse_host="https://cloud.langfuse.com",
        anthropic_api_key=None,
        openai_api_key=None,
        gemini_api_key=None,
    )


@pytest.fixture(autouse=True)
def _reset_module_state() -> None:
    """Ensure each test starts with the tracing init flag cleared."""
    tracing._reset_for_tests()
    yield
    tracing._reset_for_tests()


def test_init_tracing_returns_false_when_keys_missing() -> None:
    assert tracing.init_tracing(_config(with_langfuse=False)) is False
    assert tracing.is_initialized() is False


def test_init_tracing_returns_true_when_keys_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(var, raising=False)

    assert tracing.init_tracing(_config(with_langfuse=True)) is True
    assert tracing.is_initialized() is True
    # Env vars are exported for the langfuse SDK to find on import.
    import os

    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-lf-xxx"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-lf-yyy"


def test_init_tracing_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    cfg = _config(with_langfuse=True)
    assert tracing.init_tracing(cfg) is True
    assert tracing.init_tracing(cfg) is True
    assert tracing.is_initialized() is True


def test_observe_is_a_no_op_passthrough_when_disabled() -> None:
    """When Langfuse isn't configured, @observe still wraps the call
    and returns the same value — never raises, never breaks logic."""

    @tracing.observe(name="dummy")
    def doubler(x: int) -> int:
        return x * 2

    # Don't init tracing — the decorator should still work.
    assert doubler(5) == 10


def test_observe_passes_through_arguments_and_return() -> None:
    @tracing.observe(name="echo")
    def echo(a: str, b: int, *, c: float = 0.0) -> dict:
        return {"a": a, "b": b, "c": c}

    assert echo("hi", 1, c=1.5) == {"a": "hi", "b": 1, "c": 1.5}


def test_tag_current_span_is_noop_when_not_initialised() -> None:
    """The helper must NEVER raise even when Langfuse is off."""
    assert tracing.is_initialized() is False
    tracing.tag_current_span(foo="bar", count=3)  # no exception


def test_tag_current_span_swallows_sdk_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Langfuse SDK errors out (offline, auth, …), the agent keeps running."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    tracing.init_tracing(_config(with_langfuse=True))

    class _Boom:
        def update_current_span(self, **_: object) -> None:
            raise RuntimeError("simulated transport failure")

    monkeypatch.setattr(tracing, "get_client", lambda: _Boom())
    # Must not raise.
    tracing.tag_current_span(any="thing")


def test_init_tracing_loads_default_config_when_none_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling init_tracing() with no argument must call load_config()."""
    fake_config = _config(with_langfuse=False)

    with patch(
        "aone.observability.tracing.load_config",
        return_value=fake_config,
    ) as mock_load:
        result = tracing.init_tracing()

    assert mock_load.called
    assert result is False
