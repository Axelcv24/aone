"""Tests for ``aone.llm.client``."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aone.config import Config
from aone.llm.client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY_ATTEMPTS,
    CompletionResult,
    LLMClient,
    Role,
)


def _config(**overrides: Any) -> Config:
    """Build a Config with sensible defaults; tests override what they need."""
    defaults = {
        "groq_api_key": "fake-groq-key",
        "model_generation": "groq/llama-3.3-70b-versatile",
        "model_classification": "groq/llama-3.1-8b-instant",
        "embedding_provider": "local",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "sync_limit": 500,
        "langfuse_public_key": None,
        "langfuse_secret_key": None,
        "langfuse_host": "https://cloud.langfuse.com",
        "anthropic_api_key": None,
        "openai_api_key": None,
        "gemini_api_key": None,
    }
    defaults.update(overrides)
    return Config(**defaults)


def _fake_response(
    *,
    text: str = "hi",
    model: str = "groq/llama-3.3-70b-versatile",
    prompt_tokens: int = 5,
    completion_tokens: int = 10,
) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = text
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


# ─── model_for ───────────────────────────────────────────────────────


def test_model_for_generation_uses_config_generation() -> None:
    client = LLMClient(_config())
    assert client.model_for(Role.GENERATION) == "groq/llama-3.3-70b-versatile"


def test_model_for_classification_uses_config_classification() -> None:
    client = LLMClient(_config())
    assert client.model_for(Role.CLASSIFICATION) == "groq/llama-3.1-8b-instant"


def test_model_for_accepts_string_role() -> None:
    client = LLMClient(_config())
    assert client.model_for("generation") == "groq/llama-3.3-70b-versatile"
    assert client.model_for("CLASSIFICATION") == "groq/llama-3.1-8b-instant"


def test_model_for_unknown_role_raises() -> None:
    client = LLMClient(_config())
    with pytest.raises(ValueError, match="Unknown role"):
        client.model_for("magic")


# ─── complete() ──────────────────────────────────────────────────────


@patch("aone.llm.client.litellm.completion")
def test_complete_uses_configured_generation_model_by_default(
    mock_completion: MagicMock,
) -> None:
    mock_completion.return_value = _fake_response()
    client = LLMClient(_config())

    client.complete([{"role": "user", "content": "hi"}])

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["model"] == "groq/llama-3.3-70b-versatile"
    assert kwargs["stream"] is False
    assert kwargs["num_retries"] == DEFAULT_RETRY_ATTEMPTS
    assert kwargs["max_tokens"] == DEFAULT_MAX_TOKENS


@patch("aone.llm.client.litellm.completion")
def test_complete_uses_classification_model_when_role_is_classification(
    mock_completion: MagicMock,
) -> None:
    mock_completion.return_value = _fake_response(model="groq/llama-3.1-8b-instant")
    client = LLMClient(_config())

    client.complete([{"role": "user", "content": "hi"}], role=Role.CLASSIFICATION)

    assert mock_completion.call_args.kwargs["model"] == "groq/llama-3.1-8b-instant"


@patch("aone.llm.client.litellm.completion")
def test_complete_explicit_model_overrides_role(mock_completion: MagicMock) -> None:
    mock_completion.return_value = _fake_response(model="anthropic/claude-haiku-4-5")
    client = LLMClient(_config())

    client.complete(
        [{"role": "user", "content": "hi"}],
        model="anthropic/claude-haiku-4-5",
    )

    assert mock_completion.call_args.kwargs["model"] == "anthropic/claude-haiku-4-5"


@patch("aone.llm.client.litellm.completion")
def test_complete_returns_text_and_usage(mock_completion: MagicMock) -> None:
    mock_completion.return_value = _fake_response(
        text="hello there", prompt_tokens=7, completion_tokens=3
    )
    client = LLMClient(_config())

    result = client.complete([{"role": "user", "content": "hi"}])

    assert isinstance(result, CompletionResult)
    assert result.text == "hello there"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3
    assert result.total_tokens == 10


@patch("aone.llm.client.litellm.completion")
def test_complete_passes_through_max_tokens_and_temperature(
    mock_completion: MagicMock,
) -> None:
    mock_completion.return_value = _fake_response()
    client = LLMClient(_config())

    client.complete(
        [{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=0.2,
    )

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.2


@patch("aone.llm.client.litellm.completion")
def test_complete_passes_extra_kwargs_through(mock_completion: MagicMock) -> None:
    mock_completion.return_value = _fake_response()
    client = LLMClient(_config())

    client.complete(
        [{"role": "user", "content": "hi"}],
        top_p=0.9,
        response_format={"type": "json_object"},
    )

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["top_p"] == 0.9
    assert kwargs["response_format"] == {"type": "json_object"}


@patch("aone.llm.client.litellm.completion")
def test_complete_uses_custom_num_retries(mock_completion: MagicMock) -> None:
    mock_completion.return_value = _fake_response()
    client = LLMClient(_config(), num_retries=7)

    client.complete([{"role": "user", "content": "hi"}])

    assert mock_completion.call_args.kwargs["num_retries"] == 7


@patch("aone.llm.client.litellm.completion")
def test_complete_handles_missing_usage_gracefully(mock_completion: MagicMock) -> None:
    """Some providers don't return token usage — the wrapper must not crash."""
    response = MagicMock()
    response.choices[0].message.content = "ok"
    response.model = "groq/llama-3.3-70b-versatile"
    response.usage = None
    mock_completion.return_value = response

    client = LLMClient(_config())
    result = client.complete([{"role": "user", "content": "hi"}])

    assert result.text == "ok"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0


@patch("aone.llm.client.litellm.completion")
def test_complete_handles_none_content_gracefully(mock_completion: MagicMock) -> None:
    response = MagicMock()
    response.choices[0].message.content = None
    response.model = "groq/llama-3.3-70b-versatile"
    response.usage.prompt_tokens = 1
    response.usage.completion_tokens = 0
    mock_completion.return_value = response

    client = LLMClient(_config())
    assert client.complete([{"role": "user", "content": "hi"}]).text == ""


# ─── stream() ────────────────────────────────────────────────────────


def _delta_chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


@patch("aone.llm.client.litellm.completion")
def test_stream_yields_text_deltas(mock_completion: MagicMock) -> None:
    mock_completion.return_value = iter(
        [_delta_chunk("Hello "), _delta_chunk("world"), _delta_chunk(None)]
    )
    client = LLMClient(_config())

    text = "".join(client.stream([{"role": "user", "content": "hi"}]))

    assert text == "Hello world"
    assert mock_completion.call_args.kwargs["stream"] is True


@patch("aone.llm.client.litellm.completion")
def test_stream_uses_classification_model_when_requested(
    mock_completion: MagicMock,
) -> None:
    mock_completion.return_value = iter([])
    client = LLMClient(_config())

    list(client.stream([{"role": "user", "content": "hi"}], role=Role.CLASSIFICATION))

    assert mock_completion.call_args.kwargs["model"] == "groq/llama-3.1-8b-instant"


@patch("aone.llm.client.litellm.completion")
def test_stream_explicit_model_overrides_role(mock_completion: MagicMock) -> None:
    mock_completion.return_value = iter([])
    client = LLMClient(_config())

    list(
        client.stream(
            [{"role": "user", "content": "hi"}],
            model="openai/gpt-4o-mini",
        )
    )

    assert mock_completion.call_args.kwargs["model"] == "openai/gpt-4o-mini"


# ─── Provider-agnostic switching (the headline AC) ───────────────────


@patch("aone.llm.client.litellm.completion")
def test_switching_generation_model_via_config_does_not_require_code_change(
    mock_completion: MagicMock,
) -> None:
    """ADR-005 in one test: the same call routes to a different provider
    based purely on configuration."""
    mock_completion.return_value = _fake_response()

    # Config A: free Groq stack (current v0 default)
    client_a = LLMClient(_config(model_generation="groq/llama-3.3-70b-versatile"))
    client_a.complete([{"role": "user", "content": "hi"}])
    assert mock_completion.call_args.kwargs["model"] == "groq/llama-3.3-70b-versatile"

    # Config B: paid Claude — identical call, different .env
    client_b = LLMClient(_config(model_generation="anthropic/claude-haiku-4-5"))
    client_b.complete([{"role": "user", "content": "hi"}])
    assert mock_completion.call_args.kwargs["model"] == "anthropic/claude-haiku-4-5"

    # Config C: paid OpenAI
    client_c = LLMClient(_config(model_generation="openai/gpt-4o-mini"))
    client_c.complete([{"role": "user", "content": "hi"}])
    assert mock_completion.call_args.kwargs["model"] == "openai/gpt-4o-mini"
