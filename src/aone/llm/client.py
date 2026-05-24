"""Provider-agnostic LLM client (ADR-005).

Wraps :mod:`litellm` so the rest of the codebase talks to "the model"
without caring whether it's Groq Llama, Claude Haiku, GPT-4o, or
Gemini. Model selection comes from configuration via the ``role``
parameter: ``GENERATION`` reads ``AONE_MODEL_GENERATION``,
``CLASSIFICATION`` reads ``AONE_MODEL_CLASSIFICATION``.

The wrapper adds three things on top of plain ``litellm.completion``:

1. Config-driven model resolution. Callers say *what kind of call* they
   want, not *which model*. Switching ``AONE_MODEL_GENERATION`` in
   ``.env`` from ``groq/llama-3.3-70b-versatile`` to
   ``anthropic/claude-haiku-4-5`` requires no code changes.
2. A single chokepoint for cross-cutting concerns (retries, future
   Langfuse instrumentation in AONE-505).
3. A typed return value (:class:`CompletionResult`) instead of an
   opaque provider-specific response object.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import litellm

from aone.config import Config, load_config

DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_MAX_TOKENS = 1024


class Role(StrEnum):
    """Logical purpose of a call — maps to a configured model."""

    GENERATION = "generation"
    CLASSIFICATION = "classification"


@dataclass(frozen=True)
class CompletionResult:
    """Outcome of a non-streaming completion call."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Single entry-point for LLM calls in Aone.

    Construct with an explicit :class:`Config` for tests; in production
    the default constructor loads from environment/``.env``.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        num_retries: int = DEFAULT_RETRY_ATTEMPTS,
    ) -> None:
        self._config = config or load_config()
        self._num_retries = num_retries

    # ─── Model resolution ─────────────────────────────────────────

    def model_for(self, role: Role | str) -> str:
        """Resolve a logical role to its configured model ID."""
        role_value = role.value if isinstance(role, Role) else str(role).lower()
        if role_value == Role.GENERATION.value:
            return self._config.model_generation
        if role_value == Role.CLASSIFICATION.value:
            return self._config.model_classification
        raise ValueError(
            f"Unknown role {role!r}. Expected 'generation' or 'classification'."
        )

    # ─── Non-streaming ────────────────────────────────────────────

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        role: Role | str = Role.GENERATION,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        **extra: Any,
    ) -> CompletionResult:
        """Run a completion and return the full result.

        Args:
            messages: chat-completion messages (``role``, ``content`` …).
            role: logical role — picks the model from config.
            model: explicit override that bypasses ``role``. Useful for
                spot-tests against a specific provider.
            max_tokens: cap on output tokens. Required for some providers
                (notably Anthropic) — defaulted to a safe value.
            temperature: sampling temperature (``None`` = provider default).
            **extra: anything else LiteLLM accepts (``top_p``,
                ``response_format``, ``tools`` …).
        """
        resolved_model = model or self.model_for(role)

        response = litellm.completion(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_retries=self._num_retries,
            stream=False,
            **extra,
        )

        return CompletionResult(
            text=response.choices[0].message.content or "",
            model=response.model,
            prompt_tokens=_safe_int(getattr(response.usage, "prompt_tokens", 0))
            if response.usage
            else 0,
            completion_tokens=_safe_int(getattr(response.usage, "completion_tokens", 0))
            if response.usage
            else 0,
        )

    # ─── Streaming ────────────────────────────────────────────────

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        role: Role | str = Role.GENERATION,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        **extra: Any,
    ) -> Iterator[str]:
        """Run a streaming completion, yielding text deltas as they arrive."""
        resolved_model = model or self.model_for(role)

        chunks = litellm.completion(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_retries=self._num_retries,
            stream=True,
            **extra,
        )

        for chunk in chunks:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def _safe_int(value: Any) -> int:
    """Return ``value`` as an int, or 0 if it isn't already one."""
    return value if isinstance(value, int) else 0
