"""Langfuse tracing for the Aone agent (AONE-505).

This module owns the observability cross-cutting concern. Two surfaces:

* :func:`init_tracing` — call once during CLI startup. Reads the
  ``LANGFUSE_*`` keys from :class:`Config` and wires LiteLLM to send
  every LLM completion to Langfuse as a *generation*. No-op (returns
  ``False``) when Langfuse is not configured, so the rest of the
  codebase doesn't have to branch on availability.

* :data:`observe` — the decorator re-exported from the Langfuse SDK.
  Wrap any function whose inputs/outputs are interesting (nodes,
  tools, the top-level ``ask``). When Langfuse isn't initialised the
  decorator passes through cleanly — no traces emitted, no errors.

The result is a full agent trace per ``aone ask`` invocation:

    Trace: aone-ask
      ├─ Span: classify_intent
      │    └─ Generation: groq/llama-3.1-8b-instant
      ├─ Span: select_tools (no LLM)
      ├─ Span: execute_tools
      │    └─ Generation(s): from summarize_thread / aggregate
      └─ Span: generate_response
           └─ Generation: groq/llama-3.3-70b-versatile

Visible at ``LANGFUSE_HOST`` (default ``https://cloud.langfuse.com``).
"""

from __future__ import annotations

import logging
import os

from langfuse import observe

from aone.config import Config, load_config

_logger = logging.getLogger(__name__)
_initialized = False


def init_tracing(config: Config | None = None) -> bool:
    """Initialise Langfuse if the user has configured the keys.

    Idempotent — subsequent calls are no-ops.

    Tracing is delivered through the :data:`observe` decorator applied
    to the agent's nodes (``classify_intent``, ``execute_tools``,
    ``generate_response``, ``ask``). LiteLLM's built-in Langfuse
    callback is intentionally NOT registered here: as of langfuse-py
    4.x it crashes (it still imports ``langfuse.version`` which v4
    removed). Tracking the LLM calls themselves can be revisited when
    LiteLLM updates the integration.

    Args:
        config: optional pre-loaded :class:`Config`. When ``None`` we
            load fresh from the environment.

    Returns:
        ``True`` when Langfuse is now active, ``False`` when keys are
        missing or initialisation was already done.
    """
    global _initialized

    if _initialized:
        return True

    cfg = config or load_config()
    if not cfg.langfuse_enabled:
        _logger.debug("Langfuse keys not set; tracing disabled.")
        return False

    # Langfuse SDK auto-configures from these env vars. We mirror the
    # values from the Config so the rest of the SDK (and litellm's
    # langfuse callback) finds them no matter how the process was
    # started.
    os.environ["LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key or ""
    os.environ["LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key or ""
    os.environ["LANGFUSE_HOST"] = cfg.langfuse_host

    _initialized = True
    _logger.info("Langfuse tracing enabled (host=%s).", cfg.langfuse_host)
    return True


def is_initialized() -> bool:
    """True if :func:`init_tracing` succeeded at least once this process."""
    return _initialized


def _reset_for_tests() -> None:
    """Test-only helper: drop the cached init flag so ``init_tracing``
    runs again. Tests use this between cases."""
    global _initialized
    _initialized = False


__all__ = ["init_tracing", "is_initialized", "observe"]
