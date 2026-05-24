"""Tests for ``aone.agent.classify`` (AONE-401).

Two layers:

* **Unit tests** with a mocked :class:`LLMClient` cover parsing,
  prompt construction, fallback behaviour, and config plumbing. Fast
  and deterministic.

* **Golden-set integration test** runs the real classifier against
  ``evals/intent_classification.jsonl`` and asserts the ≥85% accuracy
  bar from the AC. Uses the real Groq endpoint; skipped automatically
  when ``GROQ_API_KEY`` isn't set (e.g. on CI without secrets).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aone.agent.classify import classify_intent
from aone.agent.intents import Intent
from aone.config import ConfigError, load_config
from aone.llm.client import CompletionResult, LLMClient, Role


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_SET_PATH = PROJECT_ROOT / "evals" / "intent_classification.jsonl"
ACCURACY_TARGET = 0.85


def _fake_client(reply: str) -> MagicMock:
    """Build a mocked LLMClient that returns ``reply`` for any complete() call."""
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = CompletionResult(
        text=reply,
        model="groq/llama-3.1-8b-instant",
        prompt_tokens=10,
        completion_tokens=2,
    )
    return client


# ─── Intent.parse — robustness ───────────────────────────────────────


def test_intent_parse_exact_match() -> None:
    assert Intent.parse("aggregate_amounts") == Intent.AGGREGATE_AMOUNTS
    assert Intent.parse("summarize") == Intent.SUMMARIZE
    assert Intent.parse("find_emails") == Intent.FIND_EMAILS
    assert Intent.parse("list_contacts") == Intent.LIST_CONTACTS
    assert Intent.parse("general_qa") == Intent.GENERAL_QA


def test_intent_parse_tolerates_whitespace_and_case() -> None:
    assert Intent.parse("  AGGREGATE_AMOUNTS  ") == Intent.AGGREGATE_AMOUNTS
    assert Intent.parse("Summarize") == Intent.SUMMARIZE


def test_intent_parse_tolerates_trailing_punctuation() -> None:
    assert Intent.parse("aggregate_amounts.") == Intent.AGGREGATE_AMOUNTS
    assert Intent.parse('"summarize"') == Intent.SUMMARIZE
    assert Intent.parse("find_emails!") == Intent.FIND_EMAILS


def test_intent_parse_picks_intent_from_chatty_reply() -> None:
    """Small models sometimes say 'the intent is X' — we still extract X."""
    assert Intent.parse("the intent is summarize") == Intent.SUMMARIZE


def test_intent_parse_unknown_falls_back_to_general_qa() -> None:
    assert Intent.parse("garbage") == Intent.GENERAL_QA
    assert Intent.parse("") == Intent.GENERAL_QA
    assert Intent.parse("   ") == Intent.GENERAL_QA


# ─── classify_intent — wiring ────────────────────────────────────────


def test_classify_intent_uses_classification_role() -> None:
    client = _fake_client("aggregate_amounts")
    classify_intent("¿cuánto me debe Acme?", client=client)

    _, kwargs = client.complete.call_args
    assert kwargs["role"] == Role.CLASSIFICATION
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 15


def test_classify_intent_passes_question_as_user_message() -> None:
    client = _fake_client("summarize")
    classify_intent("resúmeme la semana", client=client)

    _, kwargs = client.complete.call_args
    messages = kwargs["messages"]
    assert messages[-1] == {"role": "user", "content": "resúmeme la semana"}
    assert messages[0]["role"] == "system"


def test_classify_intent_returns_parsed_intent() -> None:
    client = _fake_client("aggregate_amounts")
    assert classify_intent("anything", client=client) == Intent.AGGREGATE_AMOUNTS


def test_classify_intent_falls_back_on_garbage_reply() -> None:
    client = _fake_client("???")
    assert classify_intent("anything", client=client) == Intent.GENERAL_QA


@patch("aone.agent.classify.LLMClient")
def test_classify_intent_creates_default_client_when_none_given(
    mock_llm_cls: MagicMock,
) -> None:
    mock_llm_cls.return_value = _fake_client("find_emails")
    result = classify_intent("show me emails about X")
    assert result == Intent.FIND_EMAILS
    mock_llm_cls.assert_called_once_with()


# ─── Golden-set integration ──────────────────────────────────────────


def _load_golden_set() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_SET_PATH.read_text().splitlines() if line.strip()]


def _maybe_skip_without_api_key() -> None:
    try:
        cfg = load_config()
    except ConfigError as exc:
        pytest.skip(f"Config not loadable, skipping integration test: {exc}")
    if not cfg.groq_api_key:
        pytest.skip("GROQ_API_KEY not set; skipping integration test")


def test_golden_set_file_has_at_least_20_examples() -> None:
    examples = _load_golden_set()
    assert len(examples) >= 20, (
        f"Golden set has {len(examples)} examples; ticket requires at least 20."
    )


def test_golden_set_examples_use_known_intents() -> None:
    valid = {i.value for i in Intent}
    for ex in _load_golden_set():
        assert ex["expected_intent"] in valid, (
            f"Unknown intent in golden set: {ex['expected_intent']!r}"
        )


@pytest.mark.integration
def test_classify_intent_accuracy_on_golden_set() -> None:
    """≥85% accuracy on the seeded examples (AONE-401 AC)."""
    _maybe_skip_without_api_key()

    examples = _load_golden_set()
    client = LLMClient()  # uses real config (Groq Llama 3.1 8B by default)

    correct = 0
    failures: list[tuple[str, str, str]] = []
    for ex in examples:
        predicted = classify_intent(ex["question"], client=client)
        if predicted.value == ex["expected_intent"]:
            correct += 1
        else:
            failures.append((ex["question"], ex["expected_intent"], predicted.value))
        # Groq free tier caps at 30 RPM on the 8B classifier; pace
        # ourselves so re-running the test back-to-back doesn't trip it.
        time.sleep(0.5)

    accuracy = correct / len(examples)
    if failures:
        report = "\n".join(
            f"  '{q}' → expected {exp}, got {got}" for q, exp, got in failures
        )
        print(
            f"\nClassification accuracy: {accuracy:.1%} "
            f"({correct}/{len(examples)})\nMisclassifications:\n{report}"
        )

    assert accuracy >= ACCURACY_TARGET, (
        f"Accuracy {accuracy:.1%} is below target {ACCURACY_TARGET:.0%}. "
        f"See stdout for which examples failed."
    )
