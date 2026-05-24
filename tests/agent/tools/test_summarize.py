"""Tests for ``aone.agent.tools.summarize`` (AONE-407)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aone.agent.tools.summarize import (
    DEFAULT_MAX_BULLETS,
    SummarizeThread,
    ThreadSummary,
)
from aone.gmail.types import Email
from aone.llm.client import CompletionResult, LLMClient, Role
from aone.storage.cache import EmailCache


def _email(
    id_: str,
    *,
    thread_id: str = "T1",
    body: str = "Email body content.",
    from_: str = "alice@x.com",
    date: int = 1_700_000_000_000,
    subject: str = "subject",
) -> Email:
    return Email(
        id=id_,
        thread_id=thread_id,
        from_=from_,
        to=["axel@example.com"],
        subject=subject,
        body_text=body,
        body_html=f"<p>{body}</p>",
        body_clean=body,
        snippet=body[:80],
        internal_date=date,
        labels=["INBOX"],
    )


def _cache(*emails: Email) -> EmailCache:
    cache = EmailCache()
    cache.add_many(list(emails))
    return cache


def _fake_llm(text: str = "• Summary point one\n• Summary point two") -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = CompletionResult(
        text=text,
        model="groq/llama-3.3-70b-versatile",
        prompt_tokens=120,
        completion_tokens=45,
    )
    return client


# ─── Selector validation ─────────────────────────────────────────────


def test_requires_either_thread_id_or_email_ids() -> None:
    tool = SummarizeThread(EmailCache(), _fake_llm())
    with pytest.raises(ValueError, match="Provide either"):
        tool()


def test_rejects_both_thread_id_and_email_ids() -> None:
    tool = SummarizeThread(EmailCache(), _fake_llm())
    with pytest.raises(ValueError, match="not both"):
        tool(thread_id="T1", email_ids=["m1"])


# ─── Thread-id path ──────────────────────────────────────────────────


def test_summarises_full_thread_by_thread_id() -> None:
    cache = _cache(
        _email("m2", thread_id="T1", date=200, body="second message"),
        _email("m1", thread_id="T1", date=100, body="first message"),
        _email("m3", thread_id="T1", date=300, body="third message"),
        _email("other", thread_id="T2", date=999, body="unrelated"),
    )
    llm = _fake_llm()
    tool = SummarizeThread(cache, llm)

    summary = tool(thread_id="T1")

    assert isinstance(summary, ThreadSummary)
    assert summary.email_count == 3
    assert summary.text.startswith("• Summary")
    assert summary.model == "groq/llama-3.3-70b-versatile"
    assert summary.total_tokens == 120 + 45

    # The user-content prompt must include all three bodies in order.
    user_message = llm.complete.call_args.kwargs["messages"][1]["content"]
    assert "first message" in user_message
    assert "second message" in user_message
    assert "third message" in user_message
    assert user_message.index("first message") < user_message.index("third message")


def test_unknown_thread_returns_empty_without_calling_llm() -> None:
    llm = _fake_llm()
    tool = SummarizeThread(_cache(_email("m", thread_id="T1")), llm)

    summary = tool(thread_id="DOES-NOT-EXIST")

    assert summary.email_count == 0
    assert summary.text == ""
    llm.complete.assert_not_called()


# ─── Email-ids path ──────────────────────────────────────────────────


def test_summarises_explicit_email_ids() -> None:
    cache = _cache(
        _email("a", date=100, body="first"),
        _email("b", date=200, body="second"),
        _email("c", date=300, body="third"),
    )
    llm = _fake_llm()
    tool = SummarizeThread(cache, llm)

    summary = tool(email_ids=["a", "c"])
    assert summary.email_count == 2


def test_email_ids_sorted_oldest_first_in_prompt() -> None:
    """Regardless of input order, the transcript is chronological."""
    cache = _cache(
        _email("a", date=300, body="latest"),
        _email("b", date=100, body="earliest"),
        _email("c", date=200, body="middle"),
    )
    llm = _fake_llm()
    tool = SummarizeThread(cache, llm)

    tool(email_ids=["a", "b", "c"])
    user_message = llm.complete.call_args.kwargs["messages"][1]["content"]
    assert user_message.index("earliest") < user_message.index("middle") < user_message.index("latest")


def test_unknown_email_ids_are_skipped_silently() -> None:
    cache = _cache(_email("a", body="alpha"))
    llm = _fake_llm()
    tool = SummarizeThread(cache, llm)

    summary = tool(email_ids=["a", "missing"])
    assert summary.email_count == 1


def test_empty_email_ids_list_returns_empty_without_llm() -> None:
    llm = _fake_llm()
    tool = SummarizeThread(_cache(), llm)
    summary = tool(email_ids=[])
    assert summary.email_count == 0
    llm.complete.assert_not_called()


# ─── Metadata ────────────────────────────────────────────────────────


def test_summary_metadata_lists_unique_senders() -> None:
    cache = _cache(
        _email("a", from_="alice@x.com", date=100),
        _email("b", from_="Bob <bob@y.com>", date=200),
        _email("c", from_="alice@x.com", date=300),
    )
    summary = SummarizeThread(cache, _fake_llm())(email_ids=["a", "b", "c"])
    assert summary.senders == ["alice@x.com", "bob@y.com"]


def test_summary_metadata_tracks_date_range() -> None:
    cache = _cache(
        _email("a", date=100),
        _email("b", date=300),
        _email("c", date=200),
    )
    summary = SummarizeThread(cache, _fake_llm())(email_ids=["a", "b", "c"])
    assert summary.earliest_date_ms == 100
    assert summary.latest_date_ms == 300


# ─── Prompt construction ─────────────────────────────────────────────


def test_uses_generation_role_with_temperature_above_zero() -> None:
    """Some creativity helps summary fluency; we don't want temperature 0."""
    cache = _cache(_email("m", body="hi"))
    llm = _fake_llm()

    SummarizeThread(cache, llm)(email_ids=["m"])

    kwargs = llm.complete.call_args.kwargs
    assert kwargs["role"] == Role.GENERATION
    assert kwargs["temperature"] > 0


def test_system_prompt_mentions_max_bullets() -> None:
    cache = _cache(_email("m", body="hi"))
    llm = _fake_llm()

    SummarizeThread(cache, llm)(email_ids=["m"], max_bullets=7)

    system_msg = llm.complete.call_args.kwargs["messages"][0]["content"]
    assert "7 bullets" in system_msg


def test_default_max_bullets_used_when_not_passed() -> None:
    cache = _cache(_email("m", body="hi"))
    llm = _fake_llm()

    SummarizeThread(cache, llm)(email_ids=["m"])

    system_msg = llm.complete.call_args.kwargs["messages"][0]["content"]
    assert f"{DEFAULT_MAX_BULLETS} bullets" in system_msg


def test_prompt_includes_email_headers() -> None:
    cache = _cache(
        _email(
            "m",
            from_="Alice <alice@x.com>",
            subject="Q3 budget approval",
            date=1_700_000_000_000,
            body="Please approve.",
        ),
    )
    llm = _fake_llm()
    SummarizeThread(cache, llm)(email_ids=["m"])

    user_message = llm.complete.call_args.kwargs["messages"][1]["content"]
    assert "alice@x.com" in user_message
    assert "Q3 budget approval" in user_message
    assert "2023-11-14" in user_message  # ms 1_700_000_000_000 → 2023-11-14 UTC


# ─── Schema ──────────────────────────────────────────────────────────


def test_input_schema_lists_oneof_selector() -> None:
    schema = SummarizeThread.input_schema()
    assert {"required": ["thread_id"]} in schema["oneOf"]
    assert {"required": ["email_ids"]} in schema["oneOf"]
    assert "thread_id" in schema["properties"]
    assert "email_ids" in schema["properties"]
    assert schema["properties"]["max_bullets"]["default"] == DEFAULT_MAX_BULLETS
