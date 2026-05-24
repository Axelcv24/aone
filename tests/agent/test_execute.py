"""Tests for ``aone.agent.execute`` (AONE-408)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from aone.agent.execute import KNOWN_TOOLS, ExecuteTools, ToolResults, known_tools
from aone.agent.select_tools import _INTENT_TO_TOOLS
from aone.agent.tools.amounts import AggregateResult, Group
from aone.agent.tools.contacts import Contact
from aone.agent.tools.summarize import ThreadSummary
from aone.gmail.types import Email
from aone.llm.client import CompletionResult, LLMClient
from aone.storage.cache import EmailCache
from aone.storage.vector import VectorIndex


class _FakeEmbedder:
    provider_name = "fake"
    model_name = "fake/v1"
    dims = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(t)), float(t.count("a")), float(t.count("b")), 1.0]
            for t in texts
        ]


def _email(
    id_: str,
    *,
    body: str,
    from_: str = "billing@acme.com",
    thread_id: str = "t-default",
    date: int = 1_700_000_000_000,
) -> Email:
    return Email(
        id=id_,
        thread_id=thread_id,
        from_=from_,
        to=["axel@example.com"],
        subject="invoice",
        body_text=body,
        body_html=f"<p>{body}</p>",
        body_clean=body,
        snippet=body[:80],
        internal_date=date,
        labels=["INBOX"],
    )


def _fake_llm(text: str = "ok") -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = CompletionResult(
        text=text,
        model="groq/llama-3.3-70b-versatile",
        prompt_tokens=10,
        completion_tokens=5,
    )
    return client


def _setup(*emails: Email) -> tuple[EmailCache, VectorIndex, MagicMock]:
    cache = EmailCache()
    index = VectorIndex(_FakeEmbedder())
    if emails:
        cache.add_many(list(emails))
        index.add_many(list(emails))
    return cache, index, _fake_llm()


# ─── Per-tool dispatch ───────────────────────────────────────────────


def test_search_emails_runs_with_question_as_query() -> None:
    cache, index, llm = _setup(
        _email("m", body="apple", from_="alice@x.com"),
    )
    dispatcher = ExecuteTools(cache, index, llm)

    result = dispatcher(tool_names=["search_emails"], question="apple pie")

    assert result.search_emails is not None
    assert any(e.id == "m" for e in result.search_emails)
    assert result.aggregate_amounts is None
    assert result.list_contacts is None


def test_search_then_aggregate_amounts_chains_email_ids() -> None:
    cache, index, llm = _setup(
        _email(
            "m1",
            body="Invoice for $1,200.00 USD — OVERDUE",
            from_="billing@acme.com",
        ),
        _email(
            "m2",
            body="Invoice for $1,500.00 USD pending",
            from_="billing@acme.com",
        ),
    )
    dispatcher = ExecuteTools(cache, index, llm)

    result = dispatcher(
        tool_names=["search_emails", "aggregate_amounts"],
        question="invoice",
    )

    assert result.aggregate_amounts is not None
    assert isinstance(result.aggregate_amounts, AggregateResult)
    totals = result.aggregate_amounts.grand_total_by_currency
    assert totals["USD"] == Decimal("2700.00")


def test_search_then_summarize_thread_chains_email_ids() -> None:
    cache, index, llm = _setup(
        _email("m1", body="first message"),
        _email("m2", body="second message"),
    )
    dispatcher = ExecuteTools(cache, index, llm)

    result = dispatcher(
        tool_names=["search_emails", "summarize_thread"],
        question="anything",
    )

    assert isinstance(result.summarize_thread, ThreadSummary)
    assert result.summarize_thread.email_count > 0
    llm.complete.assert_called()  # summarize hit the LLM


def test_summarize_without_prior_search_returns_empty_summary() -> None:
    """Defensive: if select_tools mis-orders, summarize_thread runs
    against an empty email list and gracefully returns an empty
    ThreadSummary instead of crashing."""
    cache, index, llm = _setup()
    dispatcher = ExecuteTools(cache, index, llm)

    result = dispatcher(tool_names=["summarize_thread"], question="anything")
    assert isinstance(result.summarize_thread, ThreadSummary)
    assert result.summarize_thread.email_count == 0
    assert result.summarize_thread.text == ""
    llm.complete.assert_not_called()


def test_list_contacts_runs_without_question_args() -> None:
    cache, index, llm = _setup(
        _email("m1", from_="alice@x.com", body="hi"),
        _email("m2", from_="alice@x.com", body="hello"),
        _email("m3", from_="bob@y.com", body="hey"),
    )
    dispatcher = ExecuteTools(cache, index, llm)

    result = dispatcher(tool_names=["list_contacts"], question="anything")

    assert result.list_contacts is not None
    assert all(isinstance(c, Contact) for c in result.list_contacts)
    addresses = {c.email_address for c in result.list_contacts}
    assert addresses == {"alice@x.com", "bob@y.com"}


def test_get_thread_in_tool_list_is_skipped_cleanly() -> None:
    """No current intent dispatches get_thread directly; a stray entry
    must not crash the run."""
    cache, index, llm = _setup(_email("m", body="hi"))
    dispatcher = ExecuteTools(cache, index, llm)

    result = dispatcher(tool_names=["get_thread"], question="anything")

    assert result.get_thread is None


# ─── Error handling ──────────────────────────────────────────────────


def test_unknown_tool_name_raises() -> None:
    cache, index, llm = _setup()
    dispatcher = ExecuteTools(cache, index, llm)
    with pytest.raises(ValueError, match="Unknown tool name"):
        dispatcher(tool_names=["does-not-exist"], question="anything")


def test_empty_tool_list_returns_blank_results() -> None:
    cache, index, llm = _setup()
    dispatcher = ExecuteTools(cache, index, llm)

    result = dispatcher(tool_names=[], question="anything")

    assert isinstance(result, ToolResults)
    assert result.search_emails is None
    assert result.list_contacts is None
    assert result.aggregate_amounts is None
    assert result.summarize_thread is None


# ─── ToolResults helpers ─────────────────────────────────────────────


def test_emails_for_response_falls_back_through_search_then_thread() -> None:
    e1 = _email("a", body="x")
    e2 = _email("b", body="y")
    no_emails = ToolResults()
    only_search = ToolResults(search_emails=[e1])
    only_thread = ToolResults(get_thread=[e2])

    assert no_emails.emails_for_response == []
    assert only_search.emails_for_response == [e1]
    assert only_thread.emails_for_response == [e2]


# ─── Wiring invariants ──────────────────────────────────────────────


def test_every_tool_in_select_tools_is_known_to_execute_tools() -> None:
    """select_tools (AONE-402) must never reference a tool the
    dispatcher cannot resolve."""
    referenced = {name for tools in _INTENT_TO_TOOLS.values() for name in tools}
    assert referenced.issubset(KNOWN_TOOLS), (
        f"select_tools references unknown tools: {referenced - KNOWN_TOOLS}"
    )


def test_known_tools_helper_returns_the_constant() -> None:
    assert known_tools() == KNOWN_TOOLS


# ─── Stub-style smoke test for full chain ────────────────────────────


def test_full_chain_for_aggregate_intent_produces_groups_and_emails() -> None:
    """Smoke: walk the AGGREGATE_AMOUNTS path end-to-end on real
    cache + index (with the fake embedder), exercising both
    search_emails and aggregate_amounts together."""
    cache, index, llm = _setup(
        _email(
            "a", body="Invoice #1 — $500 USD overdue", from_="billing@acme.com",
        ),
        _email(
            "b", body="Invoice #2 — $700 USD pending", from_="billing@acme.com",
        ),
    )
    dispatcher = ExecuteTools(cache, index, llm)

    result = dispatcher(
        tool_names=["search_emails", "aggregate_amounts"],
        question="invoice",
    )

    assert result.search_emails is not None and len(result.search_emails) > 0
    assert result.aggregate_amounts is not None
    groups: list[Group] = result.aggregate_amounts.groups
    assert any(g.key == "billing@acme.com" for g in groups)
    acme_total = next(g.total for g in groups if g.key == "billing@acme.com")
    assert acme_total == Decimal("1200")
