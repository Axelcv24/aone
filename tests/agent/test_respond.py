"""Tests for ``aone.agent.respond`` (AONE-409)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from aone.agent.execute import ToolResults
from aone.agent.intents import Intent
from aone.agent.respond import (
    NO_RESULTS_RESPONSE,
    AgentResponse,
    GenerateResponse,
)
from aone.agent.tools.amounts import AggregateResult, AmountMatch, Group
from aone.agent.tools.contacts import Contact
from aone.agent.tools.summarize import ThreadSummary
from aone.gmail.types import Email
from aone.llm.client import CompletionResult, LLMClient, Role


def _email(
    id_: str,
    *,
    subject: str = "subj",
    from_: str = "alice@x.com",
    date: int = 1_700_000_000_000,
    body: str = "body",
) -> Email:
    return Email(
        id=id_,
        thread_id=f"t-{id_}",
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


def _fake_llm(text: str = "Acme owes you $3,450.00 USD across 3 invoices.") -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = CompletionResult(
        text=text,
        model="groq/llama-3.3-70b-versatile",
        prompt_tokens=180,
        completion_tokens=55,
    )
    return client


# ─── Empty / no-data short-circuit ───────────────────────────────────


def test_returns_canned_no_results_when_no_tools_ran() -> None:
    llm = _fake_llm()
    node = GenerateResponse(llm)

    response = node(
        question="anything",
        intent=Intent.GENERAL_QA,
        tool_results=ToolResults(),
    )

    assert response.text == NO_RESULTS_RESPONSE
    assert response.intent == Intent.GENERAL_QA
    assert response.tools_used == []
    assert response.model == ""
    assert response.total_tokens == 0
    llm.complete.assert_not_called()


def test_returns_no_results_when_search_returned_empty_list() -> None:
    llm = _fake_llm()
    node = GenerateResponse(llm)

    response = node(
        question="anything",
        intent=Intent.FIND_EMAILS,
        tool_results=ToolResults(search_emails=[]),
    )

    # tools_used surfaces search_emails (it ran, just empty); short-
    # circuit still kicks in because no actual data.
    assert response.text == NO_RESULTS_RESPONSE
    assert response.tools_used == ["search_emails"]
    llm.complete.assert_not_called()


# ─── LLM invocation contract ─────────────────────────────────────────


def test_calls_llm_with_generation_role_and_low_temperature() -> None:
    llm = _fake_llm()
    node = GenerateResponse(llm)
    node(
        question="cuánto me debe Acme",
        intent=Intent.AGGREGATE_AMOUNTS,
        tool_results=_aggregate_results(),
    )

    kwargs = llm.complete.call_args.kwargs
    assert kwargs["role"] == Role.GENERATION
    assert 0 < kwargs["temperature"] < 1
    assert kwargs["max_tokens"] > 0


def test_passes_through_max_tokens_override() -> None:
    llm = _fake_llm()
    GenerateResponse(llm)(
        question="x",
        intent=Intent.AGGREGATE_AMOUNTS,
        tool_results=_aggregate_results(),
        max_tokens=2048,
    )
    assert llm.complete.call_args.kwargs["max_tokens"] == 2048


def test_question_and_intent_appear_in_user_prompt() -> None:
    llm = _fake_llm()
    GenerateResponse(llm)(
        question="¿cuánto me debe Acme?",
        intent=Intent.AGGREGATE_AMOUNTS,
        tool_results=_aggregate_results(),
    )

    user_message = llm.complete.call_args.kwargs["messages"][1]["content"]
    assert "¿cuánto me debe Acme?" in user_message
    assert "aggregate_amounts" in user_message


# ─── Per-tool context rendering ──────────────────────────────────────


def test_aggregate_context_includes_groups_and_grand_total() -> None:
    llm = _fake_llm()
    GenerateResponse(llm)(
        question="cuánto me debe Acme",
        intent=Intent.AGGREGATE_AMOUNTS,
        tool_results=_aggregate_results(),
    )
    ctx = llm.complete.call_args.kwargs["messages"][1]["content"]

    assert "billing@acme.com: USD 3,450.00" in ctx
    assert "Grand total by currency: USD 3,450.00" in ctx
    assert "[acme-1024]" in ctx  # match-level citation in the underlying list


def test_contacts_context_lists_each_contact() -> None:
    contacts = [
        Contact(
            email_address="alice@x.com",
            name="Alice",
            message_count=5,
            first_seen_ms=1_700_000_000_000,
            last_seen_ms=1_700_000_500_000,
        ),
        Contact(
            email_address="bob@y.com",
            name="",
            message_count=2,
            first_seen_ms=1_700_000_200_000,
            last_seen_ms=1_700_000_300_000,
        ),
    ]
    llm = _fake_llm()
    GenerateResponse(llm)(
        question="who writes me most",
        intent=Intent.LIST_CONTACTS,
        tool_results=ToolResults(list_contacts=contacts),
    )
    ctx = llm.complete.call_args.kwargs["messages"][1]["content"]

    assert "Alice <alice@x.com>" in ctx
    assert "5 messages" in ctx
    assert "bob@y.com" in ctx
    assert "2 messages" in ctx


def test_summary_context_includes_thread_summary_text() -> None:
    summary = ThreadSummary(
        text="• Q3 budget approved\n• Deadline 2026-09-30",
        email_count=3,
        senders=["alice@x.com", "bob@y.com"],
        earliest_date_ms=1_700_000_000_000,
        latest_date_ms=1_700_000_500_000,
        model="groq/llama-3.3-70b-versatile",
        total_tokens=120,
    )
    llm = _fake_llm()
    GenerateResponse(llm)(
        question="resume mis conversaciones de Q3",
        intent=Intent.SUMMARIZE,
        tool_results=ToolResults(
            search_emails=[_email("a"), _email("b"), _email("c")],
            summarize_thread=summary,
        ),
    )
    ctx = llm.complete.call_args.kwargs["messages"][1]["content"]

    assert "Q3 budget approved" in ctx
    assert "Participants: alice@x.com, bob@y.com" in ctx


def test_email_context_caps_at_max_emails() -> None:
    """Pull >8 emails into ToolResults; only 8 reach the context block."""
    emails = [_email(f"m{i}", subject=f"Subject {i}") for i in range(20)]
    llm = _fake_llm()
    GenerateResponse(llm)(
        question="find emails",
        intent=Intent.FIND_EMAILS,
        tool_results=ToolResults(search_emails=emails),
    )
    ctx = llm.complete.call_args.kwargs["messages"][1]["content"]

    # The header should explicitly say "8 of 20 shown".
    assert "8 of 20 shown" in ctx


# ─── tools_used / citations metadata ─────────────────────────────────


def test_tools_used_lists_only_non_none_fields() -> None:
    llm = _fake_llm()
    response = GenerateResponse(llm)(
        question="x",
        intent=Intent.AGGREGATE_AMOUNTS,
        tool_results=_aggregate_results(),
    )
    assert "search_emails" in response.tools_used
    assert "aggregate_amounts" in response.tools_used
    assert "list_contacts" not in response.tools_used


def test_citations_are_deduplicated() -> None:
    """When the same email appears as both a search hit and a match,
    it shows up once in citations."""
    llm = _fake_llm()
    response = GenerateResponse(llm)(
        question="x",
        intent=Intent.AGGREGATE_AMOUNTS,
        tool_results=_aggregate_results(),
    )
    assert len(response.citations) == len(set(response.citations))
    assert "acme-1024" in response.citations


# ─── Helpers ─────────────────────────────────────────────────────────


def _aggregate_results() -> ToolResults:
    """A canonical AGGREGATE_AMOUNTS ToolResults bundle for prompt tests."""
    matches = [
        AmountMatch(
            email_id="acme-1024",
            sender="billing@acme.com",
            amount=Decimal("1200.00"),
            currency="USD",
            raw_text="$1,200.00 USD",
            status="overdue",
        ),
        AmountMatch(
            email_id="acme-1031",
            sender="billing@acme.com",
            amount=Decimal("1500.00"),
            currency="USD",
            raw_text="$1,500.00 USD",
            status="pending",
        ),
        AmountMatch(
            email_id="acme-1042",
            sender="billing@acme.com",
            amount=Decimal("750.00"),
            currency="USD",
            raw_text="$750.00 USD",
            status="pending",
        ),
    ]
    groups = [
        Group(
            key="billing@acme.com",
            currency="USD",
            total=Decimal("3450.00"),
            count=3,
        ),
    ]
    return ToolResults(
        search_emails=[
            _email("acme-1024", subject="Invoice #1024", from_="billing@acme.com"),
            _email("acme-1031", subject="Invoice #1031", from_="billing@acme.com"),
            _email("acme-1042", subject="Invoice #1042", from_="billing@acme.com"),
        ],
        aggregate_amounts=AggregateResult(
            groups=groups,
            matches=matches,
            group_by="sender",
        ),
    )


# ─── Response shape ──────────────────────────────────────────────────


def test_agent_response_is_a_frozen_dataclass() -> None:
    import pytest

    r = AgentResponse(
        text="hi",
        intent=Intent.GENERAL_QA,
        tools_used=[],
        citations=[],
        model="m",
        total_tokens=0,
    )
    # frozen → can't reassign fields
    with pytest.raises(Exception):
        r.text = "no"  # type: ignore[misc]
