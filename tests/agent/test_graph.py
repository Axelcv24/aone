"""Tests for ``aone.agent.graph`` (AONE-410)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

from aone.agent.graph import AgentState, ask, build_agent
from aone.agent.intents import Intent
from aone.agent.respond import AgentResponse
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


def _llm_router(replies: dict[str, str]) -> MagicMock:
    """Make an LLMClient mock whose reply depends on a substring of the prompt.

    ``replies`` keys are matched against the user message; first hit
    wins. Default fallback is "general_qa" (the classifier needs to
    return *something*).
    """
    client = MagicMock(spec=LLMClient)

    def _complete(messages: list[dict[str, Any]], **kwargs: Any) -> CompletionResult:
        text = "general_qa"
        if messages:
            user_content = " ".join(m.get("content", "") for m in messages)
            for marker, reply in replies.items():
                if marker in user_content:
                    text = reply
                    break
        return CompletionResult(
            text=text,
            model="groq/llama-3.3-70b-versatile",
            prompt_tokens=20,
            completion_tokens=10,
        )

    client.complete.side_effect = _complete
    return client


def _setup(*emails: Email) -> tuple[EmailCache, VectorIndex]:
    cache = EmailCache()
    index = VectorIndex(_FakeEmbedder())
    if emails:
        cache.add_many(list(emails))
        index.add_many(list(emails))
    return cache, index


# ─── Shape ───────────────────────────────────────────────────────────


def test_build_agent_returns_invokable_graph() -> None:
    cache, index = _setup()
    llm = _llm_router({"general_qa": "general_qa", "Question": "There's nothing in your inbox."})

    agent = build_agent(cache, index, llm)

    assert hasattr(agent, "invoke")


def test_invoke_returns_state_with_response_populated() -> None:
    cache, index = _setup(_email("m", body="hi there"))
    llm = _llm_router(
        {
            # classify_intent looks at the question; "anything" → general_qa
            "anything": "general_qa",
            # generate_response sees "Question:" in the user message
            "Question:": "Here's what I found.",
        }
    )

    agent = build_agent(cache, index, llm)
    state: AgentState = agent.invoke({"question": "anything"})  # type: ignore[assignment]

    assert state["intent"] == Intent.GENERAL_QA
    assert state["tool_names"] == ["search_emails"]
    assert state["tool_results"] is not None
    assert isinstance(state["response"], AgentResponse)


def test_ask_helper_returns_response_directly() -> None:
    cache, index = _setup(_email("m", body="hi"))
    llm = _llm_router({"Question:": "Got it.", "anything": "general_qa"})
    agent = build_agent(cache, index, llm)

    response = ask(agent, "anything")

    assert isinstance(response, AgentResponse)
    assert response.text == "Got it."


# ─── Node ordering & state flow ──────────────────────────────────────


def test_classify_runs_before_select_runs_before_execute_runs_before_respond() -> None:
    """We rely on later nodes seeing earlier nodes' output; verify the
    sequence by reading the state at the end."""
    cache, index = _setup(_email("m", body="hi"))
    llm = _llm_router({"Question:": "ok", "anything": "find_emails"})
    agent = build_agent(cache, index, llm)

    state: AgentState = agent.invoke({"question": "anything"})  # type: ignore[assignment]

    # If classify ran first, intent is set:
    assert state["intent"] == Intent.FIND_EMAILS
    # If select ran on intent, tool_names matches the FIND_EMAILS mapping:
    assert state["tool_names"] == ["search_emails"]
    # If execute ran on tool_names, tool_results populated:
    assert state["tool_results"].search_emails is not None
    # If respond ran on tool_results, response is the AgentResponse:
    assert state["response"].intent == Intent.FIND_EMAILS


# ─── Each intent path runs end-to-end ────────────────────────────────


def test_aggregate_amounts_intent_runs_full_chain() -> None:
    cache, index = _setup(
        _email(
            "a", body="Invoice for $1,200.00 USD pending", from_="billing@acme.com",
        ),
        _email(
            "b", body="Invoice for $1,500.00 USD pending", from_="billing@acme.com",
        ),
    )
    llm = _llm_router(
        {
            "cuánto": "aggregate_amounts",
            "Question:": "Acme owes USD 2,700.00 across 2 invoices.",
        }
    )
    agent = build_agent(cache, index, llm)

    state: AgentState = agent.invoke({"question": "¿cuánto me debe Acme?"})  # type: ignore[assignment]

    assert state["intent"] == Intent.AGGREGATE_AMOUNTS
    assert state["tool_names"] == ["search_emails", "aggregate_amounts"]
    agg = state["tool_results"].aggregate_amounts
    assert agg is not None
    assert agg.grand_total_by_currency["USD"] == Decimal("2700.00")
    assert "aggregate_amounts" in state["response"].tools_used


def test_list_contacts_intent_runs_without_search() -> None:
    cache, index = _setup(
        _email("a", body="x", from_="alice@x.com"),
        _email("b", body="y", from_="alice@x.com"),
        _email("c", body="z", from_="bob@y.com"),
    )
    llm = _llm_router({"contactos": "list_contacts", "Question:": "Top: Alice (2)."})
    agent = build_agent(cache, index, llm)

    state: AgentState = agent.invoke({"question": "lista mis contactos"})  # type: ignore[assignment]

    assert state["intent"] == Intent.LIST_CONTACTS
    assert state["tool_names"] == ["list_contacts"]
    # search_emails should NOT have run for list_contacts intent:
    assert state["tool_results"].search_emails is None
    assert state["tool_results"].list_contacts is not None


def test_summarize_intent_runs_search_then_summarize() -> None:
    cache, index = _setup(
        _email("a", body="first message about the project"),
        _email("b", body="second message about the project"),
    )
    llm = _llm_router(
        {
            "resume": "summarize",
            # Both the summarize_thread call and generate_response call
            # ask the same LLM; the matching above maps the summarize-
            # thread prompt (which contains "Summarise the following")
            # to a fixed reply, and the final response prompt (which
            # contains "Question:") to another.
            "Summarise the following": "• point one\n• point two",
            "Question:": "Here's a recap of two emails.",
        }
    )
    agent = build_agent(cache, index, llm)

    state: AgentState = agent.invoke({"question": "resume estas conversaciones"})  # type: ignore[assignment]

    assert state["intent"] == Intent.SUMMARIZE
    assert state["tool_names"] == ["search_emails", "summarize_thread"]
    summary = state["tool_results"].summarize_thread
    assert summary is not None
    assert summary.email_count > 0


def test_no_data_intent_returns_canned_no_results() -> None:
    """Empty cache + index → search returns [], respond short-circuits
    with the canned no-results string and skips the final LLM call."""
    cache, index = _setup()
    llm = _llm_router({"anything": "find_emails"})
    agent = build_agent(cache, index, llm)

    response = ask(agent, "anything")

    # tools_used has search_emails (it ran), but text is the canned one
    # because the result was empty.
    assert "search_emails" in response.tools_used
    assert response.model == ""
    assert "couldn't find" in response.text.lower()
