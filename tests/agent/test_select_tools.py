"""Tests for ``aone.agent.select_tools`` (AONE-402)."""

from __future__ import annotations

from aone.agent.intents import Intent
from aone.agent.select_tools import _INTENT_TO_TOOLS, select_tools


# ─── Per-intent mappings ─────────────────────────────────────────────


def test_aggregate_amounts_runs_search_then_aggregate() -> None:
    assert select_tools(Intent.AGGREGATE_AMOUNTS) == [
        "search_emails",
        "aggregate_amounts",
    ]


def test_summarize_runs_search_then_summarize_thread() -> None:
    assert select_tools(Intent.SUMMARIZE) == [
        "search_emails",
        "summarize_thread",
    ]


def test_find_emails_runs_search_only() -> None:
    assert select_tools(Intent.FIND_EMAILS) == ["search_emails"]


def test_list_contacts_runs_list_contacts_only() -> None:
    """Contact analytics work off the cache, no search step needed."""
    assert select_tools(Intent.LIST_CONTACTS) == ["list_contacts"]


def test_general_qa_falls_back_to_search() -> None:
    """Ambiguous queries get the most general tool: semantic search."""
    assert select_tools(Intent.GENERAL_QA) == ["search_emails"]


# ─── Properties ──────────────────────────────────────────────────────


def test_returned_list_is_a_fresh_copy() -> None:
    """Caller mutations must not leak into subsequent calls."""
    first = select_tools(Intent.FIND_EMAILS)
    first.append("aggregate_amounts")

    second = select_tools(Intent.FIND_EMAILS)
    assert second == ["search_emails"]


def test_tool_order_is_stable_across_calls() -> None:
    a = select_tools(Intent.AGGREGATE_AMOUNTS)
    b = select_tools(Intent.AGGREGATE_AMOUNTS)
    assert a == b


def test_every_intent_resolves_to_a_nonempty_list_of_strings() -> None:
    """Smoke test for the module-level coverage invariant."""
    for intent in Intent:
        result = select_tools(intent)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(name, str) and name for name in result)


def test_referenced_tools_are_within_the_v0_universe() -> None:
    """Catch typos / drift: every referenced name must match a Sprint 4 tool ticket."""
    v0_tools = {
        "search_emails",      # AONE-403
        "get_thread",         # AONE-404
        "list_contacts",      # AONE-405
        "aggregate_amounts",  # AONE-406
        "summarize_thread",   # AONE-407
    }
    referenced = {name for tools in _INTENT_TO_TOOLS.values() for name in tools}
    unknown = referenced - v0_tools
    assert not unknown, f"INTENT_TO_TOOLS references unknown tools: {unknown}"


def test_search_emails_is_used_by_most_intents() -> None:
    """Sanity check the central role of semantic search."""
    intents_using_search = [
        intent
        for intent, tools in _INTENT_TO_TOOLS.items()
        if "search_emails" in tools
    ]
    # 4 of 5 intents should rely on semantic search.
    assert len(intents_using_search) >= 4
