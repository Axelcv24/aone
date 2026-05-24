"""Tests for ``aone.agent.tools.thread`` (AONE-404)."""

from __future__ import annotations

from aone.agent.tools.thread import GetThread
from aone.gmail.types import Email
from aone.storage.cache import EmailCache


def _email(id_: str, *, thread_id: str, date: int) -> Email:
    return Email(
        id=id_,
        thread_id=thread_id,
        from_="a@x.com",
        to=["b@y.com"],
        subject=f"msg {id_}",
        body_text=f"body of {id_}",
        body_html=f"<p>body of {id_}</p>",
        body_clean=f"body of {id_}",
        snippet=f"body of {id_}",
        internal_date=date,
        labels=["INBOX"],
    )


def _populated_cache() -> EmailCache:
    cache = EmailCache()
    cache.add_many(
        [
            # Thread T1: three messages, on purpose inserted out of order.
            _email("m3", thread_id="T1", date=300),
            _email("m1", thread_id="T1", date=100),
            _email("m2", thread_id="T1", date=200),
            # Thread T2: lone message.
            _email("only", thread_id="T2", date=999),
        ]
    )
    return cache


# ─── Basics ──────────────────────────────────────────────────────────


def test_returns_full_thread_sorted_oldest_first() -> None:
    tool = GetThread(_populated_cache())
    result = tool(thread_id="T1")

    assert [e.id for e in result] == ["m1", "m2", "m3"]


def test_returns_singleton_thread() -> None:
    tool = GetThread(_populated_cache())
    result = tool(thread_id="T2")

    assert len(result) == 1
    assert result[0].id == "only"


def test_returns_empty_list_for_unknown_thread() -> None:
    tool = GetThread(_populated_cache())
    assert tool(thread_id="does-not-exist") == []


def test_returns_empty_list_when_cache_is_empty() -> None:
    tool = GetThread(EmailCache())
    assert tool(thread_id="anything") == []


def test_returns_empty_list_for_empty_thread_id() -> None:
    """Defensive: empty string shouldn't match every email accidentally."""
    tool = GetThread(_populated_cache())
    assert tool(thread_id="") == []


def test_returns_email_objects() -> None:
    tool = GetThread(_populated_cache())
    result = tool(thread_id="T1")
    assert all(isinstance(e, Email) for e in result)


# ─── Properties ──────────────────────────────────────────────────────


def test_does_not_mutate_the_cache() -> None:
    cache = _populated_cache()
    GetThread(cache)(thread_id="T1")
    GetThread(cache)(thread_id="T1")
    # Cache still contains all 4 emails after multiple calls.
    assert len(cache) == 4


def test_sort_order_is_stable_across_calls() -> None:
    tool = GetThread(_populated_cache())
    a = [e.id for e in tool(thread_id="T1")]
    b = [e.id for e in tool(thread_id="T1")]
    assert a == b


# ─── Schema ──────────────────────────────────────────────────────────


def test_input_schema_requires_thread_id() -> None:
    schema = GetThread.input_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["thread_id"]
    assert "thread_id" in schema["properties"]
    assert schema["properties"]["thread_id"]["type"] == "string"
