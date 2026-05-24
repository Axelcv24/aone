"""Tests for ``aone.agent.tools.contacts`` (AONE-405)."""

from __future__ import annotations

import pytest

from aone.agent.tools.contacts import Contact, ListContacts
from aone.gmail.types import Email
from aone.storage.cache import EmailCache


def _email(id_: str, *, from_: str, date: int) -> Email:
    return Email(
        id=id_,
        thread_id=f"t-{id_}",
        from_=from_,
        to=["axel@example.com"],
        subject="s",
        body_text="b",
        body_html="<p>b</p>",
        body_clean="b",
        snippet="b",
        internal_date=date,
        labels=["INBOX"],
    )


def _cache(*emails: Email) -> EmailCache:
    cache = EmailCache()
    cache.add_many(list(emails))
    return cache


# ─── Basic aggregation ───────────────────────────────────────────────


def test_empty_cache_returns_no_contacts() -> None:
    assert ListContacts(EmailCache())() == []


def test_each_unique_sender_becomes_one_contact() -> None:
    cache = _cache(
        _email("m1", from_="alice@x.com", date=100),
        _email("m2", from_="bob@y.com", date=200),
    )
    result = ListContacts(cache)()
    assert {c.email_address for c in result} == {"alice@x.com", "bob@y.com"}
    assert all(c.message_count == 1 for c in result)


def test_multiple_emails_from_same_address_coalesce_into_one_contact() -> None:
    cache = _cache(
        _email("m1", from_="Acme Corp <billing@acme.com>", date=100),
        _email("m2", from_="billing@acme.com", date=200),
        _email("m3", from_="Acme Billing <billing@acme.com>", date=300),
    )
    result = ListContacts(cache)()
    assert len(result) == 1
    contact = result[0]
    assert contact.email_address == "billing@acme.com"
    assert contact.message_count == 3
    assert contact.first_seen_ms == 100
    assert contact.last_seen_ms == 300


def test_contact_name_is_taken_from_the_most_recent_message_that_has_one() -> None:
    cache = _cache(
        _email("m1", from_="Old Name <a@x.com>", date=100),
        _email("m2", from_="a@x.com", date=200),
        _email("m3", from_="New Name <a@x.com>", date=300),
    )
    contact = ListContacts(cache)()[0]
    assert contact.name == "New Name"


def test_unparseable_from_headers_are_skipped() -> None:
    cache = _cache(
        _email("m1", from_="", date=100),
        _email("m2", from_="not an email", date=200),
        _email("m3", from_="real@x.com", date=300),
    )
    result = ListContacts(cache)()
    assert [c.email_address for c in result] == ["real@x.com"]


# ─── Sorting ─────────────────────────────────────────────────────────


def test_sort_by_count_descending_by_default() -> None:
    cache = _cache(
        _email("a1", from_="alice@x.com", date=100),
        _email("b1", from_="bob@y.com", date=200),
        _email("b2", from_="bob@y.com", date=300),
        _email("b3", from_="bob@y.com", date=400),
        _email("c1", from_="carl@z.com", date=500),
        _email("c2", from_="carl@z.com", date=600),
    )
    result = ListContacts(cache)()
    assert [c.email_address for c in result] == ["bob@y.com", "carl@z.com", "alice@x.com"]


def test_sort_by_recent() -> None:
    cache = _cache(
        _email("m1", from_="oldest@x.com", date=100),
        _email("m2", from_="middle@x.com", date=200),
        _email("m3", from_="newest@x.com", date=300),
    )
    result = ListContacts(cache)(sort_by="recent")
    assert [c.email_address for c in result] == [
        "newest@x.com",
        "middle@x.com",
        "oldest@x.com",
    ]


def test_sort_by_unknown_raises() -> None:
    with pytest.raises(ValueError, match="sort_by"):
        ListContacts(EmailCache())(sort_by="alphabetical")


def test_tie_break_by_email_address_for_determinism() -> None:
    """When counts tie, the order must be stable across runs."""
    cache = _cache(
        _email("a", from_="zach@x.com", date=100),
        _email("b", from_="alan@x.com", date=200),
    )
    result = ListContacts(cache)()
    # Both have count=1; tie-break is alphabetical on the address.
    assert [c.email_address for c in result] == ["alan@x.com", "zach@x.com"]


# ─── Filters ─────────────────────────────────────────────────────────


def test_min_messages_filter() -> None:
    cache = _cache(
        _email("a", from_="alice@x.com", date=100),
        _email("b1", from_="bob@y.com", date=200),
        _email("b2", from_="bob@y.com", date=300),
    )
    result = ListContacts(cache)(min_messages=2)
    assert [c.email_address for c in result] == ["bob@y.com"]


def test_last_seen_before_finds_inactive_contacts() -> None:
    """``last_seen_before_ms=300`` returns contacts last seen before 300."""
    cache = _cache(
        _email("a", from_="inactive@x.com", date=100),
        _email("b1", from_="active@y.com", date=200),
        _email("b2", from_="active@y.com", date=400),
    )
    result = ListContacts(cache)(last_seen_before_ms=300)
    assert [c.email_address for c in result] == ["inactive@x.com"]


def test_last_seen_before_excludes_contacts_with_a_message_at_or_after_cutoff() -> None:
    """Boundary: exactly at the cutoff is still active (strict <)."""
    cache = _cache(_email("m", from_="on-the-line@x.com", date=300))
    assert ListContacts(cache)(last_seen_before_ms=300) == []


def test_top_n_caps_the_result_list() -> None:
    cache = _cache(
        _email("a", from_="a@x.com", date=100),
        _email("b", from_="b@x.com", date=200),
        _email("c", from_="c@x.com", date=300),
        _email("d", from_="d@x.com", date=400),
    )
    result = ListContacts(cache)(top_n=2)
    assert len(result) == 2


# ─── Schema ──────────────────────────────────────────────────────────


def test_input_schema_exposes_all_kwargs_with_defaults() -> None:
    schema = ListContacts.input_schema()
    props = schema["properties"]
    assert "min_messages" in props
    assert props["min_messages"]["default"] == 1
    assert "last_seen_before_ms" in props
    assert "top_n" in props
    assert "sort_by" in props
    assert props["sort_by"]["enum"] == ["count", "recent"]


def test_contact_dataclass_is_hashable_and_immutable() -> None:
    c = Contact(
        email_address="a@x.com",
        name="Alice",
        message_count=2,
        first_seen_ms=100,
        last_seen_ms=200,
    )
    # frozen dataclass → hashable
    assert hash(c) == hash(c)
    with pytest.raises(Exception):
        c.message_count = 99  # type: ignore[misc]
