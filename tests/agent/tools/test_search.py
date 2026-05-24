"""Tests for ``aone.agent.tools.search`` (AONE-403)."""

from __future__ import annotations

from aone.agent.tools.search import SearchEmails
from aone.gmail.types import Email
from aone.storage.cache import EmailCache
from aone.storage.vector import VectorIndex


class _FakeEmbedder:
    """Deterministic embedder mirroring the one used in vector tests."""

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
    from_: str = "alice@x.com",
    date: int = 1_700_000_000_000,
    labels: list[str] | None = None,
) -> Email:
    return Email(
        id=id_,
        thread_id=f"t-{id_}",
        from_=from_,
        to=["axel@example.com"],
        subject=body[:30],
        body_text=body,
        body_html=f"<p>{body}</p>",
        body_clean=body,
        snippet=body[:80],
        internal_date=date,
        labels=labels or ["INBOX"],
    )


def _setup() -> tuple[EmailCache, VectorIndex]:
    cache = EmailCache()
    index = VectorIndex(_FakeEmbedder())

    emails = [
        _email(
            "m-apple",
            body="apple pie recipe with cinnamon",
            from_="alice@x.com",
            date=100,
            labels=["INBOX"],
        ),
        _email(
            "m-banana",
            body="banana banana banana",
            from_="bob@y.com",
            date=200,
            labels=["INBOX", "STARRED"],
        ),
        _email(
            "m-acme",
            body="acme invoice 1024 amount due",
            from_="Acme Billing <billing@acme.com>",
            date=300,
            labels=["INBOX"],
        ),
        _email(
            "m-acme2",
            body="acme follow-up about the invoice",
            from_="billing@acme.com",
            date=400,
            labels=["INBOX"],
        ),
        _email(
            "m-archive",
            body="archived old conversation",
            from_="carl@z.com",
            date=500,
            labels=["Label_Old"],
        ),
    ]
    cache.add_many(emails)
    index.add_many(emails)
    return cache, index


# ─── Basic search ────────────────────────────────────────────────────


def test_returns_at_most_k_results() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(query="anything", k=3)
    assert len(result) <= 3


def test_empty_query_returns_empty_list() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    assert tool(query="") == []
    assert tool(query="anything", k=0) == []


def test_returns_email_objects_from_cache() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(query="banana", k=5)
    assert all(isinstance(e, Email) for e in result)
    # banana is the closest match in our fake-embedder space
    assert any(e.id == "m-banana" for e in result)


def test_query_against_empty_index_returns_empty() -> None:
    cache = EmailCache()
    index = VectorIndex(_FakeEmbedder())
    assert SearchEmails(cache, index)(query="anything") == []


# ─── Sender filter ───────────────────────────────────────────────────


def test_sender_substring_match() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(query="invoice", sender="acme", k=5)
    assert {e.id for e in result} == {"m-acme", "m-acme2"}


def test_sender_exact_match_when_at_sign_present() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    # Substring "acme" would match both Acme emails, but the explicit
    # address only matches messages from exactly that mailbox.
    result = tool(query="invoice", sender="billing@acme.com", k=5)
    assert {e.id for e in result} == {"m-acme", "m-acme2"}


def test_sender_exact_match_is_case_insensitive() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(query="invoice", sender="BILLING@ACME.COM", k=5)
    assert {e.id for e in result} == {"m-acme", "m-acme2"}


def test_sender_filter_with_no_matches_returns_empty() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    assert tool(query="invoice", sender="not-real@nowhere.com") == []


# ─── Date filter ─────────────────────────────────────────────────────


def test_date_from_inclusive() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(query="anything", date_from_ms=300, k=10)
    # All emails with internal_date >= 300 pass the filter
    # (the fake embedder yields all of them in the candidate pool).
    assert {e.id for e in result} == {"m-acme", "m-acme2", "m-archive"}


def test_date_to_inclusive() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(query="invoice", date_to_ms=300, k=10)
    # Acme #1 is at date 300 (inclusive), Acme #2 at 400 is filtered.
    assert "m-acme" in {e.id for e in result}
    assert "m-acme2" not in {e.id for e in result}


def test_date_range_both_bounds() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(
        query="anything",
        date_from_ms=200,
        date_to_ms=400,
        k=10,
    )
    ids = {e.id for e in result}
    assert "m-apple" not in ids  # date=100
    assert "m-archive" not in ids  # date=500


# ─── Label filter ────────────────────────────────────────────────────


def test_label_filter_matches_exact_label() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(query="anything", label="STARRED", k=10)
    assert {e.id for e in result} == {"m-banana"}


def test_label_filter_excludes_emails_without_label() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(query="conversation", label="INBOX", k=10)
    # m-archive only has Label_Old, so it doesn't appear.
    assert "m-archive" not in {e.id for e in result}


def test_label_filter_can_match_custom_labels() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(query="anything", label="Label_Old", k=10)
    assert {e.id for e in result} == {"m-archive"}


# ─── Combined filters ────────────────────────────────────────────────


def test_combining_sender_and_date_range() -> None:
    cache, index = _setup()
    tool = SearchEmails(cache, index)

    result = tool(
        query="invoice",
        sender="acme",
        date_from_ms=350,
        k=10,
    )
    # Acme #2 (date=400) passes both filters; Acme #1 (date=300) fails date.
    assert {e.id for e in result} == {"m-acme2"}


# ─── Schema ──────────────────────────────────────────────────────────


def test_input_schema_lists_query_as_required() -> None:
    schema = SearchEmails.input_schema()
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert schema["required"] == ["query"]


def test_input_schema_covers_all_filter_kwargs() -> None:
    schema = SearchEmails.input_schema()
    props = schema["properties"]
    for field in ("query", "sender", "date_from_ms", "date_to_ms", "label", "k"):
        assert field in props, f"input_schema missing {field!r}"
