"""Tests for ``EmailCache.stats`` (AONE-303)."""

from __future__ import annotations

from pathlib import Path

from aone.gmail.types import Email
from aone.storage.cache import EmailCache


def _email(
    id_: str,
    *,
    from_: str = "alice@x.com",
    internal_date: int = 1_700_000_000_000,
) -> Email:
    return Email(
        id=id_,
        thread_id=f"t-{id_}",
        from_=from_,
        to=["b@y.com"],
        subject="s",
        body_text="b",
        body_html="<p>b</p>",
        body_clean="b",
        snippet="b",
        internal_date=internal_date,
        labels=["INBOX"],
    )


# ─── Empty cache ─────────────────────────────────────────────────────


def test_stats_empty_cache_returns_zeros_and_nones() -> None:
    stats = EmailCache().stats()
    assert stats.email_count == 0
    assert stats.earliest_internal_date is None
    assert stats.latest_internal_date is None
    assert stats.top_senders == []
    assert stats.disk_size_bytes is None


# ─── Counts and date range ───────────────────────────────────────────


def test_stats_email_count_matches_cache_size() -> None:
    cache = EmailCache()
    cache.add_many(_email(f"m{i}") for i in range(7))
    assert cache.stats().email_count == 7


def test_stats_min_and_max_internal_date() -> None:
    cache = EmailCache()
    cache.add_many(
        [
            _email("a", internal_date=1_700_000_000_000),
            _email("b", internal_date=1_700_000_999_999),
            _email("c", internal_date=1_600_000_000_000),
        ]
    )
    stats = cache.stats()
    assert stats.earliest_internal_date == 1_600_000_000_000
    assert stats.latest_internal_date == 1_700_000_999_999


def test_stats_ignores_zero_internal_date_entries() -> None:
    """Some test/fake messages have internal_date=0; they should not skew the range."""
    cache = EmailCache()
    cache.add_many(
        [
            _email("a", internal_date=1_700_000_000_000),
            _email("b", internal_date=0),
        ]
    )
    stats = cache.stats()
    assert stats.earliest_internal_date == 1_700_000_000_000
    assert stats.latest_internal_date == 1_700_000_000_000


# ─── Top senders ─────────────────────────────────────────────────────


def test_top_senders_groups_by_email_address_ignoring_display_name() -> None:
    """`Alice <a@x.com>` and `a@x.com` count as the same sender."""
    cache = EmailCache()
    cache.add_many(
        [
            _email("m1", from_="Alice <a@x.com>"),
            _email("m2", from_="a@x.com"),
            _email("m3", from_="ALICE <a@x.com>"),
            _email("m4", from_="Bob <b@y.com>"),
        ]
    )
    stats = cache.stats()
    assert stats.top_senders[0] == ("a@x.com", 3)
    assert stats.top_senders[1] == ("b@y.com", 1)


def test_top_senders_respects_top_n() -> None:
    cache = EmailCache()
    cache.add_many(
        [
            _email("a1", from_="acme@x.com"),
            _email("a2", from_="acme@x.com"),
            _email("a3", from_="acme@x.com"),
            _email("b1", from_="beta@y.com"),
            _email("b2", from_="beta@y.com"),
            _email("c1", from_="gamma@z.com"),
        ]
    )
    top2 = cache.stats(top_n=2).top_senders
    assert len(top2) == 2
    assert top2[0] == ("acme@x.com", 3)
    assert top2[1] == ("beta@y.com", 2)


def test_top_senders_default_is_five() -> None:
    cache = EmailCache()
    cache.add_many(_email(f"m{i}", from_=f"sender{i}@x.com") for i in range(8))
    assert len(cache.stats().top_senders) == 5


def test_top_senders_skips_emails_with_unparseable_from() -> None:
    cache = EmailCache()
    cache.add_many(
        [
            _email("m1", from_=""),
            _email("m2", from_="just a name with no email"),
            _email("m3", from_="real@x.com"),
        ]
    )
    senders = cache.stats().top_senders
    assert senders == [("real@x.com", 1)]


# ─── Disk size ───────────────────────────────────────────────────────


def test_disk_size_is_none_before_save() -> None:
    cache = EmailCache()
    cache.add(_email("m1"))
    assert cache.stats().disk_size_bytes is None


def test_disk_size_is_set_after_save(tmp_path: Path) -> None:
    cache = EmailCache()
    cache.add(_email("m1"))
    cache.save(tmp_path / "cache.pkl")

    size = cache.stats().disk_size_bytes
    assert size is not None
    assert size > 0


def test_disk_size_is_set_after_load(tmp_path: Path) -> None:
    path = tmp_path / "cache.pkl"
    seed = EmailCache()
    seed.add(_email("m1"))
    seed.save(path)

    loaded = EmailCache.load(path)
    size = loaded.stats().disk_size_bytes
    assert size is not None
    assert size > 0


def test_disk_size_is_set_after_load_or_create_empty(tmp_path: Path) -> None:
    """Even when the file doesn't exist yet, the path is remembered;
    once we save() the size becomes available without specifying it again."""
    cache = EmailCache.load_or_create(tmp_path / "fresh.pkl")
    assert cache.stats().disk_size_bytes is None  # file doesn't exist yet
    cache.add(_email("m1"))
    cache.save()
    assert cache.stats().disk_size_bytes is not None
