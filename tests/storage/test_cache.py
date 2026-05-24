"""Tests for ``aone.storage.cache``."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from aone.gmail.types import Email
from aone.storage.cache import (
    SCHEMA_VERSION,
    CacheSchemaError,
    EmailCache,
)


def _email(id_: str, *, subject: str = "subject", body: str = "body") -> Email:
    return Email(
        id=id_,
        thread_id=f"t-{id_}",
        from_="a@x.com",
        to=["b@y.com"],
        subject=subject,
        body_text=body,
        body_html=f"<p>{body}</p>",
        body_clean=body,
        snippet=body[:40],
        internal_date=1_700_000_000_000,
        labels=["INBOX"],
    )


# ─── Mapping API ─────────────────────────────────────────────────────


def test_empty_cache_is_empty() -> None:
    cache = EmailCache()
    assert len(cache) == 0
    assert list(cache) == []
    assert cache.get("nope") is None
    assert "nope" not in cache


def test_add_then_get_and_contains() -> None:
    cache = EmailCache()
    email = _email("m1", subject="hello")

    cache.add(email)

    assert len(cache) == 1
    assert "m1" in cache
    assert cache.get("m1") == email
    assert list(cache) == [email]


def test_add_replaces_existing_id() -> None:
    cache = EmailCache()
    cache.add(_email("m1", subject="v1"))
    cache.add(_email("m1", subject="v2"))

    assert len(cache) == 1
    assert cache.get("m1") is not None
    assert cache.get("m1").subject == "v2"


def test_add_many_counts_new_ids_only() -> None:
    cache = EmailCache()
    cache.add(_email("m1"))

    new_count = cache.add_many(
        [
            _email("m1", subject="updated"),  # not new
            _email("m2"),  # new
            _email("m3"),  # new
        ]
    )

    assert new_count == 2
    assert len(cache) == 3


def test_constructor_copies_input_dict() -> None:
    """Mutating the input dict after construction must not affect the cache."""
    backing = {"m1": _email("m1")}
    cache = EmailCache(emails=backing)
    backing["m2"] = _email("m2")
    assert len(cache) == 1


# ─── Persistence ─────────────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cache.pkl"
    cache = EmailCache()
    cache.add(_email("a"))
    cache.add(_email("b", subject="second"))

    cache.save(path)
    loaded = EmailCache.load(path)

    assert len(loaded) == 2
    assert loaded.get("a") is not None
    assert loaded.get("b").subject == "second"


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deeper" / "cache.pkl"
    EmailCache().save(nested)
    assert nested.exists()


def test_save_is_atomic_no_leftover_tmpfiles(tmp_path: Path) -> None:
    """After save, only the destination file (and gitkeeps etc.) should remain."""
    path = tmp_path / "cache.pkl"
    cache = EmailCache()
    cache.add(_email("a"))
    cache.save(path)

    # No leftover .cache-*.pkl tempfiles
    leftovers = list(tmp_path.glob(".cache-*"))
    assert leftovers == []
    assert path.exists()


def test_save_overwrites_existing_atomically(tmp_path: Path) -> None:
    path = tmp_path / "cache.pkl"

    EmailCache(emails={"a": _email("a", subject="old")}).save(path)
    EmailCache(emails={"a": _email("a", subject="new")}).save(path)

    loaded = EmailCache.load(path)
    assert loaded.get("a").subject == "new"


def test_load_raises_when_schema_version_mismatches(tmp_path: Path) -> None:
    path = tmp_path / "cache.pkl"
    with path.open("wb") as f:
        pickle.dump({"schema_version": SCHEMA_VERSION + 99, "emails": {}}, f)

    with pytest.raises(CacheSchemaError, match="schema version"):
        EmailCache.load(path)


def test_load_or_create_returns_empty_when_file_absent(tmp_path: Path) -> None:
    cache = EmailCache.load_or_create(tmp_path / "missing.pkl")
    assert len(cache) == 0


def test_load_or_create_loads_when_file_exists(tmp_path: Path) -> None:
    path = tmp_path / "cache.pkl"
    EmailCache(emails={"x": _email("x")}).save(path)

    cache = EmailCache.load_or_create(path)
    assert len(cache) == 1
    assert "x" in cache


def test_load_raises_filenotfound_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EmailCache.load(tmp_path / "nope.pkl")
