"""Tests for ``aone.storage.vector``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aone.gmail.types import Email
from aone.storage.vector import (
    SCHEMA_VERSION,
    VectorIndex,
    VectorIndexError,
)


class FakeEmbedder:
    """Deterministic embedder for fast, network-free tests."""

    provider_name = "fake"
    model_name = "fake/v1"
    dims = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Cheap signature: [length, count(a), count(b), 1.0]
        return [
            [float(len(t)), float(t.count("a")), float(t.count("b")), 1.0]
            for t in texts
        ]


def _email(id_: str, body: str) -> Email:
    return Email(
        id=id_,
        thread_id=f"t-{id_}",
        from_="a@x.com",
        to=["b@y.com"],
        subject="subj",
        body_text=body,
        body_html=body,
        body_clean=body,
        snippet=body[:40],
        internal_date=0,
        labels=[],
    )


# ─── Basics ──────────────────────────────────────────────────────────


def test_empty_index_search_returns_empty_list() -> None:
    index = VectorIndex(FakeEmbedder())
    assert len(index) == 0
    assert index.search("anything") == []


def test_add_then_search_finds_the_email() -> None:
    index = VectorIndex(FakeEmbedder())
    index.add(_email("m1", "aaaa"))

    results = index.search("aaaa", k=5)

    assert len(results) == 1
    assert results[0][0] == "m1"
    assert results[0][1] >= 0.0


def test_add_many_then_search_returns_top_k_sorted_by_distance() -> None:
    index = VectorIndex(FakeEmbedder())
    index.add_many(
        [
            _email("aaa", "aaa"),
            _email("bbb", "bbb"),
            _email("aab", "aab"),
        ]
    )

    results = index.search("aaa", k=3)

    assert len(results) == 3
    # Closest to "aaa" should be "aaa" itself (distance 0).
    assert results[0][0] == "aaa"
    # Distances must be non-decreasing.
    distances = [r[1] for r in results]
    assert distances == sorted(distances)


def test_search_respects_k() -> None:
    index = VectorIndex(FakeEmbedder())
    index.add_many([_email(f"m{i}", "a" * i) for i in range(1, 6)])

    assert len(index.search("aaa", k=2)) == 2
    assert len(index.search("aaa", k=100)) == 5  # capped at index size


def test_add_many_empty_iterable_is_noop() -> None:
    index = VectorIndex(FakeEmbedder())
    index.add_many([])
    assert len(index) == 0


def test_search_with_k_zero_returns_empty() -> None:
    index = VectorIndex(FakeEmbedder())
    index.add(_email("m1", "x"))
    assert index.search("x", k=0) == []


# ─── Persistence ─────────────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "idx.faiss"
    embedder = FakeEmbedder()
    index = VectorIndex(embedder)
    index.add_many([_email("a", "apple"), _email("b", "banana"), _email("c", "cherry")])
    index.save(path)

    loaded = VectorIndex.load(embedder, path)

    assert len(loaded) == 3
    results = loaded.search("apple", k=1)
    assert results[0][0] == "a"


def test_save_writes_meta_file_with_provider_and_model(tmp_path: Path) -> None:
    path = tmp_path / "idx.faiss"
    index = VectorIndex(FakeEmbedder())
    index.add(_email("x", "hi"))
    index.save(path)

    meta_path = path.with_name(path.name + ".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["provider"] == "fake"
    assert meta["model"] == "fake/v1"
    assert meta["dims"] == 4
    assert meta["ids"] == ["x"]


def test_load_raises_on_provider_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "idx.faiss"
    VectorIndex(FakeEmbedder()).save(path)  # provider="fake"

    class OtherEmbedder(FakeEmbedder):
        provider_name = "different"

    with pytest.raises(VectorIndexError, match="Re-index required"):
        VectorIndex.load(OtherEmbedder(), path)


def test_load_raises_on_model_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "idx.faiss"
    VectorIndex(FakeEmbedder()).save(path)

    class OtherModel(FakeEmbedder):
        model_name = "fake/v2"

    with pytest.raises(VectorIndexError, match="Re-index required"):
        VectorIndex.load(OtherModel(), path)


def test_load_raises_on_schema_version_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "idx.faiss"
    index = VectorIndex(FakeEmbedder())
    index.add(_email("x", "hi"))
    index.save(path)

    # Corrupt the meta to simulate an upgrade.
    meta_path = path.with_name(path.name + ".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = SCHEMA_VERSION + 1
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(VectorIndexError, match="schema version"):
        VectorIndex.load(FakeEmbedder(), path)


def test_load_raises_filenotfound_when_no_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(FakeEmbedder(), tmp_path / "missing.faiss")


def test_load_or_create_returns_empty_when_no_file(tmp_path: Path) -> None:
    index = VectorIndex.load_or_create(FakeEmbedder(), tmp_path / "missing.faiss")
    assert len(index) == 0


def test_load_or_create_loads_when_file_exists(tmp_path: Path) -> None:
    path = tmp_path / "idx.faiss"
    seed = VectorIndex(FakeEmbedder())
    seed.add(_email("x", "hi"))
    seed.save(path)

    loaded = VectorIndex.load_or_create(FakeEmbedder(), path)
    assert len(loaded) == 1


def test_save_then_save_overwrites_atomically(tmp_path: Path) -> None:
    path = tmp_path / "idx.faiss"
    embedder = FakeEmbedder()

    first = VectorIndex(embedder)
    first.add(_email("a", "first"))
    first.save(path)

    second = VectorIndex(embedder)
    second.add_many([_email("b", "x"), _email("c", "y")])
    second.save(path)

    loaded = VectorIndex.load(embedder, path)
    assert len(loaded) == 2
    assert {r[0] for r in loaded.search("x", k=10)} == {"b", "c"}
    # No leftover .idx-*.faiss / .idx-*.json tempfiles.
    assert list(tmp_path.glob(".idx-*")) == []
