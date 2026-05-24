"""FAISS in-memory vector index over Gmail messages (ADR-002, ADR-005).

The index is keyed positionally — FAISS stores vectors in insertion order
and returns row indices on search. We maintain a parallel ``ids`` list
mapping each position back to a Gmail message ID, then the caller looks
the actual :class:`Email` up in :class:`~aone.storage.cache.EmailCache`.

Persistence is two files at the same path:

* ``index.faiss``         — the binary FAISS index
* ``index.faiss.meta.json`` — schema version, provider/model used to
                              build it, dims, and the positional id list

The meta file is the authority on whether a loaded index is compatible
with the current embedder. Switching ``AONE_EMBEDDING_PROVIDER`` from
``local`` to ``litellm`` (or changing the model) invalidates the
embeddings, so ``load()`` checks both fields and refuses to load with a
clear error if they differ.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np

from aone.gmail.types import Email
from aone.llm.embeddings import Embedder

SCHEMA_VERSION = 1
DEFAULT_INDEX_PATH = Path.home() / ".aone" / "index.faiss"
_META_SUFFIX = ".meta.json"


class VectorIndexError(RuntimeError):
    """Raised when the on-disk index is incompatible with the current embedder."""


class VectorIndex:
    """L2 FAISS index over the ``body_clean`` of each :class:`Email`."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._index: faiss.Index | None = None
        self._ids: list[str] = []

    def __len__(self) -> int:
        return len(self._ids)

    # ─── Writes ────────────────────────────────────────────────────

    def add(self, email: Email) -> None:
        """Embed and add a single email."""
        self.add_many([email])

    def add_many(self, emails: Iterable[Email]) -> None:
        """Embed and add a batch in one FAISS call (much faster than per-email)."""
        batch = list(emails)
        if not batch:
            return

        vectors = self._embedder.embed([e.body_clean for e in batch])
        arr = np.asarray(vectors, dtype="float32")
        self._ensure_index(arr.shape[1])
        assert self._index is not None
        self._index.add(arr)
        self._ids.extend(e.id for e in batch)

    def _ensure_index(self, dims: int) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatL2(dims)
        elif self._index.d != dims:
            raise VectorIndexError(
                f"Dimensionality mismatch: index is {self._index.d}-dim, "
                f"new vectors are {dims}-dim."
            )

    # ─── Search ────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return ``[(email_id, distance), …]`` sorted nearest-first.

        Distance is L2 (lower = more similar). Returns an empty list if
        the index is empty.
        """
        if self._index is None or len(self._ids) == 0 or k <= 0:
            return []

        query_vec = self._embedder.embed([query])
        arr = np.asarray(query_vec, dtype="float32")
        distances, indices = self._index.search(arr, min(k, len(self._ids)))
        return [
            (self._ids[idx], float(distances[0][n]))
            for n, idx in enumerate(indices[0])
            if idx >= 0
        ]

    # ─── Persistence ───────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        """Write index + meta atomically to ``path`` and ``path + .meta.json``."""
        path = path or DEFAULT_INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = self._meta_path(path)

        meta = {
            "schema_version": SCHEMA_VERSION,
            "provider": self._embedder.provider_name,
            "model": self._embedder.model_name,
            "dims": self._index.d if self._index is not None else 0,
            "ids": list(self._ids),
        }

        # Atomic write for both files: tempfile + os.replace.
        tmp_idx = _atomic_write_bytes(
            path,
            lambda target: faiss.write_index(self._index, str(target))
            if self._index is not None
            else target.write_bytes(b""),
        )
        tmp_meta = _atomic_write_bytes(
            meta_path,
            lambda target: target.write_text(json.dumps(meta, indent=2)),
        )
        # Clean up intermediates (handled inside _atomic_write_bytes)
        del tmp_idx, tmp_meta

    @classmethod
    def load(cls, embedder: Embedder, path: Path | None = None) -> VectorIndex:
        """Load an index and validate it matches the given embedder.

        Raises:
            FileNotFoundError: when ``path`` or the meta file is missing.
            VectorIndexError: when schema version, provider, model, or
                dims disagree with the current embedder.
        """
        path = path or DEFAULT_INDEX_PATH
        meta_path = cls._meta_path(path)

        if not path.exists():
            raise FileNotFoundError(f"Index file missing: {path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Index meta file missing: {meta_path}")

        meta = json.loads(meta_path.read_text())
        cls._validate_meta(meta, embedder)

        instance = cls(embedder)
        instance._ids = list(meta["ids"])
        if meta["dims"] > 0:
            instance._index = faiss.read_index(str(path))
            if instance._index.d != meta["dims"]:
                raise VectorIndexError(
                    f"On-disk index has {instance._index.d} dims, "
                    f"meta says {meta['dims']}. Index file corrupt; re-build."
                )
        return instance

    @classmethod
    def load_or_create(cls, embedder: Embedder, path: Path | None = None) -> VectorIndex:
        """Load if ``path`` exists, otherwise return an empty index."""
        path = path or DEFAULT_INDEX_PATH
        if path.exists() and cls._meta_path(path).exists():
            return cls.load(embedder, path)
        return cls(embedder)

    # ─── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _meta_path(path: Path) -> Path:
        return path.with_name(path.name + _META_SUFFIX)

    @staticmethod
    def _validate_meta(meta: dict, embedder: Embedder) -> None:
        version = meta.get("schema_version")
        if version != SCHEMA_VERSION:
            raise VectorIndexError(
                f"Index has schema version {version!r}; expected "
                f"{SCHEMA_VERSION}. Re-build to upgrade."
            )

        if meta.get("provider") != embedder.provider_name:
            raise VectorIndexError(
                f"Index was built with provider {meta.get('provider')!r}, "
                f"current embedder is {embedder.provider_name!r}. "
                f"Re-index required: switching providers invalidates the "
                f"embedding space."
            )

        if meta.get("model") != embedder.model_name:
            raise VectorIndexError(
                f"Index was built with model {meta.get('model')!r}, "
                f"current embedder uses {embedder.model_name!r}. "
                f"Re-index required."
            )


def _atomic_write_bytes(path: Path, writer) -> Path:
    """Write a file atomically: writer is invoked on a tempfile, then renamed."""
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=".idx-", suffix=path.suffix or ".tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path
