"""In-memory email cache with on-disk pickle persistence.

The cache is the source of truth between Gmail syncs. It's a dict keyed
by Gmail message ID; ``save`` serialises the entire mapping to a pickle
file atomically. The on-disk format is versioned — bumping
:data:`SCHEMA_VERSION` is a hard break and callers must re-sync.

In v0 there is exactly one cache file per user (see ADR-001 for why a
real database is deliberately deferred to v1).
"""

from __future__ import annotations

import os
import pickle
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from aone.gmail.addresses import extract_email_address
from aone.gmail.types import Email

SCHEMA_VERSION = 1
DEFAULT_CACHE_PATH = Path.home() / ".aone" / "cache.pkl"


class CacheSchemaError(RuntimeError):
    """Raised when the on-disk cache has an unrecognised schema version."""


@dataclass(frozen=True)
class CacheStats:
    """Summary metrics over a cache. Returned by :meth:`EmailCache.stats`."""

    email_count: int
    earliest_internal_date: int | None
    latest_internal_date: int | None
    top_senders: list[tuple[str, int]]
    disk_size_bytes: int | None


class EmailCache:
    """Dict-backed cache of Gmail messages.

    Instances behave like a small read-mostly mapping: ``len()``,
    ``in``, ``iter()``, and ``.get()`` work as expected. Writes go
    through :meth:`add` / :meth:`add_many`. Nothing happens on disk
    until you call :meth:`save`.
    """

    def __init__(self, emails: dict[str, Email] | None = None) -> None:
        self._emails: dict[str, Email] = dict(emails) if emails else {}
        # Set by save() / load() / load_or_create() so stats() can report
        # the file size without the caller having to track the path.
        self._path: Path | None = None

    # ─── Mapping-style read API ────────────────────────────────────

    def __len__(self) -> int:
        return len(self._emails)

    def __contains__(self, message_id: object) -> bool:
        return message_id in self._emails

    def __iter__(self) -> Iterator[Email]:
        return iter(self._emails.values())

    def get(self, message_id: str) -> Email | None:
        return self._emails.get(message_id)

    # ─── Write API ─────────────────────────────────────────────────

    def add(self, email: Email) -> None:
        """Insert or replace a single message."""
        self._emails[email.id] = email

    def add_many(self, emails: Iterable[Email]) -> int:
        """Bulk-insert. Returns the number of *new* IDs (updates don't count)."""
        added = 0
        for email in emails:
            if email.id not in self._emails:
                added += 1
            self._emails[email.id] = email
        return added

    # ─── Persistence ───────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        """Write the cache to ``path`` atomically.

        Strategy: dump to a sibling tempfile in the same directory, then
        :func:`os.replace` it over the destination. On POSIX this is
        atomic — a reader either sees the old version or the new one,
        never a half-written file.
        """
        if path is None:
            path = self._path or DEFAULT_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"schema_version": SCHEMA_VERSION, "emails": self._emails}

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=".cache-",
            suffix=".pkl",
            delete=False,
        ) as tmp:
            pickle.dump(payload, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path = Path(tmp.name)

        try:
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        self._path = path

    @classmethod
    def load(cls, path: Path | None = None) -> EmailCache:
        """Load a cache from ``path``.

        Raises:
            FileNotFoundError: if the cache file does not exist.
            CacheSchemaError: if the on-disk schema version does not
                match :data:`SCHEMA_VERSION`. Treat this as "data is
                from an older Aone — re-run ``aone sync``".
        """
        path = path or DEFAULT_CACHE_PATH
        with path.open("rb") as f:
            payload = pickle.load(f)

        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise CacheSchemaError(
                f"Cache at {path} has schema version {version!r}; "
                f"expected {SCHEMA_VERSION}. Re-sync to rebuild."
            )
        instance = cls(emails=payload["emails"])
        instance._path = path
        return instance

    @classmethod
    def load_or_create(cls, path: Path | None = None) -> EmailCache:
        """Load if the file exists, otherwise return an empty cache."""
        path = path or DEFAULT_CACHE_PATH
        if path.exists():
            return cls.load(path)
        instance = cls()
        instance._path = path
        return instance

    # ─── Stats ─────────────────────────────────────────────────────

    def stats(self, *, top_n: int = 5) -> CacheStats:
        """Compute summary statistics over the cached emails.

        ``disk_size_bytes`` is taken from the path the cache was most
        recently saved to or loaded from. If the cache has never touched
        disk, the size is ``None``.

        Args:
            top_n: number of senders to include in ``top_senders``.

        Returns:
            A :class:`CacheStats` snapshot.
        """
        emails = list(self._emails.values())
        if not emails:
            return CacheStats(
                email_count=0,
                earliest_internal_date=None,
                latest_internal_date=None,
                top_senders=[],
                disk_size_bytes=_size_of(self._path),
            )

        dates = [e.internal_date for e in emails if e.internal_date]
        sender_counts: Counter[str] = Counter()
        for e in emails:
            address = extract_email_address(e.from_)
            if address:
                sender_counts[address] += 1

        return CacheStats(
            email_count=len(emails),
            earliest_internal_date=min(dates) if dates else None,
            latest_internal_date=max(dates) if dates else None,
            top_senders=sender_counts.most_common(top_n),
            disk_size_bytes=_size_of(self._path),
        )


def _size_of(path: Path | None) -> int | None:
    """File size in bytes, or ``None`` if the path is unset / missing."""
    if path is None:
        return None
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None
