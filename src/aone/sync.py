"""Sync logic for ``aone sync`` (AONE-501).

Extracted from the CLI so it stays testable. The CLI handles file
I/O, auth, and progress reporting; this module is pure: given an
authenticated Gmail service plus a cache and index, it pulls the most
recent messages from the server and merges new ones in.

Re-running ``perform_sync`` is idempotent — already-cached IDs are
skipped, so the second invocation only fetches the delta.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aone.gmail.client import get_message, list_messages
from aone.gmail.types import Email
from aone.storage.cache import EmailCache
from aone.storage.vector import VectorIndex


@dataclass(frozen=True)
class SyncResult:
    """Outcome of a sync run, for CLI summary and tests."""

    listed: int            # message IDs returned by Gmail
    already_cached: int    # of those, how many we already had
    fetched: int           # how many we successfully downloaded
    failed: int            # how many fetches errored out
    cache_size: int        # total cache size after the sync
    index_size: int        # total index size after the sync


def perform_sync(
    *,
    service: Any,
    cache: EmailCache,
    index: VectorIndex,
    limit: int,
    query: str | None = None,
    on_fetch: Callable[[Email], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> SyncResult:
    """Pull up to ``limit`` recent messages and merge them into cache + index.

    Args:
        service: authenticated Gmail v1 service (from ``get_service``).
        cache: target :class:`EmailCache`. Mutated in place.
        index: target :class:`VectorIndex`. Mutated in place.
        limit: maximum number of message IDs to consider.
        query: optional Gmail search query (web-UI syntax).
        on_fetch: optional callback invoked once per successfully
            fetched email — used by the CLI to drive a progress bar.
        on_error: optional callback ``(message_id, exception)`` for
            fetches that raised. The sync keeps going on per-message
            errors so a single corrupt message doesn't abort the run.

    Returns:
        A :class:`SyncResult` with counts for the CLI summary.
    """
    ids = list_messages(service, limit=limit, query=query)
    new_ids = [mid for mid in ids if mid not in cache]

    fetched: list[Email] = []
    failed = 0
    for mid in new_ids:
        try:
            email = get_message(service, mid)
        except Exception as exc:  # noqa: BLE001 — keep syncing on per-message errors
            failed += 1
            if on_error is not None:
                on_error(mid, exc)
            continue
        fetched.append(email)
        if on_fetch is not None:
            on_fetch(email)

    if fetched:
        cache.add_many(fetched)
        index.add_many(fetched)

    return SyncResult(
        listed=len(ids),
        already_cached=len(ids) - len(new_ids),
        fetched=len(fetched),
        failed=failed,
        cache_size=len(cache),
        index_size=len(index),
    )
