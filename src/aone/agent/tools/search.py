"""``search_emails`` tool (AONE-403).

Semantic search over the FAISS index with optional metadata filters
(sender, date range, Gmail label). Returns the top-k matching
:class:`Email` instances, ranked by L2 distance against the query
embedding.

The class is callable so it composes naturally with the
``execute_tools`` registry (AONE-408): the registry holds one instance
per tool, and ``execute_tools`` invokes ``tool(**args)``.

Tool descriptions and ``input_schema`` are JSON-Schema-shaped so
exposing the tool over MCP (v2 backlog item AONE-908) is a thin
wrapping job, not a rewrite (ADR-004).
"""

from __future__ import annotations

from typing import Any

from aone.gmail.addresses import extract_email_address
from aone.gmail.types import Email
from aone.storage.cache import EmailCache
from aone.storage.vector import VectorIndex


class SearchEmails:
    """Tool: semantic search over the cached emails with metadata filters."""

    NAME = "search_emails"
    DESCRIPTION = (
        "Semantic search over the user's email cache. Returns the top-k "
        "emails most relevant to the natural-language query, optionally "
        "narrowed by sender (substring or exact email match), an "
        "internal-date range (milliseconds since epoch), or a Gmail "
        "label such as INBOX or STARRED."
    )

    def __init__(self, cache: EmailCache, index: VectorIndex) -> None:
        self._cache = cache
        self._index = index

    def __call__(
        self,
        *,
        query: str,
        sender: str | None = None,
        date_from_ms: int | None = None,
        date_to_ms: int | None = None,
        label: str | None = None,
        k: int = 10,
    ) -> list[Email]:
        """Search emails matching ``query`` and the optional filters.

        Args:
            query: natural-language search string. The tool embeds this
                via the index's embedder and compares against every
                cached email's body.
            sender: substring match against the bare sender address by
                default (e.g. ``"acme"`` matches ``"billing@acme.com"``).
                Strings containing ``@`` are matched exactly.
            date_from_ms: minimum ``internal_date`` (inclusive). Useful
                for "after Apr 1, 2026" style filters.
            date_to_ms: maximum ``internal_date`` (inclusive).
            label: Gmail label (e.g. ``"INBOX"``, ``"STARRED"``,
                ``"Label_123"``). Match is exact.
            k: maximum number of emails to return.

        Returns:
            List of :class:`Email`, length ``≤ k``, ordered by semantic
            similarity (most similar first). Empty list when the query
            is empty or no emails clear the filters.
        """
        if not query or k <= 0:
            return []

        # If filters are present, pull a larger candidate pool first so
        # the filters can prune without starving the result set. The
        # multiplier (×5, capped at 50) is a heuristic; for realistic
        # v0 inbox sizes (~500 messages) it's a no-op.
        has_filter = any(
            f is not None
            for f in (sender, date_from_ms, date_to_ms, label)
        )
        pool_size = max(k * 5, 50) if has_filter else k

        hits = self._index.search(query, k=pool_size)

        results: list[Email] = []
        for email_id, _distance in hits:
            email = self._cache.get(email_id)
            if email is None:
                continue
            if not _matches_filters(email, sender, date_from_ms, date_to_ms, label):
                continue
            results.append(email)
            if len(results) >= k:
                break

        return results

    @classmethod
    def input_schema(cls) -> dict[str, Any]:
        """JSON-Schema shape suitable for MCP / OpenAI function calling."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language query. The full email body is "
                        "compared semantically; the query does not need "
                        "to use Gmail search syntax."
                    ),
                },
                "sender": {
                    "type": "string",
                    "description": (
                        "Filter by sender. Substring (e.g. 'acme') unless "
                        "the value contains '@', then exact match."
                    ),
                },
                "date_from_ms": {
                    "type": "integer",
                    "description": "Earliest internal_date (ms since epoch, inclusive).",
                },
                "date_to_ms": {
                    "type": "integer",
                    "description": "Latest internal_date (ms since epoch, inclusive).",
                },
                "label": {
                    "type": "string",
                    "description": "Gmail label (INBOX, STARRED, Label_123, …). Exact match.",
                },
                "k": {
                    "type": "integer",
                    "description": "Maximum results to return.",
                    "default": 10,
                    "minimum": 1,
                },
            },
            "required": ["query"],
        }


def _matches_filters(
    email: Email,
    sender: str | None,
    date_from_ms: int | None,
    date_to_ms: int | None,
    label: str | None,
) -> bool:
    if sender is not None:
        address = extract_email_address(email.from_)
        if "@" in sender:
            if address != sender.lower():
                return False
        else:
            if sender.lower() not in address:
                return False

    if date_from_ms is not None and email.internal_date < date_from_ms:
        return False
    if date_to_ms is not None and email.internal_date > date_to_ms:
        return False

    if label is not None and label not in email.labels:
        return False

    return True
