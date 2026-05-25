"""``search_emails`` tool (AONE-403).

Hybrid search over the local cache + FAISS index with optional
metadata filters (sender, date range, Gmail label). Returns the top-k
matching :class:`Email` instances.

Retrieval strategy:

1. **Literal identifier match** — if the query contains a distinctive
   token (invoice #, order #, alphanumeric ID), scan ``subject`` and
   ``body_clean`` for a direct substring hit. Pure semantic search
   misses these consistently because two emails containing
   ``#318900206`` are not necessarily semantically near each other.
2. **Vector search** — FAISS over the embedded ``body_clean``.
3. Filters (sender / date / label) apply after both stages and the
   union is deduplicated.

The literal stage always runs first so an exact identifier match
floats to the top, even when the embedder ranks the email far from
the query.

The class is callable so it composes naturally with the
``execute_tools`` registry (AONE-408): the registry holds one instance
per tool, and ``execute_tools`` invokes ``tool(**args)``.

Tool descriptions and ``input_schema`` are JSON-Schema-shaped so
exposing the tool over MCP (v2 backlog item AONE-908) is a thin
wrapping job, not a rewrite (ADR-004).
"""

from __future__ import annotations

import re
from typing import Any

from aone.gmail.addresses import extract_email_address
from aone.gmail.types import Email
from aone.storage.cache import EmailCache
from aone.storage.vector import VectorIndex

# Distinctive identifiers users mention in queries: invoice/order numbers
# (``#318900206``, ``1042``), structured IDs (``ABC-123``, ``Q3-2026``).
# ≥5 digits avoids matching short years or counts; the ``#`` prefix is
# optional because users write both ``#1042`` and ``1042``.
_IDENTIFIER_RE = re.compile(
    r"#\w{3,}|\b\d{5,}\b|\b[A-Z]{2,}-?\d{3,}\b",
    re.IGNORECASE,
)


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

        # Stage 1: literal identifier matches. If the user mentioned a
        # specific number or code that appears verbatim in any cached
        # email's subject/body, surface those first regardless of
        # semantic similarity. Floors the failure mode where the
        # embedder ranks the right email out of the candidate pool.
        literal_hits = _find_identifier_matches(query, self._cache)

        has_filter = any(
            f is not None
            for f in (sender, date_from_ms, date_to_ms, label)
        )

        # Pool sizing: when a structural filter is active (sender,
        # date, label), we ask FAISS for the ENTIRE index. Without
        # this, cross-language queries against a brand's emails fail:
        # for "muéstrame mi confirmación de Levi" (Spanish) against
        # an English marketing inbox, all 123 Levi emails sit outside
        # the top 1000 by FAISS distance. A cap of 1000 throws them
        # away before the filter ever sees them.
        #
        # The cost is bounded — FAISS IndexFlatL2 is linear in index
        # size; at 2000 emails this is sub-millisecond. The filter
        # then narrows to the relevant subset, and the loop break
        # at len(results) >= k still short-circuits cheaply.
        if has_filter:
            pool_size = len(self._index)
        else:
            pool_size = k

        hits = self._index.search(query, k=pool_size)

        # Merge: literal identifier hits first (in order found), then
        # FAISS hits, deduped by email id.
        results: list[Email] = []
        seen_ids: set[str] = set()
        for email in literal_hits:
            if email.id in seen_ids:
                continue
            if not _matches_filters(email, sender, date_from_ms, date_to_ms, label):
                continue
            results.append(email)
            seen_ids.add(email.id)
            if len(results) >= k:
                return results

        for email_id, _distance in hits:
            if email_id in seen_ids:
                continue
            email = self._cache.get(email_id)
            if email is None:
                continue
            if not _matches_filters(email, sender, date_from_ms, date_to_ms, label):
                continue
            results.append(email)
            seen_ids.add(email.id)
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


def _find_identifier_matches(query: str, cache: EmailCache) -> list[Email]:
    """Find cache emails whose subject/body contains an identifier from ``query``.

    Used by ``SearchEmails.__call__`` to compose hybrid retrieval:
    literal hits are returned first so an exact ``#318900206`` always
    floats above the FAISS ranking, no matter the semantic distance.
    """
    identifiers = {m.group(0).lstrip("#").lower() for m in _IDENTIFIER_RE.finditer(query)}
    if not identifiers:
        return []

    matches: list[Email] = []
    for email in cache:
        haystack = (email.subject + " " + email.body_clean).lower()
        if any(identifier in haystack for identifier in identifiers):
            matches.append(email)
    return matches


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
