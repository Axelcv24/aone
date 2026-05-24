"""``list_contacts`` tool (AONE-405).

Aggregates the cache by sender address and returns one
:class:`Contact` per unique correspondent, with message count,
first/last-seen timestamps, and the most recent display name observed.

Supports the two recurring questions:

* "Who are my top contacts?" → ``sort_by="count"`` and ``top_n``
* "Who hasn't replied in 30 days?" → ``last_seen_before_ms=<cutoff>``

No network, no LLM, no I/O. Pure cache walk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aone.gmail.addresses import parse_from
from aone.storage.cache import EmailCache


@dataclass(frozen=True)
class Contact:
    """A unique correspondent aggregated from the cache."""

    email_address: str
    name: str
    message_count: int
    first_seen_ms: int
    last_seen_ms: int


class _Aggregate:
    """Mutable accumulator used while walking the cache."""

    __slots__ = ("name", "count", "first_ms", "last_ms")

    def __init__(self, name: str, when: int) -> None:
        self.name = name
        self.count = 0
        self.first_ms = when
        self.last_ms = when


SORT_BY_COUNT = "count"
SORT_BY_RECENT = "recent"
_VALID_SORT = frozenset({SORT_BY_COUNT, SORT_BY_RECENT})


class ListContacts:
    """Tool: list unique correspondents in the cache, with filters."""

    NAME = "list_contacts"
    DESCRIPTION = (
        "Return unique correspondents from the user's email cache, "
        "each with their message count, first-seen and last-seen "
        "dates, and most recent display name. Supports filtering by "
        "minimum message count and inactivity threshold; sortable by "
        "message count or most-recent-first."
    )

    def __init__(self, cache: EmailCache) -> None:
        self._cache = cache

    def __call__(
        self,
        *,
        min_messages: int = 1,
        last_seen_before_ms: int | None = None,
        top_n: int | None = None,
        sort_by: str = SORT_BY_COUNT,
    ) -> list[Contact]:
        """Return contacts matching the filters, sorted as requested.

        Args:
            min_messages: include only contacts with at least this many
                messages in the cache.
            last_seen_before_ms: include only contacts whose most recent
                message arrived strictly before this timestamp. Useful
                for "inactive since N days" questions — caller computes
                the cutoff as ``now_ms - delta``.
            top_n: cap the result at this many contacts (after sorting).
                ``None`` means no cap.
            sort_by: ``"count"`` for most-frequent-first (the default),
                ``"recent"`` for most-recently-active first.

        Returns:
            List of :class:`Contact`, deterministically ordered.
        """
        if sort_by not in _VALID_SORT:
            raise ValueError(
                f"Unknown sort_by={sort_by!r}. Expected one of "
                f"{sorted(_VALID_SORT)}."
            )

        aggregates = self._aggregate_by_address()

        contacts: list[Contact] = []
        for address, agg in aggregates.items():
            if agg.count < min_messages:
                continue
            if last_seen_before_ms is not None and agg.last_ms >= last_seen_before_ms:
                continue
            contacts.append(
                Contact(
                    email_address=address,
                    name=agg.name,
                    message_count=agg.count,
                    first_seen_ms=agg.first_ms,
                    last_seen_ms=agg.last_ms,
                )
            )

        # Stable secondary sort by email address so ties have a
        # predictable order — keeps tests deterministic.
        if sort_by == SORT_BY_COUNT:
            contacts.sort(key=lambda c: (-c.message_count, c.email_address))
        else:
            contacts.sort(key=lambda c: (-c.last_seen_ms, c.email_address))

        if top_n is not None:
            contacts = contacts[:top_n]

        return contacts

    def _aggregate_by_address(self) -> dict[str, _Aggregate]:
        by_address: dict[str, _Aggregate] = {}
        for email in self._cache:
            name, address = parse_from(email.from_)
            if not address:
                continue
            agg = by_address.get(address)
            when = email.internal_date
            if agg is None:
                by_address[address] = agg = _Aggregate(name=name, when=when)
            agg.count += 1
            if when < agg.first_ms:
                agg.first_ms = when
            if when > agg.last_ms:
                agg.last_ms = when
                # Most recent display name wins — handy when a contact
                # eventually starts sending with a fuller signature.
                if name:
                    agg.name = name
        return by_address

    @classmethod
    def input_schema(cls) -> dict[str, Any]:
        """JSON-Schema shape for MCP / OpenAI function calling."""
        return {
            "type": "object",
            "properties": {
                "min_messages": {
                    "type": "integer",
                    "description": (
                        "Include only contacts with at least this many "
                        "messages in the cache."
                    ),
                    "default": 1,
                    "minimum": 1,
                },
                "last_seen_before_ms": {
                    "type": "integer",
                    "description": (
                        "Include only contacts whose most recent message "
                        "arrived strictly before this timestamp "
                        "(milliseconds since epoch). Use to find "
                        "inactive contacts."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": "Cap the result list at this many contacts.",
                    "minimum": 1,
                },
                "sort_by": {
                    "type": "string",
                    "enum": [SORT_BY_COUNT, SORT_BY_RECENT],
                    "description": (
                        "'count' for most-frequent-first (default), "
                        "'recent' for most-recently-active first."
                    ),
                    "default": SORT_BY_COUNT,
                },
            },
        }
