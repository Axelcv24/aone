"""``get_thread`` tool (AONE-404).

Returns every email in a Gmail thread (conversation), ordered oldest
first. Used when the user follows up on a search result — e.g. after
``search_emails`` returns a single match, the agent calls
``get_thread(thread_id=hit.thread_id)`` to give downstream tools (and
the final response writer) the full conversation context.

No network: all data comes from :class:`EmailCache`. Sub-millisecond
for typical inboxes.
"""

from __future__ import annotations

from typing import Any

from aone.gmail.types import Email
from aone.storage.cache import EmailCache


class GetThread:
    """Tool: fetch a full conversation by thread ID from the local cache."""

    NAME = "get_thread"
    DESCRIPTION = (
        "Return every email in a Gmail thread (conversation), ordered "
        "from oldest to newest. Use this when the user asks about a "
        "specific conversation or wants the full context around an "
        "email returned by search_emails."
    )

    def __init__(self, cache: EmailCache) -> None:
        self._cache = cache

    def __call__(self, *, thread_id: str) -> list[Email]:
        """Return the thread's emails sorted by ``internal_date`` ascending.

        Args:
            thread_id: the Gmail thread ID. Available on every
                :class:`Email` as ``email.thread_id``.

        Returns:
            Ordered list of :class:`Email`. Empty list when the thread
            ID is unknown or there are no cached messages in it. The
            empty case is intentionally non-fatal — callers expect to
            handle "thread not in cache, please sync" gracefully.
        """
        if not thread_id:
            return []

        emails = [email for email in self._cache if email.thread_id == thread_id]
        emails.sort(key=lambda e: e.internal_date)
        return emails

    @classmethod
    def input_schema(cls) -> dict[str, Any]:
        """JSON-Schema shape for MCP / OpenAI function calling."""
        return {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": (
                        "Gmail thread ID. Use the value of "
                        "``Email.thread_id`` from a previous search "
                        "result."
                    ),
                },
            },
            "required": ["thread_id"],
        }
