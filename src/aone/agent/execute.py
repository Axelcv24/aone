"""``execute_tools`` — the third LangGraph node (AONE-408).

Takes the ordered list of tool names produced by ``select_tools`` and
runs them, plumbing the right arguments into each: the user's question
goes into ``search_emails``; the ``email_ids`` of those search hits
feed ``aggregate_amounts`` and ``summarize_thread``.

For v0 the dispatch is explicit and sequential. Every intent chain in
:mod:`aone.agent.select_tools` is linear (one or two tools, second
depends on first), so there's no parallel opportunity to chase yet —
the "run in parallel when independent" note in the ticket is reserved
for v1 intents that compose multiple branches.
"""

from __future__ import annotations

from dataclasses import dataclass

from aone.agent.tools.amounts import AggregateAmounts, AggregateResult
from aone.agent.tools.contacts import Contact, ListContacts
from aone.agent.tools.search import SearchEmails
from aone.agent.tools.summarize import SummarizeThread, ThreadSummary
from aone.agent.tools.thread import GetThread
from aone.gmail.types import Email
from aone.llm.client import LLMClient
from aone.storage.cache import EmailCache
from aone.storage.vector import VectorIndex

DEFAULT_SEARCH_K = 10


@dataclass(frozen=True)
class ToolResults:
    """Bag of every tool's output for a single user question.

    Fields are ``None`` for tools that didn't run on this turn.
    :func:`emails_for_response` is a convenience for
    :func:`generate_response` (AONE-409): pick the most natural set of
    emails to cite, falling back gracefully.
    """

    search_emails: list[Email] | None = None
    list_contacts: list[Contact] | None = None
    aggregate_amounts: AggregateResult | None = None
    summarize_thread: ThreadSummary | None = None
    get_thread: list[Email] | None = None

    @property
    def emails_for_response(self) -> list[Email]:
        """Best email set to ground the final answer in."""
        return list(self.search_emails or self.get_thread or [])


class ExecuteTools:
    """Tool dispatcher: name strings → real callables, in order."""

    def __init__(
        self,
        cache: EmailCache,
        index: VectorIndex,
        llm_client: LLMClient,
        *,
        search_k: int = DEFAULT_SEARCH_K,
    ) -> None:
        self._search_k = search_k
        self._search = SearchEmails(cache, index)
        self._get_thread = GetThread(cache)
        self._list_contacts = ListContacts(cache)
        self._aggregate = AggregateAmounts(cache, llm_client)
        self._summarize = SummarizeThread(cache, llm_client)

    def __call__(
        self,
        *,
        tool_names: list[str],
        question: str,
    ) -> ToolResults:
        """Run the named tools in order. Returns whatever each produced.

        Args:
            tool_names: ordered tool names from :func:`select_tools`.
            question: the user's original question. Used as the query
                for ``search_emails`` and threaded through to any tool
                that needs it.

        Returns:
            :class:`ToolResults`. Fields not produced on this turn are
            left ``None``.
        """
        accumulator = _Accumulator()

        for name in tool_names:
            if name == "search_emails":
                accumulator.search_emails = self._search(
                    query=question, k=self._search_k
                )
            elif name == "list_contacts":
                accumulator.list_contacts = self._list_contacts()
            elif name == "aggregate_amounts":
                accumulator.aggregate_amounts = self._aggregate(
                    email_ids=_ids_from(accumulator.search_emails),
                )
            elif name == "summarize_thread":
                ids = _ids_from(accumulator.search_emails)
                if ids:
                    accumulator.summarize_thread = self._summarize(
                        email_ids=ids
                    )
                else:
                    accumulator.summarize_thread = _empty_summary()
            elif name == "get_thread":
                # Currently no intent dispatches get_thread directly.
                # If a caller wires it up later it'll need a thread_id;
                # for now the dispatcher just skips it cleanly so a
                # mistaken mapping doesn't crash an entire question.
                continue
            else:
                raise ValueError(f"Unknown tool name: {name!r}")

        return accumulator.snapshot()


class _Accumulator:
    """Mutable scratchpad while running tools; frozen on the way out."""

    __slots__ = (
        "search_emails",
        "list_contacts",
        "aggregate_amounts",
        "summarize_thread",
        "get_thread",
    )

    def __init__(self) -> None:
        self.search_emails: list[Email] | None = None
        self.list_contacts: list[Contact] | None = None
        self.aggregate_amounts: AggregateResult | None = None
        self.summarize_thread: ThreadSummary | None = None
        self.get_thread: list[Email] | None = None

    def snapshot(self) -> ToolResults:
        return ToolResults(
            search_emails=self.search_emails,
            list_contacts=self.list_contacts,
            aggregate_amounts=self.aggregate_amounts,
            summarize_thread=self.summarize_thread,
            get_thread=self.get_thread,
        )


def _ids_from(emails: list[Email] | None) -> list[str]:
    return [e.id for e in emails] if emails else []


def _empty_summary() -> ThreadSummary:
    return ThreadSummary(
        text="",
        email_count=0,
        senders=[],
        earliest_date_ms=None,
        latest_date_ms=None,
        model="",
        total_tokens=0,
    )


# Convenience for callers (and tests) who want to introspect the
# tool universe this dispatcher knows about.
KNOWN_TOOLS: frozenset[str] = frozenset(
    {
        "search_emails",
        "get_thread",
        "list_contacts",
        "aggregate_amounts",
        "summarize_thread",
    }
)


def known_tools() -> frozenset[str]:
    """Names of tools the dispatcher can resolve."""
    return KNOWN_TOOLS


__all__: list[str] = ["ExecuteTools", "ToolResults", "KNOWN_TOOLS", "known_tools"]
