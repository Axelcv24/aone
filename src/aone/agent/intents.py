"""Intent vocabulary used by the agent graph.

The five intents are the result-side contract between
:func:`aone.agent.classify.classify_intent` and the ``select_tools``
node (AONE-402). Adding a sixth intent requires updating both
``classify`` (prompt + examples) and ``select_tools`` (mapping).
"""

from __future__ import annotations

from enum import StrEnum


class Intent(StrEnum):
    """Coarse-grained classification of a user question."""

    AGGREGATE_AMOUNTS = "aggregate_amounts"
    SUMMARIZE = "summarize"
    FIND_EMAILS = "find_emails"
    LIST_CONTACTS = "list_contacts"
    GENERAL_QA = "general_qa"

    @classmethod
    def parse(cls, raw: str) -> Intent:
        """Parse a model reply into an :class:`Intent`.

        Tolerates trailing punctuation, leading whitespace, casing
        variation, and stray words ("the intent is aggregate_amounts").
        Anything that doesn't resolve to a known intent falls back to
        :attr:`GENERAL_QA` — that bucket is the safe default for
        ambiguous queries.
        """
        if not raw:
            return cls.GENERAL_QA

        cleaned = raw.strip().lower()
        # Strip punctuation that small classifiers love to add.
        cleaned = cleaned.strip(" .,;:!?\"'`")

        # Try the whole reply first, then each whitespace-separated token.
        candidates = [cleaned] + cleaned.split()
        for candidate in candidates:
            stripped = candidate.strip(" .,;:!?\"'`")
            for intent in cls:
                if stripped == intent.value:
                    return intent

        return cls.GENERAL_QA
