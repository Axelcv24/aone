"""Domain types for Gmail messages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Email:
    """Parsed Gmail message, ready to be cached, embedded, or queried.

    Designed to be MIME-agnostic at the boundary: the parser collapses the
    payload tree into a single ``body_text`` (plain text, preferred for
    embeddings) and ``body_html`` (kept around for display or downstream
    inspection).
    """

    id: str
    thread_id: str
    from_: str
    to: list[str]
    subject: str
    body_text: str
    body_html: str
    snippet: str
    internal_date: int
    labels: list[str]
