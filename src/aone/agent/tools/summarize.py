"""``summarize_thread`` tool (AONE-407).

Summarizes a Gmail thread (or any ordered list of emails) into a short
bullet-style brief. Used by the ``summarize`` intent path to produce
the response the user actually reads.

The tool is intentionally thin: it stitches the email bodies into a
clean context block, attaches a system prompt, and asks the configured
generation model (default ``groq/llama-3.3-70b-versatile``). The LLM
does the actual summarisation work — we just give it good ingredients
(``body_clean`` already has signatures and quoted replies removed by
AONE-204).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from aone.agent.tools.thread import GetThread
from aone.gmail.addresses import extract_email_address
from aone.gmail.types import Email
from aone.llm.client import LLMClient, Role
from aone.storage.cache import EmailCache

DEFAULT_MAX_BULLETS = 5
DEFAULT_MAX_TOKENS = 600

_SYSTEM_PROMPT = """\
You are summarising an email thread or set of related emails.

Produce a brief summary that:
- Captures the main topic and outcome.
- Highlights key decisions, action items, pending questions, and amounts.
- Mentions who said what when it matters (use names or email addresses).
- Uses bullet points; up to {max_bullets} bullets.
- Replies in the same language as the emails.
- Does NOT invent details. If a fact is not in the emails, omit it."""


@dataclass(frozen=True)
class ThreadSummary:
    """Output of :class:`SummarizeThread`."""

    text: str
    email_count: int
    senders: list[str]
    earliest_date_ms: int | None
    latest_date_ms: int | None
    model: str
    total_tokens: int


class SummarizeThread:
    """Tool: summarise a Gmail thread or a set of related emails."""

    NAME = "summarize_thread"
    DESCRIPTION = (
        "Summarise a Gmail thread (by thread_id) or any explicit list of "
        "emails (by email_ids) into a short bullet-style brief. Uses the "
        "configured generation model. Use after search_emails or when "
        "the user asks 'what happened in this conversation'."
    )

    def __init__(self, cache: EmailCache, llm_client: LLMClient) -> None:
        self._cache = cache
        self._llm = llm_client
        # Reuse get_thread so thread expansion is consistent with the
        # rest of the agent.
        self._get_thread = GetThread(cache)

    def __call__(
        self,
        *,
        thread_id: str | None = None,
        email_ids: list[str] | None = None,
        max_bullets: int = DEFAULT_MAX_BULLETS,
    ) -> ThreadSummary:
        """Summarise either a thread or an explicit list of emails.

        Exactly one of ``thread_id`` or ``email_ids`` must be supplied.

        Args:
            thread_id: Gmail thread ID. ``get_thread`` is used to
                expand it into the full conversation, ordered oldest
                first.
            email_ids: explicit list of message IDs. Useful when the
                caller has just run ``search_emails`` and wants those
                hits summarised regardless of thread membership.
            max_bullets: cap on bullet count the model should produce.
                Hint only — the model is asked, not enforced.

        Returns:
            :class:`ThreadSummary` with the summary text and metadata.
            When the resolved email list is empty (e.g. unknown
            thread_id), returns an empty summary without calling the
            LLM.

        Raises:
            ValueError: when neither or both selectors are provided.
        """
        if thread_id is not None and email_ids is not None:
            raise ValueError(
                "Pass exactly one of thread_id or email_ids, not both."
            )
        if thread_id is None and email_ids is None:
            raise ValueError(
                "Provide either thread_id or email_ids."
            )

        emails = self._resolve_emails(thread_id, email_ids)
        if not emails:
            return _empty_summary()

        context = _build_context_block(emails)
        result = self._llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT.format(max_bullets=max_bullets),
                },
                {"role": "user", "content": context},
            ],
            role=Role.GENERATION,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.3,
        )

        senders = sorted(
            {
                addr
                for addr in (extract_email_address(e.from_) for e in emails)
                if addr
            }
        )
        dates = [e.internal_date for e in emails if e.internal_date]

        return ThreadSummary(
            text=result.text,
            email_count=len(emails),
            senders=senders,
            earliest_date_ms=min(dates) if dates else None,
            latest_date_ms=max(dates) if dates else None,
            model=result.model,
            total_tokens=result.total_tokens,
        )

    def _resolve_emails(
        self,
        thread_id: str | None,
        email_ids: list[str] | None,
    ) -> list[Email]:
        if thread_id is not None:
            return self._get_thread(thread_id=thread_id)
        assert email_ids is not None  # narrowed by the caller's validation
        out: list[Email] = []
        for eid in email_ids:
            email = self._cache.get(eid)
            if email is not None:
                out.append(email)
        # Keep the same oldest-first ordering get_thread uses, so the
        # context block reads chronologically regardless of input order.
        out.sort(key=lambda e: e.internal_date)
        return out

    @classmethod
    def input_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "Gmail thread ID — expanded via get_thread.",
                },
                "email_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Explicit list of email IDs to summarise. "
                        "Mutually exclusive with thread_id."
                    ),
                },
                "max_bullets": {
                    "type": "integer",
                    "default": DEFAULT_MAX_BULLETS,
                    "minimum": 1,
                    "description": "Soft cap on bullets the model should emit.",
                },
            },
            "oneOf": [
                {"required": ["thread_id"]},
                {"required": ["email_ids"]},
            ],
        }


# ─── Helpers ─────────────────────────────────────────────────────────


def _build_context_block(emails: list[Email]) -> str:
    """Render emails into a model-readable transcript."""
    parts = [
        f"[Email {i + 1} of {len(emails)}]\n"
        f"FROM:    {email.from_}\n"
        f"DATE:    {_format_date(email.internal_date)}\n"
        f"SUBJECT: {email.subject}\n"
        f"BODY:\n{email.body_clean}"
        for i, email in enumerate(emails)
    ]
    transcript = "\n\n---\n\n".join(parts)
    return (
        f"Summarise the following {len(emails)} email(s):\n\n"
        f"{transcript}"
    )


def _format_date(ms: int) -> str:
    if not ms:
        return "(unknown)"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


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
