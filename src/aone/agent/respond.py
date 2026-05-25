"""``generate_response`` — the fourth and final LangGraph node (AONE-409).

Consumes the ``ToolResults`` produced by ``execute_tools`` and asks
the configured generation model to compose the user-facing answer.

Design choices:

* **The numbers come from Python, the prose from the LLM.** The
  context block includes the pre-computed totals from
  ``aggregate_amounts``; the model is instructed to reuse them
  verbatim rather than re-sum. Same for contact rankings, thread
  summaries, and date ranges.

* **Per-intent context shape, single system prompt.** Each
  :class:`~aone.agent.tools.amounts.AggregateResult` /
  :class:`~aone.agent.tools.contacts.Contact` /
  :class:`~aone.agent.tools.summarize.ThreadSummary` /
  :class:`~aone.gmail.types.Email` knows how to render itself for the
  prompt; the system prompt itself stays intent-agnostic so the
  model decides framing.

* **No tools to cite ⇒ no LLM call.** When every field in
  ``ToolResults`` is ``None`` (the user's question matched nothing)
  the node returns a short polite "I couldn't find anything" string
  without spending tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from aone.agent.execute import ToolResults
from aone.agent.intents import Intent
from aone.agent.tools.amounts import AggregateResult
from aone.agent.tools.contacts import Contact
from aone.agent.tools.summarize import ThreadSummary
from aone.gmail.types import Email
from aone.llm.client import LLMClient, Role
from aone.observability.tracing import observe
from aone.storage.cache import EmailCache

DEFAULT_MAX_TOKENS = 600
DEFAULT_TEMPERATURE = 0.3
MAX_EMAILS_IN_CONTEXT = 8
# Per-email body excerpt sent to the LLM. Snippet (200 chars) is too
# little — totals, dates, status often live deeper in the body. 2000
# chars × 8 emails ≈ 4000 tokens of email content, comfortable under
# Llama 3.3's 128k window.
MAX_BODY_CHARS_PER_EMAIL = 2000

NO_RESULTS_RESPONSE = (
    "I couldn't find anything in your inbox that answers that question. "
    "Try `aone sync` to refresh, or rephrase the question — for example "
    "mention a sender, a subject, or a date range."
)


_SYSTEM_PROMPT = """\
You are Aone, a personal business assistant for the user's email inbox.

Answer the user's question using ONLY the structured tool output and email context provided below. Do NOT invent facts.

Rules:
- When the context already gives a precomputed total, balance, or count, REUSE it verbatim. Do not recompute.
- Cite emails by their Subject and Sender (or message ID in brackets) when making specific claims.
- Always show the currency next to monetary amounts (USD, EUR, MXN, …).
- If the provided context does not answer the question, say so plainly — do not speculate.
- Reply in the same language as the user's question.
- Be concise. Use bullet points when listing more than two items.
"""


@dataclass(frozen=True)
class AgentResponse:
    """The output of :class:`GenerateResponse` — what the user sees."""

    text: str
    intent: Intent
    tools_used: list[str]
    citations: list[str]  # email IDs included in the context block
    model: str
    total_tokens: int


class GenerateResponse:
    """LangGraph node: turn structured tool results into the final answer."""

    def __init__(
        self,
        llm_client: LLMClient,
        cache: EmailCache | None = None,
    ) -> None:
        self._llm = llm_client
        # Inbox catalog snapshot — included in every prompt so the
        # model knows *what exists* in the cache even when search hits
        # don't surface the relevant emails. Cheap (one walk over the
        # cache at agent-build time); huge quality win for questions
        # like "do I have emails from Levi?".
        self._inbox_catalog = _render_inbox_catalog(cache) if cache else ""

    @observe(name="generate_response")
    def __call__(
        self,
        *,
        question: str,
        intent: Intent,
        tool_results: ToolResults,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> AgentResponse:
        """Compose the user-facing answer.

        Args:
            question: the user's original natural-language question.
            intent: classification from ``classify_intent``. Threaded
                through to the response metadata; the prompt itself
                doesn't branch on it (the tool outputs already encode
                what to surface).
            tool_results: bundle from ``execute_tools``.
            max_tokens: cap on output tokens — Anthropic providers
                require it, others ignore it.

        Returns:
            :class:`AgentResponse`.
        """
        tools_used = _tools_used(tool_results)
        citations = _citations(tool_results)

        if not tools_used or not _has_any_data(tool_results):
            return AgentResponse(
                text=NO_RESULTS_RESPONSE,
                intent=intent,
                tools_used=tools_used,
                citations=citations,
                model="",
                total_tokens=0,
            )

        context = _build_context(intent, tool_results)
        system_prompt = _SYSTEM_PROMPT
        if self._inbox_catalog:
            system_prompt += "\n\n" + self._inbox_catalog
        result = self._llm.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\n{context}",
                },
            ],
            role=Role.GENERATION,
            max_tokens=max_tokens,
            temperature=DEFAULT_TEMPERATURE,
        )

        return AgentResponse(
            text=result.text,
            intent=intent,
            tools_used=tools_used,
            citations=citations,
            model=result.model,
            total_tokens=result.total_tokens,
        )


# ─── Tools-used / citations introspection ────────────────────────────


def _tools_used(results: ToolResults) -> list[str]:
    """Return the names of tools that produced data on this turn."""
    used: list[str] = []
    if results.search_emails is not None:
        used.append("search_emails")
    if results.list_contacts is not None:
        used.append("list_contacts")
    if results.aggregate_amounts is not None:
        used.append("aggregate_amounts")
    if results.summarize_thread is not None:
        used.append("summarize_thread")
    if results.get_thread is not None:
        used.append("get_thread")
    return used


def _citations(results: ToolResults) -> list[str]:
    """Email IDs included in the context block."""
    seen: set[str] = set()
    ordered: list[str] = []
    for email in results.emails_for_response:
        if email.id not in seen:
            ordered.append(email.id)
            seen.add(email.id)
    if results.aggregate_amounts:
        for match in results.aggregate_amounts.matches:
            if match.email_id not in seen:
                ordered.append(match.email_id)
                seen.add(match.email_id)
    return ordered


def _has_any_data(results: ToolResults) -> bool:
    """True iff at least one tool produced non-empty output."""
    return any(
        (
            results.search_emails,
            results.list_contacts,
            results.aggregate_amounts and results.aggregate_amounts.matches,
            results.summarize_thread and results.summarize_thread.email_count,
            results.get_thread,
        )
    )


# ─── Context rendering ───────────────────────────────────────────────


def _build_context(intent: Intent, results: ToolResults) -> str:
    """Stitch together the tool outputs into a model-readable context block."""
    sections: list[str] = [f"Detected intent: {intent.value}"]

    if results.aggregate_amounts:
        sections.append(_render_aggregate(results.aggregate_amounts))

    if results.list_contacts:
        sections.append(_render_contacts(results.list_contacts))

    if results.summarize_thread and results.summarize_thread.email_count:
        sections.append(_render_summary(results.summarize_thread))

    emails = results.emails_for_response
    if emails:
        sections.append(_render_emails(emails))

    return "\n\n".join(sections)


def _render_aggregate(agg: AggregateResult) -> str:
    lines = [f"=== Aggregate amounts (grouped by {agg.group_by}) ==="]
    for g in agg.groups:
        lines.append(
            f"- {g.key}: {g.currency} {_fmt(g.total)} "
            f"(from {g.count} amount{'' if g.count == 1 else 's'})"
        )

    grand = agg.grand_total_by_currency
    if grand:
        totals = ", ".join(
            f"{cur} {_fmt(total)}" for cur, total in sorted(grand.items())
        )
        lines.append(f"Grand total by currency: {totals}")

    if agg.matches:
        lines.append("\nUnderlying amounts:")
        for m in agg.matches:
            status = f" ({m.status})" if m.status else ""
            lines.append(
                f"  • [{m.email_id}] {m.sender}: "
                f"{m.currency} {_fmt(m.amount)} — \"{m.raw_text}\"{status}"
            )
    return "\n".join(lines)


def _render_contacts(contacts: list[Contact]) -> str:
    lines = ["=== Contacts ==="]
    for c in contacts:
        name = f"{c.name} <" if c.name else ""
        closer = ">" if c.name else ""
        lines.append(
            f"- {name}{c.email_address}{closer}: "
            f"{c.message_count} message{'' if c.message_count == 1 else 's'}, "
            f"last seen {_fmt_date(c.last_seen_ms)}"
        )
    return "\n".join(lines)


def _render_summary(summary: ThreadSummary) -> str:
    lines = [
        f"=== Thread summary ({summary.email_count} emails) ===",
        summary.text,
    ]
    if summary.senders:
        lines.append(f"Participants: {', '.join(summary.senders)}")
    return "\n".join(lines)


def _render_emails(emails: list[Email]) -> str:
    capped = emails[:MAX_EMAILS_IN_CONTEXT]
    lines = [f"=== Email context ({len(capped)} of {len(emails)} shown) ==="]
    for e in capped:
        # Prefer the cleaned body; fall back to snippet for emails
        # whose body normalization stripped everything.
        body = (e.body_clean or e.snippet or "").strip()
        truncated = len(body) > MAX_BODY_CHARS_PER_EMAIL
        excerpt = body[:MAX_BODY_CHARS_PER_EMAIL]
        if truncated:
            excerpt += "\n[…body truncated…]"
        lines.append(
            f"\n[Email {e.id}]\n"
            f"From: {e.from_}\n"
            f"Date: {_fmt_date(e.internal_date)}\n"
            f"Subject: {e.subject}\n"
            f"Body:\n{excerpt}"
        )
    return "\n".join(lines)


# ─── Small formatting helpers ────────────────────────────────────────


def _render_inbox_catalog(cache: EmailCache, top_n: int = 10) -> str:
    """Render a brief snapshot of what the cache contains.

    Goes into the system prompt so the model knows the inbox's shape
    (total size, date range, top senders) even when ``search_emails``
    didn't surface the relevant messages. Important for honest answers
    on "do I have emails from X?" — without this the model defaults to
    "no" when search comes back empty, even though the sender is in
    the cache.
    """
    stats = cache.stats(top_n=top_n)
    if stats.email_count == 0:
        return ""

    lines = [
        "── Inbox catalog (what's in the local cache) ──",
        f"Total messages: {stats.email_count}",
    ]
    if stats.earliest_internal_date and stats.latest_internal_date:
        from datetime import datetime, timezone

        earliest = datetime.fromtimestamp(
            stats.earliest_internal_date / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        latest = datetime.fromtimestamp(
            stats.latest_internal_date / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        lines.append(f"Date range: {earliest} → {latest}")
    if stats.top_senders:
        lines.append(f"Top {len(stats.top_senders)} senders:")
        for address, count in stats.top_senders:
            lines.append(f"  - {address}: {count} message(s)")
    lines.append(
        "Use this catalog to give honest answers about whether a sender "
        "is present in the inbox, even if the email context below "
        "doesn't include their messages this turn."
    )
    return "\n".join(lines)


def _fmt(amount: Decimal) -> str:
    """Format a Decimal as ``1,234.56`` (two decimal places, thousands grouped)."""
    return f"{amount:,.2f}"


def _fmt_date(ms: int | None) -> str:
    if not ms:
        return "(unknown)"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
