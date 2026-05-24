"""``select_tools`` — the second LangGraph node (AONE-402).

Maps a classified :class:`~aone.agent.intents.Intent` to the ordered
tuple of tool names the agent should invoke for that intent.
Deterministic; no LLM call.

The contract here is intentionally a list of **strings**, not function
references. ``execute_tools`` (AONE-408) resolves the names to the
actual callables. That late-binding keeps this module
import-circularity-free and makes the mapping trivial to inspect,
serialise, log, or override at runtime.

``get_thread`` (AONE-404) is not listed against any intent — it's an
internal helper that ``summarize_thread`` (AONE-407) reaches for when
it has a thread_id to expand. It will appear here later if and when a
user-facing intent needs it directly.
"""

from __future__ import annotations

from aone.agent.intents import Intent

_INTENT_TO_TOOLS: dict[Intent, tuple[str, ...]] = {
    Intent.AGGREGATE_AMOUNTS: ("search_emails", "aggregate_amounts"),
    Intent.SUMMARIZE:         ("search_emails", "summarize_thread"),
    Intent.FIND_EMAILS:       ("search_emails",),
    Intent.LIST_CONTACTS:     ("list_contacts",),
    Intent.GENERAL_QA:        ("search_emails",),
}

# Fail-fast invariant: every Intent must have a tool mapping. If a
# future Intent value is added without touching this table, the
# import itself errors out — better than a late-night surprise during
# `aone ask`.
_missing = set(Intent) - set(_INTENT_TO_TOOLS)
assert not _missing, f"select_tools: missing mapping for {_missing}"


def select_tools(intent: Intent) -> list[str]:
    """Return the ordered list of tool names to invoke for ``intent``.

    Callers receive a fresh ``list`` each call; mutating it does not
    contaminate later calls or the module-level mapping.

    Args:
        intent: a value from :class:`Intent`. Must be a known intent —
            unknown values raise :class:`KeyError`.

    Returns:
        Ordered list of tool names. ``execute_tools`` runs them in this
        order; parallelisable ones are detected downstream.
    """
    return list(_INTENT_TO_TOOLS[intent])
