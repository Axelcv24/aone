"""Tests for ``aone.gmail.normalize``.

Each test uses a realistic body sourced from common mail clients (Gmail,
Apple Mail, Outlook, mobile signatures). Inputs are crafted to closely
match what we observe in the wild rather than synthetic shapes.
"""

from __future__ import annotations

import pytest

from aone.gmail.normalize import normalize


# ─── Sample 1: RFC 3676 signature (Gmail / Apple Mail) ───────────────


SAMPLE_SIGNATURE_RFC = """\
Hi Bob,

Thanks for sending the budget over. Numbers look right.

--
Alice Lopez
CEO, Acme Corp
+1 555-867-5309
alice@acme.com
"""


def test_strips_rfc_signature() -> None:
    cleaned = normalize(SAMPLE_SIGNATURE_RFC)
    assert cleaned == "Hi Bob,\n\nThanks for sending the budget over. Numbers look right."


# ─── Sample 2: Gmail-style reply chain in English ────────────────────


SAMPLE_REPLY_CHAIN_EN = """\
Sure, 2pm works for me.

On Mon, May 24, 2026 at 3:00 PM Bob Smith <bob@acme.com> wrote:
> Are you available tomorrow at 2pm to walk through the proposal?
>
> Bob
"""


def test_strips_english_reply_chain() -> None:
    cleaned = normalize(SAMPLE_REPLY_CHAIN_EN)
    assert cleaned == "Sure, 2pm works for me."


# ─── Sample 3: Outlook-style reply ("-----Original Message-----") ────


SAMPLE_OUTLOOK_REPLY = """\
Approved. Please proceed with the vendor.

-----Original Message-----
From: Bob Smith <bob@acme.com>
Sent: Monday, May 24, 2026 3:00 PM
To: Alice Lopez <alice@acme.com>
Subject: Budget approval

Hi Alice,
Can you approve the Q3 vendor budget today?
Thanks,
Bob
"""


def test_strips_outlook_original_message_block() -> None:
    cleaned = normalize(SAMPLE_OUTLOOK_REPLY)
    assert cleaned == "Approved. Please proceed with the vendor."


# ─── Sample 4: Newsletter (no normalization expected) ────────────────


SAMPLE_NEWSLETTER = """\
This week in AI:

1. Anthropic released Claude Haiku 4.5 with a new tool-use API.
2. OpenAI announced lower prices for the embeddings endpoint.
3. LangGraph 1.2 added typed state machines.

Read more on our blog.
"""


def test_leaves_newsletter_untouched() -> None:
    cleaned = normalize(SAMPLE_NEWSLETTER)
    # No signature, no reply chain, no quoted lines: only blank lines may be
    # collapsed, content is preserved verbatim.
    assert "Claude Haiku 4.5" in cleaned
    assert "Read more on our blog." in cleaned
    assert "-- " not in cleaned  # nothing to strip


# ─── Sample 5: Plain text, no markers (no normalization expected) ────


SAMPLE_PLAIN = "Quick heads up: I'll be on vacation next week."


def test_leaves_plain_text_unchanged() -> None:
    assert normalize(SAMPLE_PLAIN) == SAMPLE_PLAIN


# ─── Additional coverage ─────────────────────────────────────────────


SAMPLE_SPANISH_REPLY = """\
Perfecto, nos vemos el lunes.

El lun, 24 may 2026 a las 15:00, Juan Pérez <juan@acme.com> escribió:
> Hola, ¿te viene bien el lunes a las 10?
> Saludos,
> Juan
"""


def test_strips_spanish_reply_chain() -> None:
    cleaned = normalize(SAMPLE_SPANISH_REPLY)
    assert cleaned == "Perfecto, nos vemos el lunes."


SAMPLE_MOBILE_SIGNATURE = """\
Looks good!

Sent from my iPhone
"""


def test_strips_mobile_signature() -> None:
    assert normalize(SAMPLE_MOBILE_SIGNATURE) == "Looks good!"


SAMPLE_COMBINED = """\
Sure.

--
Alice

On Mon, May 24, 2026 at 3:00 PM Bob <bob@acme.com> wrote:
> Are you available?
"""


def test_strips_combined_signature_and_reply() -> None:
    """Reply chain comes after signature in the source; both must go."""
    cleaned = normalize(SAMPLE_COMBINED)
    assert cleaned == "Sure."


def test_strips_quoted_lines_without_attribution() -> None:
    """Some legacy clients quote without an attribution header."""
    body = "Sure thing.\n> Earlier message line 1\n> Earlier message line 2\n"
    assert normalize(body) == "Sure thing."


# ─── Properties ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "body",
    [
        SAMPLE_SIGNATURE_RFC,
        SAMPLE_REPLY_CHAIN_EN,
        SAMPLE_OUTLOOK_REPLY,
        SAMPLE_NEWSLETTER,
        SAMPLE_PLAIN,
        SAMPLE_SPANISH_REPLY,
        SAMPLE_MOBILE_SIGNATURE,
        SAMPLE_COMBINED,
    ],
)
def test_normalize_is_idempotent(body: str) -> None:
    once = normalize(body)
    twice = normalize(once)
    assert once == twice


def test_empty_string() -> None:
    assert normalize("") == ""


def test_whitespace_only() -> None:
    assert normalize("\n\n   \n") == ""
