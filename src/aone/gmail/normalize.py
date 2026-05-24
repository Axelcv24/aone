"""Strip signatures and quoted reply chains from email bodies.

The goal is a clean version of the body suitable for embeddings and
semantic search: just what the user actually wrote, without the
historical thread or sig blocks that dominate long email bodies and
poison vector representations.

Conservative on purpose: better to leave a sentence of someone else's
reply in than to chop off the user's own content. The normalizer is
idempotent — running it twice on the same input is a no-op.
"""

from __future__ import annotations

import re

# RFC 3676 §4.3: a signature is delimited by a line containing exactly
# "-- " (two dashes, a space). Some clients drop the trailing space; we
# tolerate both.
_RFC_SIG_DELIMITER = re.compile(r"^-- ?$", re.MULTILINE)

# Ad-hoc signature markers some mobile/desktop clients emit instead of
# (or in addition to) the RFC delimiter.
_AD_HOC_SIG_MARKERS = [
    re.compile(
        r"^Sent from my (iPhone|iPad|Android|Samsung|BlackBerry|Galaxy).*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^Get Outlook for (iOS|Android).*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^Enviado desde mi (iPhone|iPad|Android|Samsung).*$",
        re.MULTILINE | re.IGNORECASE,
    ),
]

# Reply-chain markers: anything below the first match is treated as the
# previous message and discarded. Patterns intentionally anchor at line
# start to avoid matching the marker phrase inside the user's own text.
_REPLY_CHAIN_MARKERS = [
    # Gmail/Apple Mail English: "On Mon, May 24, 2026 at 3:00 PM John Doe <john@x.com> wrote:"
    re.compile(r"^On\s.{1,200}\bwrote:\s*$", re.MULTILINE),
    # Gmail Spanish: "El lun, 24 may 2026 a las 15:00, Juan <juan@x.com> escribió:"
    re.compile(r"^El\s.{1,200}\bescribió:\s*$", re.MULTILINE),
    # Gmail French: "Le ... a écrit :"
    re.compile(r"^Le\s.{1,200}\ba\s+écrit\s*:\s*$", re.MULTILINE),
    # Outlook: "-----Original Message-----" / "-----Forwarded message-----"
    re.compile(r"^-{3,}\s*Original Message\s*-{3,}\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^-{3,}\s*Forwarded message\s*-{3,}\s*$", re.MULTILINE | re.IGNORECASE),
]


def normalize(body_text: str) -> str:
    """Return a clean version of an email body suitable for embeddings.

    Order matters: reply chains are cut before signatures because a
    quoted reply can contain its own sig delimiter that would otherwise
    truncate the user's content.
    """
    if not body_text:
        return ""

    text = _strip_reply_chain(body_text)
    text = _strip_quoted_lines(text)
    text = _strip_signature(text)
    return _collapse_blank_lines(text).strip()


def _strip_reply_chain(text: str) -> str:
    """Drop everything from the first reply-chain marker onward."""
    earliest = len(text)
    for pattern in _REPLY_CHAIN_MARKERS:
        match = pattern.search(text)
        if match and match.start() < earliest:
            earliest = match.start()
    return text[:earliest]


def _strip_quoted_lines(text: str) -> str:
    """Drop lines that start with ``>`` (RFC 5322 quoted-reply notation)."""
    return "\n".join(
        line for line in text.split("\n") if not line.lstrip().startswith(">")
    )


def _strip_signature(text: str) -> str:
    """Drop everything from the first signature delimiter onward."""
    earliest = len(text)

    rfc_match = _RFC_SIG_DELIMITER.search(text)
    if rfc_match:
        earliest = min(earliest, rfc_match.start())

    for pattern in _AD_HOC_SIG_MARKERS:
        match = pattern.search(text)
        if match and match.start() < earliest:
            earliest = match.start()

    return text[:earliest]


def _collapse_blank_lines(text: str) -> str:
    """Collapse runs of 3+ blank lines into a single blank line."""
    return re.sub(r"\n{3,}", "\n\n", text)
