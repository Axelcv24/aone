"""Extract a sender filter from a natural-language question.

When the user mentions a sender (an email address or a recognisable
brand like "Levi" / "Acme"), :func:`extract_sender_filter` returns the
canonical address from the cache that ``search_emails`` should filter
by. That filter is much more reliable than relying on semantic
similarity to surface the right messages — marketing emails from
``info@mail.levi.com`` rarely contain the literal word "Levi" in the
body, and a question asking about them shouldn't be answered by
randomly-ranked FAISS hits.

Strategy (cheap, deterministic, no extra LLM call):

1. If the question contains an explicit email address, use it.
2. Otherwise tokenise the question and compare each token against the
   domain parts of every known sender in the cache. The first match
   wins. Minimum token length of 3 to avoid false positives like
   ``de`` accidentally matching ``de@example.de``.

The cache acts as the vocabulary: only senders we've actually seen
become candidates. That keeps brand recognition grounded in real data
instead of an open-ended NER guess.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from aone.storage.cache import EmailCache

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Words that aren't useful as brand tokens — common Spanish/English
# preposition / question words that often appear next to a sender name.
_STOPWORDS = frozenset(
    {
        "the", "a", "of", "to", "from", "and", "or", "for", "in", "on",
        "by", "with", "at",
        "el", "la", "los", "las", "de", "del", "y", "o", "con", "en",
        "por", "para", "que", "qué", "cuál", "cuáles", "tengo", "tienes",
        "tiene", "tienen", "hay", "mira", "puedes", "muéstrame", "muestra",
        "facturas", "factura", "correos", "correo", "emails", "email",
        "mensajes", "mensaje", "messages", "message",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'-]*")


def extract_sender_filter(question: str, cache: EmailCache) -> str | None:
    """Best-effort extraction of a sender address from the question.

    Returns ``None`` when no plausible sender is mentioned.
    """
    if not question:
        return None

    # 1. Explicit email address in the question.
    email_match = _EMAIL_RE.search(question)
    if email_match:
        return email_match.group(0).lower()

    # 2. Token vs domain match across cached senders.
    senders = _known_senders(cache)
    if not senders:
        return None

    question_tokens = _tokenize(question)
    for token in question_tokens:
        if token in _STOPWORDS or len(token) < 3:
            continue
        for sender_address in senders:
            for part in _domain_parts(sender_address):
                if _token_matches(token, part):
                    return sender_address

    return None


def _token_matches(token: str, brand_part: str) -> bool:
    """Match a question token against a sender's domain part.

    - Exact match.
    - Possessive form: ``levi's`` against ``levi`` (apostrophe + optional
      ``s`` suffix stripped).
    - Prefix when the brand part is at least 4 chars: ``leviswear`` would
      not match ``levi`` (too lax), but ``levi's`` does (after strip).

    Designed to avoid common false positives — bare ``com``/``org`` are
    excluded by the caller — while catching the realistic shapes:
    ``Levi's``, ``Adidas``, ``Acme``, ``LinkedIn``.
    """
    if token == brand_part:
        return True
    # Strip a trailing apostrophe (curly or straight) plus optional 's'.
    stripped = re.sub(r"['’]s?$", "", token)
    return stripped == brand_part


def _known_senders(cache: EmailCache) -> list[str]:
    """Top-50 sender addresses from the cache, ordered by message count."""
    # Pull a generous top-N so even mid-frequency senders are matchable.
    return [address for address, _count in cache.stats(top_n=50).top_senders]


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _domain_parts(email_address: str) -> Iterable[str]:
    """Decompose ``"info@mail.levi.com"`` into ``("mail", "levi", "com",
    "mail.levi.com")`` so a single question token ("levi") can match."""
    if "@" not in email_address:
        return []
    _local, domain = email_address.split("@", 1)
    parts = {domain.lower(), *domain.lower().split(".")}
    # Don't match on tld-only ("com", "es"), they're false-positive bait.
    parts = {p for p in parts if p not in {"com", "es", "net", "org", "io", "co"}}
    return parts
