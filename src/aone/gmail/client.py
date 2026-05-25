"""Gmail message client: list and fetch.

Builds on the authenticated service from :mod:`aone.gmail.auth` to:

* list message IDs across pages,
* fetch full message contents, and
* parse the MIME payload into a clean :class:`~aone.gmail.types.Email`.

Transient HTTP errors (429 + 5xx) are retried with exponential backoff
and jitter so a brief rate-limit spike doesn't kill a long sync.
"""

from __future__ import annotations

import base64
import random
import time
from collections.abc import Callable
from email.utils import getaddresses
from typing import Any, TypeVar

import html2text
from googleapiclient.errors import HttpError

from aone.gmail.normalize import normalize
from aone.gmail.types import Email

# Configured once at module import; cheap to reuse.
_HTML2TEXT = html2text.HTML2Text()
_HTML2TEXT.ignore_links = True       # URLs add noise to embeddings
_HTML2TEXT.ignore_images = True      # alt text not worth the noise
_HTML2TEXT.body_width = 0            # don't hard-wrap — preserves long amounts on one line
_HTML2TEXT.unicode_snob = True       # keep ñ/é/€ instead of escaping
_HTML2TEXT.skip_internal_links = True

T = TypeVar("T")

# Hard ceiling for pagination. A page is up to 100 message refs, so 500
# pages = 50,000 messages. v0 ``--limit`` defaults are far below this;
# this is purely a runaway-loop guard.
MAX_PAGES = 500

# Gmail API page size cap.
PAGE_SIZE = 100

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


def _retry(
    call: Callable[[], T],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
) -> T:
    """Invoke ``call()`` with exponential backoff on 429 / 5xx."""
    for attempt in range(max_attempts):
        try:
            return call()
        except HttpError as exc:
            status = exc.resp.status
            if status not in RETRYABLE_STATUSES or attempt == max_attempts - 1:
                raise
            sleep_for = base_delay * (2**attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_for)
    raise RuntimeError("unreachable")  # pragma: no cover


def list_messages(
    service: Any,
    *,
    limit: int = 500,
    query: str | None = None,
) -> list[str]:
    """Return the IDs of up to ``limit`` Gmail messages.

    Args:
        service: authenticated Gmail v1 service (see :func:`aone.gmail.auth.get_service`).
        limit: maximum number of message IDs to return. ``list_messages`` will
            pull across pages until it reaches this limit or runs out.
        query: optional Gmail search query (same syntax as the web UI),
            e.g. ``"is:unread"`` or ``"from:billing@acme.com after:2025/01/01"``.

    Returns:
        Message IDs in Gmail's default order (most recent first). The list
        may be shorter than ``limit`` if the mailbox has fewer messages.
    """
    if limit <= 0:
        return []

    ids: list[str] = []
    page_token: str | None = None

    for _ in range(MAX_PAGES):
        remaining = limit - len(ids)
        if remaining <= 0:
            break
        page_size = min(PAGE_SIZE, remaining)

        request = (
            service.users()
            .messages()
            .list(
                userId="me",
                maxResults=page_size,
                pageToken=page_token,
                q=query,
            )
        )
        response = _retry(request.execute)

        for ref in response.get("messages", []):
            ids.append(ref["id"])
            if len(ids) >= limit:
                return ids

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return ids


def get_message(service: Any, message_id: str) -> Email:
    """Fetch a Gmail message by ID and return a parsed :class:`Email`."""
    request = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
    )
    raw = _retry(request.execute)
    return _parse_message(raw)


def _parse_message(raw: dict[str, Any]) -> Email:
    """Convert Gmail's raw response into an :class:`Email`."""
    payload = raw.get("payload", {})
    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
    body_text, body_html = _extract_bodies(payload)

    # Marketing and transactional emails are routinely HTML-only with
    # no text/plain alternative — in that case body_text is empty and
    # the agent sees nothing. Fall back to converting the HTML so the
    # visible text (including amounts, order numbers, headlines) ends
    # up in body_text and feeds through normalize → body_clean → FAISS.
    if not body_text.strip() and body_html.strip():
        body_text = _html_to_text(body_html)

    return Email(
        id=raw["id"],
        thread_id=raw["threadId"],
        from_=headers.get("From", ""),
        to=_parse_addresses(headers.get("To", "")),
        subject=headers.get("Subject", ""),
        body_text=body_text,
        body_html=body_html,
        body_clean=normalize(body_text),
        snippet=raw.get("snippet", ""),
        internal_date=int(raw.get("internalDate", "0")),
        labels=list(raw.get("labelIds", [])),
    )


def _extract_bodies(part: dict[str, Any]) -> tuple[str, str]:
    """Walk the MIME tree and collect the first ``text/plain`` + ``text/html`` bodies.

    Attachments (any part with a ``filename``) are ignored. If the message
    contains alternative versions, the first one of each MIME type wins —
    matching what most mail clients render.
    """
    text_plain = ""
    text_html = ""

    def walk(node: dict[str, Any]) -> None:
        nonlocal text_plain, text_html
        if node.get("filename"):
            return
        mime = node.get("mimeType", "")
        data = node.get("body", {}).get("data")
        if data:
            decoded = _decode_b64url(data)
            if mime == "text/plain" and not text_plain:
                text_plain = decoded
            elif mime == "text/html" and not text_html:
                text_html = decoded
        for child in node.get("parts", []):
            walk(child)

    walk(part)
    return text_plain, text_html


def _html_to_text(html: str) -> str:
    """Convert HTML email body to plain text.

    Used as a fallback when an email has no ``text/plain`` MIME part
    (common in marketing/transactional emails). Tries to keep
    line-level structure so amounts on their own line stay visible
    while stripping links, images, CSS, and tracking pixels.
    """
    try:
        return _HTML2TEXT.handle(html).strip()
    except Exception:  # noqa: BLE001 — never let HTML parsing crash a sync
        return ""


def _decode_b64url(data: str) -> str:
    """Decode Gmail's URL-safe base64.

    Gmail strips trailing ``=`` padding and may encode bodies in a
    character set other than UTF-8 (legacy mail). We re-pad and decode
    with ``errors="replace"`` so malformed bytes never crash the parser.
    """
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _parse_addresses(header_value: str) -> list[str]:
    """Parse an address-list header (To / Cc / Bcc) into bare email strings."""
    if not header_value:
        return []
    return [addr for _, addr in getaddresses([header_value]) if addr]
