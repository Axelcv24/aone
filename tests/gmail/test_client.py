"""Tests for ``aone.gmail.client``."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError

from aone.gmail import client as client_module
from aone.gmail.client import (
    _decode_b64url,
    _parse_addresses,
    _parse_message,
    get_message,
    list_messages,
)


def _b64url(text: str) -> str:
    """Encode ``text`` the way Gmail does: URL-safe base64 without ``=`` padding."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode().rstrip("=")


def _list_service(responses: list[Any]) -> tuple[MagicMock, MagicMock]:
    """Build a mocked service whose ``.messages().list().execute()`` yields ``responses``."""
    request = MagicMock()
    request.execute.side_effect = responses
    service = MagicMock()
    service.users.return_value.messages.return_value.list.return_value = request
    return service, request


def _get_service(raw: dict[str, Any]) -> MagicMock:
    """Build a mocked service whose ``.messages().get().execute()`` returns ``raw``."""
    request = MagicMock()
    request.execute.return_value = raw
    service = MagicMock()
    service.users.return_value.messages.return_value.get.return_value = request
    return service


def _http_error(status: int) -> HttpError:
    resp = MagicMock()
    resp.status = status
    resp.reason = "fake"
    return HttpError(resp, content=b"{}")


# ─── list_messages ───────────────────────────────────────────────────


def test_list_messages_paginates_until_no_token() -> None:
    pages = [
        {"messages": [{"id": "a"}, {"id": "b"}], "nextPageToken": "tok1"},
        {"messages": [{"id": "c"}, {"id": "d"}]},
    ]
    service, request = _list_service(pages)

    ids = list_messages(service, limit=10)

    assert ids == ["a", "b", "c", "d"]
    assert request.execute.call_count == 2


def test_list_messages_stops_at_limit_mid_page() -> None:
    pages = [
        {"messages": [{"id": "a"}, {"id": "b"}, {"id": "c"}], "nextPageToken": "tok"},
    ]
    service, _ = _list_service(pages)

    assert list_messages(service, limit=2) == ["a", "b"]


def test_list_messages_empty_mailbox() -> None:
    service, _ = _list_service([{}])
    assert list_messages(service, limit=5) == []


def test_list_messages_limit_zero_skips_api() -> None:
    service, request = _list_service([])
    assert list_messages(service, limit=0) == []
    request.execute.assert_not_called()


# ─── get_message / _parse_message ────────────────────────────────────


def test_get_message_simple_text_only() -> None:
    raw = {
        "id": "msg1",
        "threadId": "thread1",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "Hello world",
        "internalDate": "1716579720000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "Alice <alice@x.com>"},
                {"name": "To", "value": "bob@y.com"},
                {"name": "Subject", "value": "hi"},
            ],
            "body": {"data": _b64url("Hello world body")},
        },
    }
    service = _get_service(raw)

    email = get_message(service, "msg1")

    assert email.id == "msg1"
    assert email.thread_id == "thread1"
    assert email.from_ == "Alice <alice@x.com>"
    assert email.to == ["bob@y.com"]
    assert email.subject == "hi"
    assert email.body_text == "Hello world body"
    assert email.body_html == ""
    assert email.snippet == "Hello world"
    assert email.internal_date == 1716579720000
    assert email.labels == ["INBOX", "UNREAD"]


def test_parse_message_multipart_alternative_picks_both_bodies() -> None:
    raw = {
        "id": "x",
        "threadId": "x",
        "snippet": "",
        "internalDate": "0",
        "labelIds": [],
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [],
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64url("plain body")}},
                {"mimeType": "text/html", "body": {"data": _b64url("<p>html body</p>")}},
            ],
        },
    }
    email = _parse_message(raw)
    assert email.body_text == "plain body"
    assert email.body_html == "<p>html body</p>"


def test_parse_message_nested_multipart_skips_attachment() -> None:
    """``multipart/mixed`` containing ``multipart/alternative`` + a PDF attachment."""
    raw = {
        "id": "x",
        "threadId": "x",
        "snippet": "",
        "internalDate": "0",
        "labelIds": [],
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [],
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": _b64url("plain")}},
                        {"mimeType": "text/html", "body": {"data": _b64url("<b>html</b>")}},
                    ],
                },
                {
                    "mimeType": "application/pdf",
                    "filename": "invoice.pdf",
                    "body": {"attachmentId": "abc123"},
                },
            ],
        },
    }
    email = _parse_message(raw)
    assert email.body_text == "plain"
    assert email.body_html == "<b>html</b>"


def test_parse_message_handles_missing_headers_and_dates() -> None:
    raw = {
        "id": "x",
        "threadId": "x",
        "payload": {"headers": []},
    }
    email = _parse_message(raw)
    assert email.from_ == ""
    assert email.to == []
    assert email.subject == ""
    assert email.snippet == ""
    assert email.internal_date == 0
    assert email.labels == []


# ─── helpers ─────────────────────────────────────────────────────────


def test_parse_addresses_single_bare() -> None:
    assert _parse_addresses("alice@x.com") == ["alice@x.com"]


def test_parse_addresses_multiple_named() -> None:
    assert _parse_addresses("Alice <alice@x.com>, Bob <bob@y.com>") == [
        "alice@x.com",
        "bob@y.com",
    ]


def test_parse_addresses_empty() -> None:
    assert _parse_addresses("") == []


def test_decode_b64url_handles_unicode() -> None:
    raw = base64.urlsafe_b64encode("héllo ✓".encode()).decode().rstrip("=")
    assert _decode_b64url(raw) == "héllo ✓"


def test_decode_b64url_replaces_invalid_utf8() -> None:
    payload = b"\xff\xfe\x00OK"
    raw = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    decoded = _decode_b64url(raw)
    # Bad bytes are replaced; the trailing "OK" survives.
    assert decoded.endswith("OK")


# ─── retry / backoff ─────────────────────────────────────────────────


def test_retry_on_429_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    service, request = _list_service([_http_error(429), {"messages": [{"id": "ok"}]}])

    assert list_messages(service, limit=5) == ["ok"]
    assert request.execute.call_count == 2


def test_retry_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    service, request = _list_service([_http_error(503)] * 5)

    with pytest.raises(HttpError):
        list_messages(service, limit=5)
    assert request.execute.call_count == 5


def test_no_retry_on_non_retryable_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    service, request = _list_service([_http_error(400)])

    with pytest.raises(HttpError):
        list_messages(service, limit=5)
    assert request.execute.call_count == 1
