"""Tests for ``aone.sync`` (AONE-501)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from aone.gmail.types import Email
from aone.storage.cache import EmailCache
from aone.storage.vector import VectorIndex
from aone.sync import SyncResult, perform_sync


class _FakeEmbedder:
    provider_name = "fake"
    model_name = "fake/v1"
    dims = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 0.0, 1.0] for t in texts]


def _email(id_: str, body: str = "body") -> Email:
    return Email(
        id=id_,
        thread_id=f"t-{id_}",
        from_="alice@x.com",
        to=["axel@example.com"],
        subject=f"s-{id_}",
        body_text=body,
        body_html=f"<p>{body}</p>",
        body_clean=body,
        snippet=body[:80],
        internal_date=1_700_000_000_000,
        labels=["INBOX"],
    )


def _stub_service(
    *,
    list_returns: list[list[dict[str, Any]]],
    get_returns: dict[str, Email | Exception],
) -> MagicMock:
    """Build a mocked Gmail service.

    list_returns is a list of pages (one element per ``.execute`` call).
    get_returns maps message_id → Email (success) or Exception (raise).
    """
    list_req = MagicMock()
    list_req.execute.side_effect = list_returns

    def _get(userId: str, id: str, format: str) -> MagicMock:  # noqa: A002
        inner = MagicMock()
        result = get_returns.get(id)
        if isinstance(result, Exception):
            inner.execute.side_effect = result
        else:
            inner.execute.return_value = _email_to_raw(result) if result else {}
        return inner

    messages_proxy = MagicMock()
    messages_proxy.list.return_value = list_req
    messages_proxy.get.side_effect = _get

    users_proxy = MagicMock()
    users_proxy.messages.return_value = messages_proxy

    service = MagicMock()
    service.users.return_value = users_proxy
    return service


def _email_to_raw(email: Email) -> dict[str, Any]:
    """Mock Gmail's raw payload for a given Email (just enough for the parser)."""
    import base64

    return {
        "id": email.id,
        "threadId": email.thread_id,
        "labelIds": list(email.labels),
        "snippet": email.snippet,
        "internalDate": str(email.internal_date),
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": email.from_},
                {"name": "To", "value": ", ".join(email.to)},
                {"name": "Subject", "value": email.subject},
            ],
            "body": {
                "data": base64.urlsafe_b64encode(email.body_text.encode()).decode().rstrip("=")
            },
        },
    }


def _setup_storage() -> tuple[EmailCache, VectorIndex]:
    return EmailCache(), VectorIndex(_FakeEmbedder())


# ─── Happy path ──────────────────────────────────────────────────────


def test_pulls_and_indexes_new_messages_into_empty_storage() -> None:
    cache, index = _setup_storage()
    service = _stub_service(
        list_returns=[{"messages": [{"id": "a"}, {"id": "b"}]}],
        get_returns={"a": _email("a"), "b": _email("b")},
    )

    result = perform_sync(service=service, cache=cache, index=index, limit=10)

    assert isinstance(result, SyncResult)
    assert result.listed == 2
    assert result.fetched == 2
    assert result.failed == 0
    assert result.already_cached == 0
    assert len(cache) == 2
    assert len(index) == 2


def test_skips_already_cached_ids() -> None:
    cache, index = _setup_storage()
    cache.add(_email("a"))
    # index intentionally not pre-populated to keep test focused

    service = _stub_service(
        list_returns=[{"messages": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}],
        get_returns={"b": _email("b"), "c": _email("c")},
    )

    result = perform_sync(service=service, cache=cache, index=index, limit=10)

    assert result.listed == 3
    assert result.already_cached == 1
    assert result.fetched == 2  # only b and c
    assert len(cache) == 3


def test_returns_zero_fetched_when_everything_is_already_cached() -> None:
    cache, index = _setup_storage()
    cache.add(_email("a"))
    cache.add(_email("b"))

    service = _stub_service(
        list_returns=[{"messages": [{"id": "a"}, {"id": "b"}]}],
        get_returns={},
    )

    result = perform_sync(service=service, cache=cache, index=index, limit=10)

    assert result.fetched == 0
    assert result.already_cached == 2


# ─── Error handling ──────────────────────────────────────────────────


def test_per_message_fetch_error_does_not_abort_the_run() -> None:
    cache, index = _setup_storage()
    service = _stub_service(
        list_returns=[{"messages": [{"id": "a"}, {"id": "broken"}, {"id": "c"}]}],
        get_returns={
            "a": _email("a"),
            "broken": RuntimeError("oh no"),
            "c": _email("c"),
        },
    )
    errors: list[tuple[str, Exception]] = []

    result = perform_sync(
        service=service,
        cache=cache,
        index=index,
        limit=10,
        on_error=lambda mid, exc: errors.append((mid, exc)),
    )

    assert result.fetched == 2
    assert result.failed == 1
    assert len(cache) == 2
    assert errors == [("broken", errors[0][1])]


def test_empty_mailbox_returns_zeroed_result() -> None:
    cache, index = _setup_storage()
    service = _stub_service(list_returns=[{}], get_returns={})

    result = perform_sync(service=service, cache=cache, index=index, limit=10)

    assert result.listed == 0
    assert result.fetched == 0
    assert len(cache) == 0
    assert len(index) == 0


# ─── Callbacks ───────────────────────────────────────────────────────


def test_on_fetch_callback_invoked_per_email() -> None:
    cache, index = _setup_storage()
    service = _stub_service(
        list_returns=[{"messages": [{"id": "a"}, {"id": "b"}]}],
        get_returns={"a": _email("a"), "b": _email("b")},
    )
    seen: list[str] = []

    perform_sync(
        service=service,
        cache=cache,
        index=index,
        limit=10,
        on_fetch=lambda email: seen.append(email.id),
    )

    assert seen == ["a", "b"]


# ─── Limit + query passthrough ───────────────────────────────────────


def test_query_passes_through_to_list_messages() -> None:
    cache, index = _setup_storage()
    service = _stub_service(
        list_returns=[{}],
        get_returns={},
    )

    perform_sync(
        service=service,
        cache=cache,
        index=index,
        limit=10,
        query="from:billing@acme.com",
    )

    # list_messages was called with q=our query
    list_kwargs = service.users().messages().list.call_args.kwargs
    assert list_kwargs["q"] == "from:billing@acme.com"
