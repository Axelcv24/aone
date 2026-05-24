"""Tests for ``aone.gmail.auth``.

All tests mock the OAuth flow and the Gmail service builder so they never
hit the network and never open a browser.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aone.gmail.auth import (
    SCOPES,
    GmailAuthError,
    get_service,
)


@pytest.fixture
def credentials_path(tmp_path: Path) -> Path:
    """A fake ``credentials.json`` whose content does not matter (calls are mocked)."""
    path = tmp_path / "credentials.json"
    path.write_text('{"installed": {"client_id": "fake", "client_secret": "fake"}}')
    return path


@pytest.fixture
def token_path(tmp_path: Path) -> Path:
    """Where ``token.json`` would live; absent unless a test writes to it."""
    return tmp_path / "token.json"


def test_raises_when_credentials_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(GmailAuthError, match="credentials.json"):
        get_service(credentials_path=missing, token_path=tmp_path / "token.json")


@patch("aone.gmail.auth.build")
@patch("aone.gmail.auth.InstalledAppFlow")
def test_first_time_auth_launches_flow_and_persists_token(
    mock_flow_cls: MagicMock,
    mock_build: MagicMock,
    credentials_path: Path,
    token_path: Path,
) -> None:
    """No cached token → InstalledAppFlow runs, token.json gets written."""
    fresh_creds = MagicMock()
    fresh_creds.to_json.return_value = '{"token": "freshly-issued"}'
    flow_instance = MagicMock()
    flow_instance.run_local_server.return_value = fresh_creds
    mock_flow_cls.from_client_secrets_file.return_value = flow_instance
    fake_service = MagicMock()
    mock_build.return_value = fake_service

    result = get_service(credentials_path=credentials_path, token_path=token_path)

    mock_flow_cls.from_client_secrets_file.assert_called_once_with(
        str(credentials_path), SCOPES
    )
    flow_instance.run_local_server.assert_called_once_with(port=0)
    assert token_path.exists()
    assert token_path.read_text() == '{"token": "freshly-issued"}'
    mock_build.assert_called_once_with(
        "gmail", "v1", credentials=fresh_creds, cache_discovery=False
    )
    assert result is fake_service


@patch("aone.gmail.auth.build")
@patch("aone.gmail.auth.Credentials")
def test_reuses_valid_cached_token(
    mock_credentials_cls: MagicMock,
    mock_build: MagicMock,
    credentials_path: Path,
    token_path: Path,
) -> None:
    """Cached creds that are still valid → no flow, no rewrite."""
    token_path.write_text('{"token": "cached-and-valid"}')

    cached = MagicMock()
    cached.valid = True
    cached.expired = False
    mock_credentials_cls.from_authorized_user_file.return_value = cached
    mock_build.return_value = MagicMock()

    get_service(credentials_path=credentials_path, token_path=token_path)

    mock_credentials_cls.from_authorized_user_file.assert_called_once_with(
        str(token_path), SCOPES
    )
    cached.refresh.assert_not_called()
    assert token_path.read_text() == '{"token": "cached-and-valid"}'


@patch("aone.gmail.auth.Request")
@patch("aone.gmail.auth.build")
@patch("aone.gmail.auth.Credentials")
def test_refreshes_expired_token_silently(
    mock_credentials_cls: MagicMock,
    mock_build: MagicMock,
    mock_request_cls: MagicMock,
    credentials_path: Path,
    token_path: Path,
) -> None:
    """Expired creds with a refresh_token → refresh in place, browser not touched."""
    token_path.write_text('{"token": "stale"}')

    cached = MagicMock()
    cached.valid = False
    cached.expired = True
    cached.refresh_token = "refresh-token-present"
    cached.to_json.return_value = '{"token": "refreshed"}'
    mock_credentials_cls.from_authorized_user_file.return_value = cached
    mock_build.return_value = MagicMock()

    get_service(credentials_path=credentials_path, token_path=token_path)

    cached.refresh.assert_called_once()
    assert token_path.read_text() == '{"token": "refreshed"}'


@patch("aone.gmail.auth.build")
@patch("aone.gmail.auth.InstalledAppFlow")
@patch("aone.gmail.auth.Credentials")
def test_relaunches_flow_when_token_invalid_without_refresh(
    mock_credentials_cls: MagicMock,
    mock_flow_cls: MagicMock,
    mock_build: MagicMock,
    credentials_path: Path,
    token_path: Path,
) -> None:
    """Cached creds that cannot refresh (e.g. revoked) → restart the OAuth flow."""
    token_path.write_text('{"token": "revoked"}')

    bad = MagicMock()
    bad.valid = False
    bad.expired = True
    bad.refresh_token = None
    mock_credentials_cls.from_authorized_user_file.return_value = bad

    fresh = MagicMock()
    fresh.to_json.return_value = '{"token": "new"}'
    flow_instance = MagicMock()
    flow_instance.run_local_server.return_value = fresh
    mock_flow_cls.from_client_secrets_file.return_value = flow_instance
    mock_build.return_value = MagicMock()

    get_service(credentials_path=credentials_path, token_path=token_path)

    bad.refresh.assert_not_called()
    mock_flow_cls.from_client_secrets_file.assert_called_once_with(
        str(credentials_path), SCOPES
    )
    assert token_path.read_text() == '{"token": "new"}'
