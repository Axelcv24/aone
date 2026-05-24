"""Gmail OAuth authentication and service factory.

Implements the desktop OAuth flow: on first run, opens the system browser
so the user can grant access; subsequent runs reuse and silently refresh
the cached token.

Spike reference: docs/spikes/gmail-oauth.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Project root: src/aone/gmail/auth.py  →  /Users/.../Aone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
DEFAULT_TOKEN_PATH = PROJECT_ROOT / "token.json"

# Minimal scope: read-only access to Gmail. The app cannot send, modify,
# or delete messages with this scope. See ADR (TBD) on scope selection.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailAuthError(RuntimeError):
    """Raised when Gmail OAuth setup is missing or broken."""


def get_service(
    credentials_path: Path | None = None,
    token_path: Path | None = None,
) -> Any:
    """Return an authenticated Gmail API service.

    Behavior:
        - If a valid cached token exists at ``token_path``, reuses it.
        - If the token exists but is expired, refreshes it silently using
          the refresh token.
        - If no token exists, launches the desktop OAuth flow: spins up a
          local server, opens the default browser to Google's consent
          screen, and waits for the redirect with the authorization code.
        - On success, the resulting credentials are written to
          ``token_path`` for future runs.

    Args:
        credentials_path: path to the OAuth client credentials JSON
            downloaded from Google Cloud Console. Defaults to
            ``<repo>/credentials.json``.
        token_path: path to the cached user token. Defaults to
            ``<repo>/token.json``.

    Returns:
        An authenticated Gmail v1 service ready for API calls
        (e.g. ``service.users().messages().list(...)``).

    Raises:
        GmailAuthError: when ``credentials.json`` is missing.
    """
    credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
    token_path = token_path or DEFAULT_TOKEN_PATH

    if not credentials_path.exists():
        raise GmailAuthError(
            f"OAuth client credentials not found at {credentials_path}.\n"
            f"Download credentials.json from Google Cloud Console "
            f"(APIs & Services → Credentials → OAuth 2.0 Client IDs) "
            f"and place it at the repo root."
        )

    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), SCOPES
            )
            # port=0 lets the OS pick a free port for the local OAuth callback.
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)
