"""AONE-201 spike validation demo.

Runs the OAuth flow (browser opens on the first run) and prints metadata
for the last 5 messages in the authenticated user's Gmail inbox. No
persistence — this script only proves the OAuth + Gmail API path works
end to end.

Run with::

    uv run python -m aone.gmail.demo
"""

from __future__ import annotations

from aone.gmail.auth import get_service


def main() -> None:
    print("Connecting to Gmail (the browser will open on the first run)…")
    service = get_service()
    print("Connected. Fetching the 5 most recent message IDs…\n")

    result = service.users().messages().list(userId="me", maxResults=5).execute()
    messages = result.get("messages", [])

    if not messages:
        print("No messages found in this account.")
        return

    for i, msg_ref in enumerate(messages, start=1):
        full = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=msg_ref["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )
        headers = {h["name"]: h["value"] for h in full["payload"]["headers"]}
        print(f"  {i}. id={msg_ref['id']}")
        print(f"     From:    {headers.get('From', '(missing)')}")
        print(f"     Subject: {headers.get('Subject', '(missing)')}")
        print(f"     Date:    {headers.get('Date', '(missing)')}\n")


if __name__ == "__main__":
    main()
