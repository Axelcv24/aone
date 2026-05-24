"""End-to-end validation script: OAuth + Gmail client.

Lists the 5 most recent messages in the authenticated user's mailbox and
prints the parsed ``Email`` fields. Used to manually validate the work
from AONE-201 (OAuth), AONE-202 (auth tests), and AONE-203 (client).

Run with::

    uv run python -m aone.gmail.demo
"""

from __future__ import annotations

from aone.gmail.auth import get_service
from aone.gmail.client import get_message, list_messages


def main() -> None:
    print("Connecting to Gmail (the browser will open on the first run)…")
    service = get_service()
    print("Connected. Listing the 5 most recent message IDs…\n")

    ids = list_messages(service, limit=5)
    if not ids:
        print("No messages found in this account.")
        return

    for i, message_id in enumerate(ids, start=1):
        email = get_message(service, message_id)
        print(f"  {i}. id={email.id}")
        print(f"     From:    {email.from_}")
        print(f"     To:      {', '.join(email.to) if email.to else '(none)'}")
        print(f"     Subject: {email.subject}")
        print(f"     Snippet: {email.snippet[:80]}")
        print(
            f"     Body:    {len(email.body_text)} chars text · "
            f"{len(email.body_html)} chars html"
        )
        print(f"     Labels:  {email.labels}\n")


if __name__ == "__main__":
    main()
