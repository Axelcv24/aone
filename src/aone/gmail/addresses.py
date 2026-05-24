"""Helpers for parsing ``From``/``To``/``Cc`` style email-address headers."""

from __future__ import annotations

from email.utils import parseaddr


def extract_email_address(header_value: str) -> str:
    """Extract the bare email address from a header like ``"Alice <a@x.com>"``.

    Returns a lowercase address, or an empty string when the header is
    missing or doesn't contain a real ``@``. Uses
    :func:`email.utils.parseaddr`, which is permissive enough to return
    "just" as a candidate address for inputs like
    ``"just a name with no email"`` — we filter those out by requiring
    an ``@`` in the result.
    """
    if not header_value:
        return ""
    _name, address = parseaddr(header_value)
    if "@" not in address:
        return ""
    return address.lower()
