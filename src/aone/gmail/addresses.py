"""Helpers for parsing ``From``/``To``/``Cc`` style email-address headers."""

from __future__ import annotations

from email.utils import parseaddr


def parse_from(header_value: str) -> tuple[str, str]:
    """Parse a header into ``(display_name, email_address)``.

    Both fields may be empty:

    * ``display_name`` is the unquoted portion of ``"Name" <addr>``;
      empty when the header is just an address.
    * ``email_address`` is lowercased and validated to contain ``@``.
      Empty when no real address is present (``parseaddr`` is otherwise
      permissive and would happily call ``"just"`` an address for
      input like ``"just a name with no email"``).
    """
    if not header_value:
        return "", ""
    name, address = parseaddr(header_value)
    if "@" not in address:
        return "", ""
    return name, address.lower()


def extract_email_address(header_value: str) -> str:
    """Return the bare email address from a ``From``/``To``/``Cc`` header.

    Convenience wrapper over :func:`parse_from` when the display name
    isn't needed.
    """
    _name, address = parse_from(header_value)
    return address
