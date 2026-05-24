"""``aggregate_amounts`` tool (AONE-406).

Extracts monetary amounts from a set of emails and aggregates them by
sender, currency, or status. The point of this tool is **trustworthy
math**: the LLM in :func:`generate_response` should never have to sum
$1,200 + $1,500 + $750 in its head. The number it cites comes from
Python's :class:`decimal.Decimal`, not from a neural network.

Why a dedicated tool and not "just let the LLM do it":

* LLMs hallucinate sums sometimes, especially across many items.
* Currencies must not be silently mixed (USD + EUR ≠ a real total).
* Status (paid / pending / overdue) needs deterministic parsing.

Pipeline:

1. **Regex** finds candidate amounts in each email's ``body_clean``
   (signatures and quoted replies already stripped by AONE-204).
2. **Status detection** runs locally over the same body, looking for
   keywords near the amount.
3. **Optional LLM validation** (off by default; opt-in) batches the
   candidates into a single classification call to filter false
   positives (invoice numbers, postal codes, dates that match the
   number shape).
4. **Aggregation** sums by ``(group_by, currency)`` using ``Decimal``;
   different currencies stay separate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from aone.gmail.addresses import extract_email_address
from aone.gmail.types import Email
from aone.llm.client import LLMClient, Role
from aone.storage.cache import EmailCache

# ─── Money parsing ───────────────────────────────────────────────────

# Currency symbol → ISO-ish code. Ambiguous symbols (e.g. ``$``) default
# to USD; callers who need disambiguation should set the code explicitly
# in their emails (e.g. ``"USD 1,200"``).
_SYMBOL_TO_CODE: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
}

_KNOWN_CODES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "INR", "BRL", "MXN", "CAD", "AUD",
        "CHF", "SEK", "NOK", "DKK", "ARS", "CLP", "COP", "PEN", "CNY",
    }
)

# Number shape: 1, 1.5, 1,200, 1,200.00, 1200, 1200.50.
#
# Two alternatives, the order matters:
#   1) Comma-grouped:  \d{1,3}(?:,\d{3})+ — requires at least one
#      group of "_thousands" (the ``+``). Without this, "1024" would
#      match as "102" and leave "4" dangling for another pattern.
#   2) Plain:          \d+ — for numbers without thousands separators.
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"

_AMOUNT_RE = re.compile(
    rf"""
    (?:
        # Pattern A: symbol prefix, optional trailing code
        (?P<symbol_a>[$€£¥₹])\s*
        (?P<amount_a>{_NUM})
        (?:\s*(?P<code_a>USD|EUR|GBP|JPY|INR|BRL|MXN|CAD|AUD|CHF|SEK|NOK|DKK|ARS|CLP|COP|PEN|CNY))?
    )
    |
    (?:
        # Pattern B: code prefix, optional $ in between (e.g. "USD $1,200")
        \b(?P<code_b>USD|EUR|GBP|JPY|INR|BRL|MXN|CAD|AUD|CHF|SEK|NOK|DKK|ARS|CLP|COP|PEN|CNY)
        \s*\$?\s*
        (?P<amount_b>{_NUM})
    )
    |
    (?:
        # Pattern C: amount followed by code (no symbol)
        (?P<amount_c>{_NUM})\s*
        (?P<code_c>USD|EUR|GBP|JPY|INR|BRL|MXN|CAD|AUD|CHF|SEK|NOK|DKK|ARS|CLP|COP|PEN|CNY)\b
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ─── Status keywords (multi-language) ────────────────────────────────

_STATUS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "paid": ("paid", "pagada", "pagado", "settled", "received", "cleared"),
    "overdue": ("overdue", "past due", "vencida", "vencido", "atrasada", "atrasado"),
    "pending": ("pending", "due", "outstanding", "pendiente", "por pagar"),
}


# ─── Public data shapes ──────────────────────────────────────────────


@dataclass(frozen=True)
class AmountMatch:
    """A single monetary amount extracted from an email."""

    email_id: str
    sender: str  # bare email address
    amount: Decimal
    currency: str
    raw_text: str
    status: str | None  # "paid" / "overdue" / "pending" / None


@dataclass(frozen=True)
class Group:
    """Sum of one ``(group_key, currency)`` slice of the matches."""

    key: str
    currency: str
    total: Decimal
    count: int


@dataclass(frozen=True)
class AggregateResult:
    """Output of :class:`AggregateAmounts`."""

    groups: list[Group]
    matches: list[AmountMatch]
    group_by: str

    @property
    def grand_total_by_currency(self) -> dict[str, Decimal]:
        """Convenience: total across all groups, keyed by currency."""
        totals: dict[str, Decimal] = {}
        for g in self.groups:
            totals[g.currency] = totals.get(g.currency, Decimal("0")) + g.total
        return totals


# ─── Sort and group-by sentinels ─────────────────────────────────────

GROUP_BY_SENDER = "sender"
GROUP_BY_CURRENCY = "currency"
GROUP_BY_STATUS = "status"
_VALID_GROUP_BY = frozenset({GROUP_BY_SENDER, GROUP_BY_CURRENCY, GROUP_BY_STATUS})


# ─── Tool class ──────────────────────────────────────────────────────


class AggregateAmounts:
    """Tool: extract and aggregate monetary amounts across emails."""

    NAME = "aggregate_amounts"
    DESCRIPTION = (
        "Extract monetary amounts from a set of emails (regex over the "
        "clean body text) and aggregate them by sender, currency, or "
        "status. Sums are computed with Decimal arithmetic and "
        "currencies are never mixed. Use whenever the user's question "
        "expects a numerical answer about money."
    )

    def __init__(
        self,
        cache: EmailCache,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._cache = cache
        self._llm = llm_client

    def __call__(
        self,
        *,
        email_ids: list[str],
        group_by: str = GROUP_BY_SENDER,
        validate_with_llm: bool = False,
    ) -> AggregateResult:
        """Aggregate amounts found in the given emails.

        Args:
            email_ids: IDs of emails to scan (typically the result of
                a prior ``search_emails`` call).
            group_by: ``"sender"``, ``"currency"``, or ``"status"``.
                Grouping is always within a currency — different
                currencies stay in separate :class:`Group` rows.
            validate_with_llm: when True (and an ``llm_client`` was
                supplied), filter regex matches through an LLM batch
                check that rejects invoice numbers, postal codes,
                dates that match the number shape, and similar
                false positives. Default ``False`` so callers don't
                spend tokens unintentionally.

        Returns:
            :class:`AggregateResult` with both the grouped totals and
            the underlying matches (so the response writer can cite
            specific invoices).
        """
        if group_by not in _VALID_GROUP_BY:
            raise ValueError(
                f"Unknown group_by={group_by!r}. "
                f"Expected one of {sorted(_VALID_GROUP_BY)}."
            )

        emails = [
            self._cache.get(eid) for eid in email_ids
        ]
        matches: list[AmountMatch] = []
        for email in emails:
            if email is None:
                continue
            matches.extend(_extract_matches(email))

        if validate_with_llm and self._llm is not None and matches:
            matches = self._validate_with_llm(matches)

        groups = _group_matches(matches, group_by)
        return AggregateResult(
            groups=groups,
            matches=matches,
            group_by=group_by,
        )

    # ── LLM validation ──────────────────────────────────────────────

    def _validate_with_llm(self, matches: list[AmountMatch]) -> list[AmountMatch]:
        """Filter out matches that the LLM judges to be non-monetary.

        One batched call; the model returns ``YES``/``NO`` per line in
        the same order as the input. If the response is shorter than
        expected we keep all matches (fail-open: never lose data the
        regex was confident about).
        """
        items = "\n".join(
            f"{i + 1}. \"{m.raw_text}\" — context: amount in email from {m.sender}"
            for i, m in enumerate(matches)
        )
        prompt = (
            "For each detected amount below, reply YES if it's a real "
            "monetary value (price, invoice total, balance, payment) or "
            "NO if it's actually a non-monetary number that happens to "
            "match the format (invoice number, postal code, phone "
            "fragment, date, ID).\n\n"
            "Reply with exactly one YES or NO per line, in the same "
            "order, no extra text.\n\n"
            f"Amounts:\n{items}"
        )

        assert self._llm is not None
        result = self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            role=Role.GENERATION,
            max_tokens=max(len(matches) * 4, 32),
            temperature=0.0,
        )
        verdicts = [
            line.strip().upper()
            for line in result.text.splitlines()
            if line.strip()
        ]
        if len(verdicts) < len(matches):
            # Fail-open: trust the regex when the LLM was vague.
            return matches
        return [m for m, v in zip(matches, verdicts, strict=False) if v.startswith("Y")]

    # ── Schema ──────────────────────────────────────────────────────

    @classmethod
    def input_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "email_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Email IDs to scan; typically from a prior search_emails.",
                },
                "group_by": {
                    "type": "string",
                    "enum": [GROUP_BY_SENDER, GROUP_BY_CURRENCY, GROUP_BY_STATUS],
                    "default": GROUP_BY_SENDER,
                },
                "validate_with_llm": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Batch-validate regex matches with the LLM to drop "
                        "false positives (invoice numbers, postal codes, …)."
                    ),
                },
            },
            "required": ["email_ids"],
        }


# ─── Extraction helpers ──────────────────────────────────────────────


def _extract_matches(email: Email) -> list[AmountMatch]:
    """Run regex + status detection over a single email's clean body.

    Within a single email we dedupe by ``(amount, currency)``: invoice
    reminders routinely mention the same total in the body and again in
    the details footer, and we want to count it once. Two distinct
    invoices in the same email that happen to be the same amount is
    rare enough not to design around in v0.
    """
    sender = extract_email_address(email.from_)
    body = email.body_clean
    if not body:
        return []

    seen: set[tuple[Decimal, str]] = set()
    out: list[AmountMatch] = []
    for m in _AMOUNT_RE.finditer(body):
        amount_str, currency = _resolve_match(m)
        if amount_str is None or currency is None:
            continue
        try:
            amount = _to_decimal(amount_str)
        except ValueError:
            continue

        key = (amount, currency)
        if key in seen:
            continue
        seen.add(key)

        status = _detect_status(body, m.start(), m.end())
        out.append(
            AmountMatch(
                email_id=email.id,
                sender=sender,
                amount=amount,
                currency=currency,
                raw_text=m.group(0).strip(),
                status=status,
            )
        )
    return out


def _resolve_match(m: re.Match[str]) -> tuple[str | None, str | None]:
    """Pick the right amount string + currency code from a regex match."""
    if m.group("amount_a") is not None:
        amount = m.group("amount_a")
        # Pattern A prefers a trailing explicit code, falls back to symbol.
        code = m.group("code_a")
        if code:
            return amount, code.upper()
        symbol = m.group("symbol_a")
        return amount, _SYMBOL_TO_CODE.get(symbol, "USD")

    if m.group("amount_b") is not None:
        return m.group("amount_b"), m.group("code_b").upper()

    if m.group("amount_c") is not None:
        return m.group("amount_c"), m.group("code_c").upper()

    return None, None


def _to_decimal(raw: str) -> Decimal:
    """Parse a US-format number string into a Decimal.

    Strips thousands separators (``,``) and uses ``.`` as the decimal
    point. European-format strings (``1.200,00``) are not supported
    in v0 — they'd need a separate pattern and locale detection.
    """
    cleaned = raw.replace(",", "")
    return Decimal(cleaned)


def _detect_status(body: str, start: int, end: int) -> str | None:
    """Find the most likely status keyword near a match.

    Scans a window of ±150 chars around the match. First keyword wins;
    falls back to scanning the whole body if the window is empty.
    """
    window_start = max(0, start - 150)
    window_end = min(len(body), end + 150)
    window = body[window_start:window_end].lower()

    for status, keywords in _STATUS_KEYWORDS.items():
        if any(kw in window for kw in keywords):
            return status
    return None


def _group_matches(matches: list[AmountMatch], group_by: str) -> list[Group]:
    """Sum matches into groups keyed by ``(group_value, currency)``."""
    totals: dict[tuple[str, str], list[AmountMatch]] = {}
    for m in matches:
        key = _key_for(m, group_by)
        totals.setdefault((key, m.currency), []).append(m)

    groups = [
        Group(
            key=key,
            currency=currency,
            total=sum((m.amount for m in items), start=Decimal("0")),
            count=len(items),
        )
        for (key, currency), items in totals.items()
    ]
    # Largest total first, then alphabetical on key for ties.
    groups.sort(key=lambda g: (-g.total, g.key))
    return groups


def _key_for(m: AmountMatch, group_by: str) -> str:
    if group_by == GROUP_BY_SENDER:
        return m.sender or "(unknown sender)"
    if group_by == GROUP_BY_CURRENCY:
        return m.currency
    if group_by == GROUP_BY_STATUS:
        return m.status or "(unknown status)"
    raise ValueError(f"Unhandled group_by={group_by!r}")  # pragma: no cover
