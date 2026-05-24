"""Tests for ``aone.agent.tools.amounts`` (AONE-406)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from aone.agent.tools.amounts import (
    AggregateAmounts,
    AggregateResult,
    AmountMatch,
    Group,
)
from aone.gmail.types import Email
from aone.llm.client import CompletionResult, LLMClient
from aone.storage.cache import EmailCache


def _email(
    id_: str,
    *,
    body: str,
    from_: str = "billing@acme.com",
    date: int = 1_700_000_000_000,
) -> Email:
    return Email(
        id=id_,
        thread_id=f"t-{id_}",
        from_=from_,
        to=["axel@example.com"],
        subject="invoice",
        body_text=body,
        body_html=f"<p>{body}</p>",
        body_clean=body,  # tests assume normalize already ran
        snippet=body[:80],
        internal_date=date,
        labels=["INBOX"],
    )


def _cache(*emails: Email) -> EmailCache:
    cache = EmailCache()
    cache.add_many(list(emails))
    return cache


def _run(*emails: Email, group_by: str = "sender") -> AggregateResult:
    cache = _cache(*emails)
    tool = AggregateAmounts(cache)
    return tool(email_ids=[e.id for e in emails], group_by=group_by)


# ─── Regex: symbol prefix ────────────────────────────────────────────


def test_extracts_dollar_with_thousands_and_decimals() -> None:
    result = _run(_email("m", body="Invoice total: $1,200.00 USD"))
    assert len(result.matches) == 1
    m = result.matches[0]
    assert m.amount == Decimal("1200.00")
    assert m.currency == "USD"


def test_extracts_euro_symbol() -> None:
    result = _run(_email("m", body="Importe: €500"))
    m = result.matches[0]
    assert m.amount == Decimal("500")
    assert m.currency == "EUR"


def test_extracts_pound_symbol() -> None:
    result = _run(_email("m", body="Cost: £2,500.50"))
    m = result.matches[0]
    assert m.amount == Decimal("2500.50")
    assert m.currency == "GBP"


def test_bare_dollar_defaults_to_usd_when_no_explicit_code() -> None:
    result = _run(_email("m", body="Cost: $42"))
    m = result.matches[0]
    assert m.currency == "USD"
    assert m.amount == Decimal("42")


# ─── Regex: code prefix / suffix ─────────────────────────────────────


def test_extracts_code_prefix_format() -> None:
    result = _run(_email("m", body="Total MXN 5,400.00"))
    m = result.matches[0]
    assert m.amount == Decimal("5400.00")
    assert m.currency == "MXN"


def test_extracts_code_suffix_format() -> None:
    result = _run(_email("m", body="Please pay 1,200 USD by Friday"))
    m = result.matches[0]
    assert m.amount == Decimal("1200")
    assert m.currency == "USD"


def test_code_prefix_takes_precedence_over_dollar_symbol() -> None:
    """``USD $1,200`` — both are present, the code wins."""
    result = _run(_email("m", body="Total: USD $1,200"))
    m = result.matches[0]
    assert m.currency == "USD"
    assert m.amount == Decimal("1200")


# ─── Negatives (false positives we must NOT match) ───────────────────


def test_does_not_match_bare_integers() -> None:
    """Invoice numbers like '1024' must not be parsed as money."""
    result = _run(_email("m", body="Invoice #1024 for your records"))
    assert result.matches == []


def test_does_not_match_dates() -> None:
    result = _run(_email("m", body="Due date: 2026-06-13"))
    assert result.matches == []


def test_deduplicates_repeated_amounts_within_a_single_email() -> None:
    """Invoice reminders mention the same total twice (body + details);
    we count it once."""
    body = (
        "Friendly reminder: invoice #1024 for $1,200.00 USD is overdue.\n"
        "Amount due: $1,200.00 USD\n"
        "Status: OVERDUE"
    )
    result = _run(_email("m", body=body))
    assert len(result.matches) == 1
    assert result.matches[0].amount == Decimal("1200.00")


def test_dedupe_is_per_email_not_global() -> None:
    """Two emails each mentioning $500 should count as two distinct amounts."""
    result = _run(
        _email("m1", body="Total: $500.00 USD", from_="a@x.com"),
        _email("m2", body="Total: $500.00 USD", from_="b@y.com"),
    )
    assert len(result.matches) == 2


# ─── Status detection ────────────────────────────────────────────────


def test_detects_paid_status_in_english() -> None:
    m = _run(_email("m", body="Invoice #99 for $3,500.00 — PAID")).matches[0]
    assert m.status == "paid"


def test_detects_overdue_status_in_spanish() -> None:
    m = _run(_email("m", body="Factura $1,200.00 USD — vencida hace 15 días")).matches[0]
    assert m.status == "overdue"


def test_detects_pending_status() -> None:
    m = _run(_email("m", body="$750 due by next Friday")).matches[0]
    assert m.status == "pending"


def test_status_is_none_when_no_keyword_present() -> None:
    m = _run(_email("m", body="Amount: $100")).matches[0]
    assert m.status is None


# ─── Aggregation ─────────────────────────────────────────────────────


def test_groups_by_sender_and_sums_in_decimal() -> None:
    result = _run(
        _email("a1", body="Invoice #1024 for $1,200.00 USD", from_="Acme <billing@acme.com>"),
        _email("a2", body="Invoice #1031 for $1,500.00 USD", from_="billing@acme.com"),
        _email("a3", body="Invoice #1042 for $750.00 USD", from_="Acme Corp <billing@acme.com>"),
        _email("b1", body="Invoice for $3,500.00 USD", from_="ar@beta.com"),
    )

    by_sender = {(g.key, g.currency): g for g in result.groups}
    acme = by_sender[("billing@acme.com", "USD")]
    assert acme.total == Decimal("3450.00")
    assert acme.count == 3

    beta = by_sender[("ar@beta.com", "USD")]
    assert beta.total == Decimal("3500.00")
    assert beta.count == 1


def test_does_not_mix_currencies_in_the_same_group() -> None:
    result = _run(
        _email("m1", body="$1,000.00 USD", from_="a@x.com"),
        _email("m2", body="€500 EUR", from_="a@x.com"),
    )
    groups = {(g.key, g.currency): g for g in result.groups}
    assert groups[("a@x.com", "USD")].total == Decimal("1000.00")
    assert groups[("a@x.com", "EUR")].total == Decimal("500")


def test_grand_total_by_currency_sums_across_senders() -> None:
    result = _run(
        _email("m1", body="$100 USD", from_="a@x.com"),
        _email("m2", body="$200 USD", from_="b@y.com"),
        _email("m3", body="€50 EUR", from_="c@z.com"),
    )
    totals = result.grand_total_by_currency
    assert totals["USD"] == Decimal("300")
    assert totals["EUR"] == Decimal("50")


def test_groups_sorted_largest_total_first() -> None:
    result = _run(
        _email("m1", body="$100 USD", from_="small@x.com"),
        _email("m2", body="$1000 USD", from_="big@x.com"),
        _email("m3", body="$500 USD", from_="medium@x.com"),
    )
    keys_in_order = [g.key for g in result.groups]
    assert keys_in_order == ["big@x.com", "medium@x.com", "small@x.com"]


def test_groups_by_currency() -> None:
    result = _run(
        _email("m1", body="$100 USD", from_="a@x.com"),
        _email("m2", body="$200 USD", from_="b@y.com"),
        _email("m3", body="€50 EUR", from_="c@z.com"),
        group_by="currency",
    )
    by_currency = {g.key: g for g in result.groups}
    assert by_currency["USD"].total == Decimal("300")
    assert by_currency["EUR"].total == Decimal("50")


def test_groups_by_status() -> None:
    result = _run(
        _email("m1", body="$500 USD — PAID"),
        _email("m2", body="$1000 USD — pending"),
        _email("m3", body="$200 USD — vencida"),
        _email("m4", body="$300 USD pending"),
        group_by="status",
    )
    by_status = {g.key: g for g in result.groups}
    assert by_status["paid"].total == Decimal("500")
    assert by_status["pending"].total == Decimal("1300")
    assert by_status["overdue"].total == Decimal("200")


def test_unknown_email_ids_are_skipped_silently() -> None:
    cache = _cache(_email("m1", body="$100 USD"))
    tool = AggregateAmounts(cache)
    result = tool(email_ids=["m1", "missing"])
    assert len(result.matches) == 1


def test_unknown_group_by_raises() -> None:
    with pytest.raises(ValueError, match="group_by"):
        AggregateAmounts(EmailCache())(email_ids=[], group_by="banana")


def test_empty_email_list_returns_empty_result() -> None:
    result = AggregateAmounts(EmailCache())(email_ids=[])
    assert result.matches == []
    assert result.groups == []


# ─── LLM validation ──────────────────────────────────────────────────


def _fake_llm(reply: str) -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = CompletionResult(
        text=reply,
        model="groq/llama-3.3-70b-versatile",
        prompt_tokens=50,
        completion_tokens=10,
    )
    return client


def test_llm_validation_drops_false_positives() -> None:
    cache = _cache(
        _email("m1", body="$1,200.00 USD"),
        _email("m2", body="$1024 USD"),  # invoice-number-shaped
        _email("m3", body="$50 USD"),
    )
    fake = _fake_llm("YES\nNO\nYES")
    tool = AggregateAmounts(cache, llm_client=fake)

    result = tool(email_ids=["m1", "m2", "m3"], validate_with_llm=True)

    assert len(result.matches) == 2
    assert {m.email_id for m in result.matches} == {"m1", "m3"}


def test_llm_validation_off_by_default_keeps_all_matches() -> None:
    cache = _cache(_email("m", body="$1024 USD"))
    fake = _fake_llm("NO")
    tool = AggregateAmounts(cache, llm_client=fake)

    result = tool(email_ids=["m"])  # default validate_with_llm=False
    fake.complete.assert_not_called()
    assert len(result.matches) == 1


def test_llm_validation_skipped_when_no_client_supplied() -> None:
    cache = _cache(_email("m", body="$1024 USD"))
    tool = AggregateAmounts(cache)  # no LLM client
    result = tool(email_ids=["m"], validate_with_llm=True)
    assert len(result.matches) == 1


def test_llm_validation_fails_open_on_short_response() -> None:
    """If the LLM replies with fewer lines than matches, keep all matches."""
    cache = _cache(
        _email("m1", body="$100 USD"),
        _email("m2", body="$200 USD"),
        _email("m3", body="$300 USD"),
    )
    fake = _fake_llm("YES\n")  # only one verdict for three matches
    tool = AggregateAmounts(cache, llm_client=fake)

    result = tool(email_ids=["m1", "m2", "m3"], validate_with_llm=True)
    assert len(result.matches) == 3


# ─── Schema ──────────────────────────────────────────────────────────


def test_input_schema_requires_email_ids() -> None:
    schema = AggregateAmounts.input_schema()
    assert schema["required"] == ["email_ids"]
    assert schema["properties"]["email_ids"]["type"] == "array"
    assert schema["properties"]["group_by"]["enum"] == [
        "sender", "currency", "status"
    ]


def test_amount_match_and_group_are_frozen_dataclasses() -> None:
    m = AmountMatch(
        email_id="m",
        sender="a@x.com",
        amount=Decimal("10"),
        currency="USD",
        raw_text="$10",
        status=None,
    )
    g = Group(key="a@x.com", currency="USD", total=Decimal("10"), count=1)
    with pytest.raises(Exception):
        m.amount = Decimal("20")  # type: ignore[misc]
    with pytest.raises(Exception):
        g.total = Decimal("20")  # type: ignore[misc]
