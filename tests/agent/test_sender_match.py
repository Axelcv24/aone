"""Tests for ``aone.agent.sender_match`` (v0.1.0 bug fix)."""

from __future__ import annotations

from aone.agent.sender_match import extract_sender_filter
from aone.gmail.types import Email
from aone.storage.cache import EmailCache


def _email(id_: str, *, from_: str) -> Email:
    return Email(
        id=id_,
        thread_id=f"t-{id_}",
        from_=from_,
        to=["axel@example.com"],
        subject="s",
        body_text="b",
        body_html="<p>b</p>",
        body_clean="b",
        snippet="b",
        internal_date=1_700_000_000_000,
        labels=["INBOX"],
    )


def _cache(*emails: Email) -> EmailCache:
    cache = EmailCache()
    cache.add_many(list(emails))
    return cache


# ─── Explicit email address in the question ──────────────────────────


def test_picks_up_explicit_email_address() -> None:
    cache = _cache(_email("m", from_="info@mail.levi.com"))
    result = extract_sender_filter(
        "qué correos tengo de info@mail.levi.com",
        cache,
    )
    assert result == "info@mail.levi.com"


def test_lowercases_explicit_email() -> None:
    cache = _cache(_email("m", from_="Info@MAIL.Levi.COM"))
    result = extract_sender_filter(
        "show me messages from Info@MAIL.Levi.COM",
        cache,
    )
    assert result == "info@mail.levi.com"


# ─── Brand / domain token fuzzy matching ─────────────────────────────


def test_matches_brand_in_question_against_known_sender_domain() -> None:
    cache = _cache(
        _email("a", from_="info@mail.levi.com"),
        _email("b", from_="quiksilver@k.quiksilver.com"),
    )

    levi_q = extract_sender_filter("facturas de Levi", cache)
    quik_q = extract_sender_filter("ofertas de quiksilver", cache)

    assert levi_q == "info@mail.levi.com"
    assert quik_q == "quiksilver@k.quiksilver.com"


def test_returns_none_when_brand_not_in_cache() -> None:
    cache = _cache(_email("m", from_="info@mail.levi.com"))
    assert extract_sender_filter("facturas de Nike", cache) is None


def test_returns_none_for_questions_without_sender() -> None:
    cache = _cache(_email("m", from_="info@mail.levi.com"))
    assert extract_sender_filter("¿cuántos correos tengo?", cache) is None
    assert extract_sender_filter("hello", cache) is None


def test_ignores_stop_words_so_no_false_positive() -> None:
    """Common Spanish/English filler ('de', 'the', 'correos', etc.) must not
    accidentally match a sender."""
    cache = _cache(_email("m", from_="alice@de.example.com"))
    # The Spanish 'de' would otherwise match the TLD 'de' — guard
    # against that.
    result = extract_sender_filter("envíame correos de alguien", cache)
    assert result is None


def test_minimum_token_length_avoids_short_token_false_positives() -> None:
    """A 2-char token like 'pe' must not match — we require length ≥ 3."""
    cache = _cache(_email("m", from_="info@pe.example.com"))
    assert extract_sender_filter("dame pe", cache) is None


def test_explicit_email_wins_over_brand_match() -> None:
    """If both an email and a brand keyword appear, the explicit email
    is the more specific signal."""
    cache = _cache(
        _email("a", from_="info@mail.levi.com"),
        _email("b", from_="other@somethingelse.com"),
    )
    result = extract_sender_filter(
        "dame los de Levi pero específicamente de other@somethingelse.com",
        cache,
    )
    assert result == "other@somethingelse.com"


def test_works_with_empty_cache() -> None:
    """No senders to match against → no filter."""
    assert extract_sender_filter("anything", EmailCache()) is None


def test_works_with_empty_question() -> None:
    cache = _cache(_email("m", from_="info@mail.levi.com"))
    assert extract_sender_filter("", cache) is None


# ─── Possessive forms ────────────────────────────────────────────────


def test_strips_possessive_apostrophe_s_to_match_brand() -> None:
    """``Levi's`` (English possessive) must match the ``levi`` domain
    part. This is exactly the bug we caught testing in dev: the
    apostrophe-s broke the equality check."""
    cache = _cache(_email("m", from_="info@mail.levi.com"))
    assert extract_sender_filter("facturas de Levi's", cache) == "info@mail.levi.com"


def test_strips_curly_apostrophe_s_too() -> None:
    """macOS keyboards default to ' (curly apostrophe). Same fix."""
    cache = _cache(_email("m", from_="info@mail.levi.com"))
    assert extract_sender_filter("facturas de Levi’s", cache) == "info@mail.levi.com"
