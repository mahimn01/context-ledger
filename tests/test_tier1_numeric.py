from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from context_ledger.anchor.text import normalize_for_match
from context_ledger.model import Citation, Claim
from context_ledger.verify import numeric_value, tier1, value_tokens_typed, values_covered
from context_ledger.window import replay


def covers(claim: str, quote: str) -> bool:
    return values_covered(normalize_for_match(claim), normalize_for_match(quote))


@pytest.mark.parametrize(
    ("claim", "quote"),
    [
        ("37", "DEFAULT_TIMEOUT = 37.0"),
        ("37.0", "DEFAULT_TIMEOUT = 37"),
        ("1,000", "limit 1000 rows"),
        ("1000", "limit 1,000 rows"),
        ("1_000", "limit 1000"),
        ("12,345,678", "n = 12345678"),
        ("300", "wait 3e2"),
        ("300", "wait 3E+2 ms"),
        ("+300", "300"),
        ("1e3", "1000"),
        ("-5", "offset = -5"),
        ("300", "timeout 300s"),
        ("2026-10-31", "released on 2026-10-31."),
        ("3.15.0", "python 3.15.0 is required"),
        ("version 3.15.0rc1", "we ship 3.15.0rc1 today"),
        ("３７", "37"),
    ],
)
def test_numeric_equivalences_match(claim, quote):
    assert covers(claim, quote)


@pytest.mark.parametrize(
    ("claim", "quote"),
    [
        ("3", "DEFAULT_TIMEOUT = 37"),
        ("37", "370"),
        ("100", "1,00"),
        ("1,00", "100"),
        ("-5", "offset = 5"),
        ("5", "offset = -5"),
        ("3.15", "python 3.15.0"),
        ("15", "python 3.15.0"),
        ("2026", "released 2026-10-31"),
        ("31", "mask 0x1F"),
        ("1000", "1e4"),
        ("0x1F", "31"),
    ],
)
def test_numeric_non_equivalences_do_not_match(claim, quote):
    assert not covers(claim, quote)


def test_token_grammar():
    assert value_tokens_typed("1,00") == [("number", "1"), ("number", "00")]
    assert value_tokens_typed("3.15.0 and 2026-10-31 and 0x1F") == [
        ("exact", "3.15.0"),
        ("exact", "2026-10-31"),
        ("exact", "0x1F"),
    ]
    assert value_tokens_typed("1,000 1_000 -2.5e3 300s") == [
        ("number", "1,000"),
        ("number", "1_000"),
        ("number", "-2.5e3"),
        ("number", "300"),
    ]
    assert value_tokens_typed("sha256 v2 a-5") == [("number", "5")]
    assert numeric_value("3E+2") == numeric_value("+300.00") == Decimal(300)
    # No rounding: values that differ in the 30th digit stay different.
    assert numeric_value("1.00000000000000000000000000001") != numeric_value("1")


def test_tier1_accepts_numeric_variant_of_quoted_value(builder):
    src = builder.source("config.py", "DEFAULT_TIMEOUT = 37.0\n")
    span = builder.span(src, 0, 22)
    builder.append(span)
    builder.call()
    state = replay(builder.events())
    claim = Claim("c", "call", "The timeout is 37 seconds.", (Citation(span.span_id, "DEFAULT_TIMEOUT = 37.0"),), "ungrounded")
    assert tier1(claim, state, builder.blobs).passed
    wrong = Claim("c", "call", "The timeout is 3 seconds.", (Citation(span.span_id, "DEFAULT_TIMEOUT = 37.0"),), "ungrounded")
    assert tier1(wrong, state, builder.blobs).reason == "value_not_covered"


@settings(max_examples=200, deadline=None)
@given(st.integers(-10**12, 10**12), st.integers(-10**12, 10**12))
def test_distinct_integers_never_match(a, b):
    if a != b:
        assert not covers(f"The value is {a}.", f"value = {b}")


def variants(d: Decimal) -> list[str]:
    """Formatting variants of one Decimal value that the grammar accepts."""
    out = [str(d), f"{d:e}", f"{d:E}", str(d.normalize()) if d else "0"]
    if d >= 0:
        out.append("+" + str(d))
    if d == d.to_integral_value():
        n = int(d)
        out += [f"{n:,}", f"{n:_}", f"{n}.0", f"{n}.000"]
    return out


@settings(max_examples=200, deadline=None)
@given(st.decimals(min_value=-10**9, max_value=10**9, allow_nan=False, allow_infinity=False, places=4), st.data())
def test_formatting_variants_of_one_value_always_match(d, data):
    forms = variants(d)
    claim = data.draw(st.sampled_from(forms))
    quote = data.draw(st.sampled_from(forms))
    assert covers(f"It is {claim} here", f"x = {quote};")
