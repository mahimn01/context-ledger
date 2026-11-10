from __future__ import annotations

import pytest

from context_ledger.model import Citation, Claim, SummarySentence
from context_ledger.verify import Verdict, tier0, tier1, value_tokens, verify_claim
from context_ledger.window import replay

CONFIG = "DEFAULT_TIMEOUT = 37\nMAX_RETRIES = 5" + "  # retry budget per request" * 5 + "\ndef pair(a,\n         b):\n    return (a, b)\n"


def claim(text, *citations):
    return Claim("claim_t", "call_t", text, tuple(citations), "ungrounded")


@pytest.fixture
def world(builder):
    """Spans: timeout (verbatim), retries (folded), pair (verbatim), plus the builder."""
    src = builder.source("config.py", CONFIG)
    timeout = builder.span(src, 0, 20)
    retries = builder.span(src, 21, CONFIG.index("\ndef"))  # long enough that folding it frees tokens
    pair = builder.span(src, CONFIG.index("def"), len(CONFIG) - 1)
    builder.append(timeout)
    retries_seg = builder.append(retries)
    builder.append(pair)
    fold = builder.fold([retries_seg.segment_id], [SummarySentence("retry setting", (retries.span_id,))])
    builder.call()
    return builder, timeout, retries, pair, fold


def test_tier0_pass_on_direct_verbatim_cite(world):
    b, timeout, *_ = world
    v = tier0(claim("x", Citation(timeout.span_id, "DEFAULT_TIMEOUT = 37")), replay(b.events()))
    assert v == Verdict(0, True, "accept", "verbatim_cited", (timeout.span_id,))


def test_tier0_refetch_for_summary_only(world):
    b, _, retries, _, fold = world
    v = tier0(claim("MAX_RETRIES is 5", Citation(f"fold:{fold.fold_id}#0")), replay(b.events()))
    assert (v.passed, v.action, v.support) == (False, "refetch", (retries.span_id,))


def test_tier0_refetch_for_dropped_and_stale(world):
    b, timeout, _, pair, _ = world
    b.evict([s.segment_id for s in b.state.segments if s.ref == pair.span_id])
    v = tier0(claim("x", Citation(pair.span_id)), replay(b.events()))
    assert (v.action, v.support) == ("refetch", (pair.span_id,))
    b.update(b.state.sources[timeout.source_id], CONFIG.replace("37", "41"))
    v = tier0(claim("x", Citation(timeout.span_id, "DEFAULT_TIMEOUT = 37")), replay(b.events()))
    assert (v.action, v.support) == ("refetch", (timeout.span_id,))


def test_tier0_verbatim_reached_only_through_fold_is_not_enough(world):
    b, _, retries, _, fold = world
    b.append(retries)
    v = tier0(claim("x", Citation(f"fold:{fold.fold_id}#0")), replay(b.events()))
    assert (v.passed, v.action, v.support) == (False, "refetch", ())


@pytest.mark.parametrize(
    ("citations", "reason"),
    [((), "no_citations"), ((Citation("inference"),), "no_span_support"), ((Citation("raw:seg_x"), Citation("inference")), "no_span_support")],
)
def test_tier0_flag(world, citations, reason):
    b = world[0]
    v = tier0(claim("x", *citations), replay(b.events()))
    assert (v.passed, v.action, v.reason) == (False, "flag", reason)


def test_tier1_accepts_real_quote(world):
    b, timeout, *_ = world
    v = tier1(claim("DEFAULT_TIMEOUT is 37", Citation(timeout.span_id, "DEFAULT_TIMEOUT = 37")), replay(b.events()), b.blobs)
    assert v == Verdict(1, True, "accept", "quotes_verified", (timeout.span_id,))


def test_tier1_nfkc_full_width_digits(world):
    b, timeout, *_ = world
    quote = "ＤＥＦＡＵＬＴ＿ＴＩＭＥＯＵＴ ＝ ３７"
    v = tier1(claim("The timeout is ３７ seconds.", Citation(timeout.span_id, quote)), replay(b.events()), b.blobs)
    assert v.passed


def test_tier1_whitespace_folding(world):
    b, _, _, pair = world[:4]
    quote = "def pair(a, b):   return (a, b)"
    v = tier1(claim("`pair` returns `(a, b)`", Citation(pair.span_id, quote)), replay(b.events()), b.blobs)
    assert v.passed


def test_tier1_minimum_quote_length(world):
    b, timeout, *_ = world
    v = tier1(claim("It is 37", Citation(timeout.span_id, "= 37")), replay(b.events()), b.blobs)
    assert (v.passed, v.action, v.reason) == (False, "refetch", "no_valid_quote")


def test_tier1_value_coverage_failure(world):
    b, timeout, *_ = world
    v = tier1(claim("DEFAULT_TIMEOUT is 41", Citation(timeout.span_id, "DEFAULT_TIMEOUT = 37")), replay(b.events()), b.blobs)
    assert (v.passed, v.action, v.reason) == (False, "refetch", "value_not_covered")
    v = tier1(claim("DEFAULT_TIMEOUT is 3", Citation(timeout.span_id, "DEFAULT_TIMEOUT = 37")), replay(b.events()), b.blobs)
    assert not v.passed  # "3" must not match inside "37"


def test_tier1_quote_not_in_span(world):
    b, timeout, *_ = world
    v = tier1(claim("x", Citation(timeout.span_id, "DEFAULT_TIMEOUT = 99 seconds")), replay(b.events()), b.blobs)
    assert (v.passed, v.reason) == (False, "no_valid_quote")


def test_tier1_requires_a_quote(world):
    b, timeout, *_ = world
    v = tier1(claim("x", Citation(timeout.span_id)), replay(b.events()), b.blobs)
    assert (v.passed, v.support) == (False, (timeout.span_id,))


def test_verify_claim_runs_tier1_only_after_tier0(world):
    b, timeout, _, _, fold = world
    state = replay(b.events())
    assert verify_claim(claim("x", Citation(f"fold:{fold.fold_id}#0")), state, b.blobs).tier == 0
    good = claim("DEFAULT_TIMEOUT is 37", Citation(timeout.span_id, "DEFAULT_TIMEOUT = 37"))
    assert verify_claim(good, state, b.blobs) == Verdict(1, True, "accept", "quotes_verified", (timeout.span_id,))


def test_value_tokens():
    text = "`retry()` uses DEFAULT_TIMEOUT = 37, os.path and fetch() with 'quoted text' and \"x y\"; it's -2.5 not sha256"
    assert value_tokens(text) == ["retry()", "DEFAULT_TIMEOUT", "37", "os.path", "fetch", "quoted text", "x y", "-2.5"]
    assert value_tokens("The answer is plain words.") == []


def test_verdict_round_trip():
    v = Verdict(1, False, "refetch", "value_not_covered", ("a", "b"))
    assert Verdict.from_dict(v.to_dict()) == v
    with pytest.raises(ValueError):
        Verdict(0, False, "maybe", "x")
