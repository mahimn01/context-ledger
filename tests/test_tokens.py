"""A.2: versioned token estimators, the tokenizer_set event, and invariants under both estimators."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant

from context_ledger.fold import BackstopPolicyV1
from context_ledger.ids import canonical_json
from context_ledger.invariants import check
from context_ledger.model import EVENT_TYPES
from context_ledger.render import rendered_window_tokens
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.tokens import (
    BYTES4_V0,
    CLASSES,
    CalibEstimator,
    CalibrationMissing,
    ClassFit,
    EstimatorError,
    calibration_estimator,
    decimal_str,
    estimator_for,
    estimator_or_fallback,
    from_payload,
    load_calibration,
    segment_class,
    source_class,
)
from context_ledger.window import manifest, replay, window_tokens
from m2util import MODULE, MODULE_PATH, index_of, mutate, pricing, props
from test_invariants_property_m2 import BUDGET, ForgivingSession, LedgerMachineM2

SHA = "ab" * 32
FITS = {
    "python": ClassFit(Fraction("0.30"), Fraction(2), Fraction("0.33")),
    "prose": ClassFit(Fraction("0.27"), Fraction(1), Fraction("0.29")),
    "json": ClassFit(Fraction("0.35"), Fraction(3), Fraction("0.4")),
    "other": ClassFit(Fraction("0.28"), Fraction(0), Fraction("0.31")),
}
CALIB = CalibEstimator("claude-haiku-5-5", FITS, SHA)


def test_bytes4_matches_m2():
    assert [BYTES4_V0.count(t) for t in ["", "a", "abcd", "abcde", "é"]] == [0, 1, 1, 2, 1]
    assert BYTES4_V0.count("abcde", "python") == BYTES4_V0.count("abcde", "json") == 2
    assert BYTES4_V0.name == "approx-bytes4" and BYTES4_V0.payload()["estimator"] == "bytes4-v0"


def test_calib_counts_conservatively_and_estimates_with_the_fit():
    assert CALIB.count("", "python") == 0
    assert CALIB.count_bytes(1000, "python") == 332  # ceil(0.33 * 1000 + 2)
    assert CALIB.estimate_bytes(1000, "python") == 302  # ceil(0.30 * 1000 + 2)
    assert CALIB.count_bytes(1, "other") == 1  # at least one token for non-empty text
    assert CALIB.count_bytes(1000, "unknown") == CALIB.count_bytes(1000, "other")
    for cls in CLASSES:
        assert all(CALIB.count_bytes(n, cls) >= CALIB.estimate_bytes(n, cls) for n in range(0, 5000, 37))


def test_payload_round_trip_and_strictness():
    payload = CALIB.payload()
    assert payload["coefficients"]["python"] == {"a": "0.3", "b": "2", "a_hi": "0.33"}
    assert from_payload(json.loads(canonical_json(payload))) == CALIB
    assert from_payload(None) is BYTES4_V0 and from_payload(BYTES4_V0.payload()) is BYTES4_V0
    bad = [
        {**payload, "extra": 1},
        {**payload, "estimator": "tiktoken"},
        {**payload, "version": 2},
        {**payload, "calibration_sha256": "nothex"},
        {**payload, "coefficients": {k: v for k, v in payload["coefficients"].items() if k != "json"}},
        {**payload, "coefficients": {**payload["coefficients"], "python": {"a": "0.30", "b": "2", "a_hi": "0.33"}}},  # not canonical
        {**payload, "coefficients": {**payload["coefficients"], "python": {"a": 0.3, "b": "2", "a_hi": "0.33"}}},
        {**payload, "coefficients": {**payload["coefficients"], "python": {"a": "0.4", "b": "2", "a_hi": "0.33"}}},  # a > a_hi
        {**BYTES4_V0.payload(), "model": "claude-haiku-5-5"},
    ]
    for p in bad:
        with pytest.raises(EstimatorError):
            from_payload(p)


def test_decimal_str():
    assert [decimal_str(Fraction(x)) for x in ("0.3", "2", "-4.5", "1/3", "-1/3", "0", "10")] == ["0.3", "2", "-4.5", "0.333333", "-0.333333", "0", "10"]
    assert decimal_str(Fraction(1, 2_000_000)) == "0.000001" and decimal_str(Fraction(-1, 3_000_000)) == "0"


def test_classes_follow_segment_kind_and_media_type():
    assert [source_class(u) for u in ("Lib/a.py", "Doc/x.rst", "README.txt", "x.md", "Modules/a.c", "a")] == [
        "python", "prose", "prose", "prose", "other", "other"
    ]  # fmt: skip
    assert source_class("noext", "text/x-python") == "python"
    s = Session(Ledger(":memory:"), MemoryBlobStore(), 100_000, BackstopPolicyV1(budget_tokens=100_000), pricing=pricing())
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    span = s.register_python_spans(src, MODULE_PATH)[0]
    seg_span = s.append_span(span.span_id)
    seg_user = s.append_raw("user", "hello")
    seg_blocks = s.append_blocks([{"type": "tool_use", "id": "t1", "name": "x", "input": {}}])
    seg_tool = s.append_raw("tool", "result", tool_use_id="t1")
    assert [segment_class(s.state, g) for g in (seg_span, seg_user, seg_blocks, seg_tool)] == ["python", "other", "json", "json"]


def calib_session(budget: int = 100_000, ledger=None, blobs=None, **policy) -> Session:
    return Session(
        ledger if ledger is not None else Ledger(":memory:"),
        blobs if blobs is not None else MemoryBlobStore(),
        budget,
        BackstopPolicyV1(budget_tokens=budget, token_counter="calib-v1", **policy),
        pricing=pricing(),
        tokenizer=CALIB,
    )


def test_tokenizer_set_is_recorded_once_and_replayed():
    ledger, blobs = Ledger(":memory:"), MemoryBlobStore()
    s = calib_session(ledger=ledger, blobs=blobs)
    assert [e.type for e in s.events] == ["pricing_set", "tokenizer_set"] and "tokenizer_set" in EVENT_TYPES
    assert estimator_for(replay(s.events)) == CALIB and s.counter == CALIB
    again = calib_session(ledger=ledger, blobs=blobs)
    assert len(again.events) == 2
    src = again.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    for span in again.register_python_spans(src, MODULE_PATH)[:3]:
        again.append_span(span.span_id)
    state = replay(again.events)
    assert rendered_window_tokens(state, blobs) == rendered_window_tokens(again.state, blobs)
    assert rendered_window_tokens(state, blobs) > rendered_window_tokens(state, blobs, BYTES4_V0)  # 0.33 > 1/4 per byte
    assert window_tokens(state, blobs) != window_tokens(replay([e for e in again.events if e.type != "tokenizer_set"]), blobs)
    assert not any(isinstance(v, float) for v in json.loads(canonical_json(s.events[1].payload))["coefficients"]["python"].values())


def test_policy_must_name_the_estimator_in_force():
    with pytest.raises(ValueError, match="not the estimator in force"):
        Session(Ledger(":memory:"), MemoryBlobStore(), 1000, BackstopPolicyV1(budget_tokens=1000), pricing=pricing(), tokenizer=CALIB)
    with pytest.raises(ValueError, match="not the estimator in force"):
        Session(Ledger(":memory:"), MemoryBlobStore(), 1000, BackstopPolicyV1(budget_tokens=1000, token_counter="calib-v1"), pricing=pricing())


def filled(session: Session, n: int = 14) -> Session:
    session.append_raw("system", "Answer from spans.", pinned=True)
    src = session.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = session.register_python_spans(src, MODULE_PATH)
    for k in range(n):
        session.append_span(spans[k % len(spans)].span_id) if k < len(spans) else session.append_raw("user", f"note {k} " * 20)
        session.call()
    return session


def test_backstop_folds_and_checks_clean_under_calib():
    s = filled(calib_session(budget=900))
    folds = [e for e in s.events if e.type == "fold"]
    assert folds and all(e.payload["fold"]["policy"]["params"]["token_counter"] == "calib-v1" for e in folds)
    assert check(s.events, s.blobs) == []


def test_tampering_with_the_estimator_breaks_p9_and_p12():
    s = filled(calib_session(budget=900))
    i = index_of(s.events, "tokenizer_set")

    def cheaper(p):
        p["coefficients"]["python"]["a_hi"] = "0.5"

    assert {"P9", "P12"} & props(mutate(s.events, i, cheaper), s.blobs)
    dropped = [e for k, e in enumerate(s.events) if k != i]
    from context_ledger.store import rechain

    assert "P9" in props(rechain(dropped), s.blobs)  # the folds name calib-v1, the ledger has bytes4-v0


def test_missing_calibration_fails_clearly_and_falls_back(tmp_path):
    with pytest.raises(CalibrationMissing, match="no calib-v1 calibration"):
        load_calibration("claude-haiku-5-5", tmp_path)
    estimator, warning = estimator_or_fallback("claude-haiku-5-5", tmp_path)
    assert estimator is BYTES4_V0 and warning.startswith("warning:") and "bytes4-v0" in warning
    data = {"model": "claude-haiku-5-5", "counter": "fake", "coefficients": {c: FITS[c].to_dict() for c in CLASSES}}
    (tmp_path / "claude-haiku-5-5-20261031.json").write_text(json.dumps(data))
    with pytest.raises(CalibrationMissing, match="fake"):
        load_calibration("claude-haiku-5-5", tmp_path)
    loaded = load_calibration("claude-haiku-5-5", tmp_path, allow_fake=True)
    assert loaded.coefficients == FITS and loaded == calibration_estimator(data, allow_fake=True)


@settings(max_examples=60, deadline=None)
@given(n=st.integers(0, 50_000), a=st.integers(1, 900), extra=st.integers(0, 200), b=st.integers(-50, 50))
def test_conservative_count_is_monotone(n, a, extra, b):
    fit = ClassFit(Fraction(a, 1000), Fraction(b), Fraction(a + extra, 1000))
    est = CalibEstimator("m", {c: fit for c in CLASSES}, SHA)
    assert est.count_bytes(n + 1) >= est.count_bytes(n) >= est.estimate_bytes(n) >= 0


class LedgerMachineCalib(LedgerMachineM2):
    """The M2 random-session machine with calib-v1 in force: every invariant (P12 and P16
    included) must still hold after every step, and replay must rebuild the window exactly."""

    def __init__(self) -> None:
        RuleBasedStateMachine.__init__(self)
        self.ledger = Ledger(":memory:")
        self.blobs = MemoryBlobStore()
        policy = BackstopPolicyV1(budget_tokens=BUDGET, cold_m=1, tail_keep=1, q0_bp=500, tau_tokens=20, token_counter="calib-v1")
        self.session = ForgivingSession(self.ledger, self.blobs, BUDGET, policy, pricing=pricing(), tokenizer=CALIB)
        self.session.append_raw("system", "Answer only from cited spans.", pinned=True)
        self.sources = []

    @invariant()
    def replay_is_identical(self):
        state = replay(self.ledger.events())
        assert manifest(state) == manifest(self.state) and state.tokenizer == self.state.tokenizer
        assert rendered_window_tokens(state, self.blobs) == rendered_window_tokens(self.state, self.blobs)


LedgerMachineCalib.TestCase.settings = settings(max_examples=40, stateful_step_count=30, deadline=None)
TestLedgerInvariantsCalib = LedgerMachineCalib.TestCase
