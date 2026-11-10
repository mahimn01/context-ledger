"""M4 item 5: the window W and fold savings are measured with calib-v2, the calibration's point
slope per piece of the prompt with no per-text intercept, the same way in every arm."""

from __future__ import annotations

import json
import math
from fractions import Fraction

import pytest
from hypothesis import settings

from context_ledger.fold import BackstopPolicyV1
from context_ledger.invariants import check
from context_ledger.render import render_segment
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.tokens import (
    CLASSES,
    CalibEstimator,
    ClassFit,
    MarginalCalibEstimator,
    estimator_for,
    from_payload,
    load_calibration,
    segment_class,
    window_estimator,
    BYTES4_V0,
)
from context_ledger.tools import dispatch
from context_ledger.window import replay
from m2util import pricing
from test_invariants_property_m2 import BUDGET, ForgivingSession
from test_tokens import LedgerMachineCalib

SHA = "cd" * 32


def calib(slopes: dict[str, Fraction], intercept: Fraction = Fraction(40)) -> CalibEstimator:
    return CalibEstimator("claude-haiku-5-5", {c: ClassFit(slopes[c], intercept, slopes[c] * 2) for c in CLASSES}, SHA)


def test_calib_v2_counts_each_piece_by_the_point_slope_without_intercept():
    v1 = calib({c: Fraction(3, 10) for c in CLASSES})
    v2 = window_estimator(v1)
    assert isinstance(v2, MarginalCalibEstimator) and v2.name == v2.estimator == "calib-v2"
    assert v2.coefficients == v1.coefficients and v2.calibration_sha256 == v1.calibration_sha256
    assert [v2.count_bytes(n, "python") for n in (0, 1, 10, 1000)] == [0, 1, 3, 300]
    assert v1.count_bytes(1000, "python") == 640  # ceil(0.6 * 1000 + 40): envelope slope plus intercept
    assert v2.estimate("x" * 1000) == v2.count("x" * 1000) == 300
    assert from_payload(json.loads(json.dumps(v2.payload()))) == v2 and v2.payload()["version"] == 2
    assert window_estimator(BYTES4_V0) is BYTES4_V0 and window_estimator(v2) is v2
    # Summing per-piece calib-v1 counts charges the intercept once per piece; calib-v2 never does.
    pieces = [200] * 10
    assert sum(v1.count_bytes(n) for n in pieces) - v1.count_bytes(sum(pieces)) == 9 * 40 + 9 * 0
    assert sum(v2.count_bytes(n) for n in pieces) == v2.count_bytes(sum(pieces))


# The three backstop folds of the 2026-10-31 W=6000 live smoke run (runs/smoke-haiku-5-5-
# 20261101T010510Z, L/episode-01.sqlite): each victim's rendered (class, bytes), the summary's
# bytes, the freed_tokens calib-v1 recorded, and the actual delta count_tokens measured
# (counterfactual prompt without the fold minus the next call's prompt, L/folds.jsonl).
SMOKE_FOLDS = [
    ([("json", 246), ("python", 397), ("python", 296), ("python", 403), ("python", 210), ("python", 200), ("json", 186)], 686, 2034, 448, 10),
    ([("json", 269), ("python", 200), ("python", 228), ("python", 227), ("python", 263), ("python", 192), ("python", 263), ("json", 241), ("other", 676)],
     1036, 2341, 571, 10),
    # Seven tool blocks: their wrappers and ids are tokens no arm's measure models (about 16% here).
    ([("json", 339), ("json", 339), ("json", 378), ("other", 686), ("other", 1036), ("other", 98), ("json", 308), ("json", 288), ("json", 295), ("other", 744)],
     1081, 2402, 1675, 20),
]  # fmt: skip


@pytest.mark.parametrize(("victims", "summary", "recorded", "actual", "tolerance_pct"), SMOKE_FOLDS)
def test_smoke_folds_net_savings_are_predicted_not_gross(victims, summary, recorded, actual, tolerance_pct):
    v1 = load_calibration("claude-haiku-5-5")  # the live 2026-10-31 calibration
    v2 = window_estimator(v1)

    def delta(est) -> int:
        return sum(est.count_bytes(n, cls) for cls, n in victims) - est.count_bytes(summary, "other")

    assert delta(v1) == recorded  # reproduces the smoke run's prediction: about 2.5x too high
    assert delta(v1) > 1.4 * actual
    predicted = delta(v2)
    assert abs(predicted - actual) * 100 <= actual * tolerance_pct and predicted <= actual * 11 // 10
    assert predicted < sum(v2.count_bytes(n, cls) for cls, n in victims)  # net of the replacement, never gross


# A fake tokenizer: per-class slopes over the pieces of one message, plus its overhead.
SLOPES = {"python": Fraction(10, 27), "prose": Fraction(10, 26), "json": Fraction(2, 5), "other": Fraction(10, 24)}
OVERHEAD = 9


def truth(pieces: list[tuple[str, str]]) -> int:
    """count_tokens of one user message holding these (class, text) pieces."""
    return OVERHEAD + sum(math.ceil(SLOPES[c] * len(t.encode("utf-8"))) for c, t in pieces)


def fitted() -> CalibEstimator:
    """What the calibrator recovers from single-text messages of this tokenizer (test_calibrate
    shows the fit): the slopes, an intercept near zero, an envelope a little above."""
    return CalibEstimator("claude-haiku-5-5", {c: ClassFit(SLOPES[c], Fraction(0), SLOPES[c] * 11 / 10) for c in CLASSES}, SHA)


MODULE = "".join(
    f'def handler_{k}(request, retries={k + 2}, timeout={30 + k}.5):\n    """Handle request kind {k}."""\n'
    f"    if request is None:\n        raise ValueError('no request {k}')\n    return request.send(retries=retries, timeout=timeout)\n\n\n"
    for k in range(9)
)


def window_pieces(session: Session) -> list[tuple[str, str]]:
    state = session.state
    return [(segment_class(state, seg), render_segment(state, seg, session.blobs)) for seg in state.segments]


def session_with(estimator, budget: int = 100_000) -> Session:
    s = Session(Ledger(":memory:"), MemoryBlobStore(), budget, BackstopPolicyV1(budget_tokens=budget, token_counter=estimator.name),
                pricing=pricing(), tokenizer=estimator)  # fmt: skip
    s.append_raw("system", "Answer from cited spans only.", pinned=True)
    src = s.register_source("Lib/handlers.py", MODULE, media_type="text/x-python")
    for span in s.register_python_spans(src, "Lib/handlers.py"):
        s.append_span(span.span_id)
        s.append_raw("user", f"Noted {span.anchor.qualname}.")
        s.call()
        s.emit_answer(f"note-{span.anchor.qualname}", [], abstained=True)  # each note is an answered turn
    return s


def model_fold(s: Session, summary: str) -> str:
    """The fake model folds positions 1..8 with one summary sentence of known length (ctx_fold
    in the mandatory zone, so the payoff gate does not decide)."""
    aliases = s.state.aliases
    targets = [aliases.alias_of(seg.ref if seg.ref_kind == "span" else f"raw:{seg.segment_id}") for seg in s.state.segments[1:9]]
    s.budget_tokens = 1  # every window is in the mandatory zone now: only delta > 0 is required
    s.policy = BackstopPolicyV1(**{**s.policy.__dict__, "budget_tokens": 1})
    sentences = [{"text": summary[k : k + 400], "cites": []} for k in range(0, len(summary), 400)]
    return dispatch(s, "ctx_fold", {"targets": targets, "summary": sentences}, turn_id="t").text


@pytest.mark.parametrize("summary_len", [80, 200, 400])
def test_predicted_net_savings_are_within_ten_percent_of_the_actual_window_delta(summary_len):
    s = session_with(window_estimator(fitted()))
    victims = window_pieces(s)[1:9]
    before = truth(window_pieces(s))
    reply = model_fold(s, "s" * summary_len)
    assert reply.startswith("ok: folded 8 segments"), reply
    fold = next(e.payload["fold"] for e in reversed(s.events) if e.type == "fold")
    actual = before - truth(window_pieces(s))
    assert actual > 0 and abs(fold["freed_tokens"] - actual) <= actual / 10
    assert check(s.events, s.blobs) == []
    assert fold["freed_tokens"] < truth(victims) - OVERHEAD  # net of the summary, not the victims' gross


def test_a_large_replacement_predicts_small_net_savings_or_is_refused():
    s = session_with(window_estimator(fitted()))
    victims = sum(window_estimator(fitted()).count(t, c) for c, t in window_pieces(s)[1:9])
    reply = model_fold(s, "s" * 2050)
    fold = next((e.payload["fold"] for e in s.events if e.type == "fold"), None)
    if fold is not None:
        assert 0 < fold["freed_tokens"] < victims // 4  # net of the summary that replaces them, not gross
    else:
        assert reply.startswith("error: fold_would_grow_window")
    s2 = session_with(window_estimator(fitted()))
    assert model_fold(s2, "s" * 4000).startswith("error: fold_would_grow_window")


def test_backstop_folds_check_clean_under_calib_v2():
    s = session_with(window_estimator(fitted()), budget=900)
    for k in range(3):
        s.append_raw("user", f"follow-up {k} " * 30)
        s.call()
    folds = [e for e in s.events if e.type == "fold"]
    assert folds and all(e.payload["fold"]["policy"]["params"]["token_counter"] == "calib-v2" for e in folds)
    assert estimator_for(replay(s.events)).name == "calib-v2"
    assert check(s.events, s.blobs) == []


class LedgerMachineCalibV2(LedgerMachineCalib):
    """The random-session machine with calib-v2 in force: every invariant holds after every step."""

    def __init__(self) -> None:
        super().__init__()
        self.ledger = Ledger(":memory:")
        self.blobs = MemoryBlobStore()
        policy = BackstopPolicyV1(budget_tokens=BUDGET, cold_m=1, tail_keep=1, q0_bp=500, tau_tokens=20, token_counter="calib-v2")
        estimator = window_estimator(calib({c: Fraction(3, 10) for c in CLASSES}, Fraction(2)))
        self.session = ForgivingSession(self.ledger, self.blobs, BUDGET, policy, pricing=pricing(), tokenizer=estimator)
        self.session.append_raw("system", "Answer only from cited spans.", pinned=True)
        self.sources = []


LedgerMachineCalibV2.TestCase.settings = settings(max_examples=25, stateful_step_count=30, deadline=None)
TestLedgerInvariantsCalibV2 = LedgerMachineCalibV2.TestCase
