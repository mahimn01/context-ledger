"""M4 item 9: `eval analyze` implements PREREG sections 4-6 on synthetic results with known answers."""

from __future__ import annotations

import csv
import json
from fractions import Fraction

import pytest

from context_ledger.eval.analyze import (
    HAIKU,
    SONNET,
    Draws,
    analyze,
    build,
    holm,
    logistic_slope,
    median,
    ols_slope,
)
from context_ledger.eval.stats import mcnemar_exact
from context_ledger.pricing import PICO_PER_USD

RHOS = (10, 100, 1000)
R = 400  # resamples: enough for these tests, fast


def rows_for(model, arm, rho, episode, wrong: int, *, correct: int | None = None, unanswerable_false: int = 0, partial=False, error_q=None):
    """One episode of 8 scored rows: 6 answerable (`wrong` wrong, the rest correct unless
    `correct` says how many), 2 unanswerable (`unanswerable_false` false answers)."""
    correct = 6 - wrong if correct is None else correct
    out = []
    for q in range(8):
        answerable = q < 6
        if answerable:
            outcome = "wrong" if q < wrong else "correct" if q < wrong + correct else "abstain"
        else:
            outcome = "false_answer" if q - 6 < unanswerable_false else "correct_abstain"
        out.append({"model": model, "arm": arm, "rho": rho, "episode": episode, "qid": f"e{episode:02d}q{q}", "outcome": outcome,
                    "answerable": answerable, "fold_probe": 3 <= q < 6, "has_upstream": q < 4, "leaked": False, "partial": partial,
                    "error": q == error_q, "folds": 1 if arm == "L" else 0, "compactions": 1 if arm == "B4" else 0})  # fmt: skip
    return out


def calls_for(model, arm, rho, episode, cost_usd: Fraction):
    usage = {"input_tokens": 10, "cache_read_input_tokens": 70, "cache_creation_input_tokens": 20, "output_tokens": 5}
    return [{"model": model, "arm": arm, "rho": rho, "episode": episode, "cost_pusd": int(cost_usd * PICO_PER_USD), "usage": usage}]


def grid(wrong: dict, cost: dict | None = None, episodes=range(1, 13), model=HAIKU, rhos=RHOS):
    """wrong[arm] = f(rho, episode) -> wrong answers out of 6; cost[arm] = USD per episode."""
    scores, calls = [], []
    for arm, f in wrong.items():
        for rho in rhos:
            for e in episodes:
                scores += rows_for(model, arm, rho, e, f(rho, e))
                calls += calls_for(model, arm, rho, e, (cost or {}).get(arm, Fraction(1, 100)))
    return scores, calls


def by_test(result):
    return {(t["test"], t["model"]): t for t in result["tests"]}


PLANTED = {10: 1, 100: 3, 1000: 5}  # wrong answers out of 6: HR 1/6, 1/2, 5/6, slope 1/3 per decade


def test_planted_slope_zero_event_cells_and_h2(tmp_path):
    scores, calls = grid({
        "B3": lambda rho, e: 0,  # zero events in every cell
        "B4": lambda rho, e: PLANTED[rho] - (e % 2 if rho == 1000 else 0),  # rising, with some episode variation
        "L": lambda rho, e: 1,  # flat
    })  # fmt: skip
    result = analyze(scores, calls, [], tmp_path, resamples=R)
    t = by_test(result)
    b4 = t[("H1-B4", HAIKU)]
    assert b4["estimate"] == ols_slope([(1, Fraction(1, 6)), (2, Fraction(1, 2)), (3, Fraction(5 * 12 - 6, 72))])
    assert b4["p"] == 0 and b4["verdict"] == "supported" and b4["ci_low"] > 0
    b3 = t[("H1-B3", HAIKU)]
    assert (b3["estimate"], b3["ci_low"], b3["ci_high"], b3["p"]) == (0, 0, 0, 1) and b3["verdict"] == "not detected at this n"
    h2 = t[("H2", HAIKU)]
    assert h2["estimate"] == -b4["estimate"] and h2["p"] == 0 and h2["verdict"] == "supported" and h2["n_episodes"] == 12
    assert t[("logistic-slope-B3", HAIKU)]["verdict"] == "not_estimable" and t[("logistic-slope-B3", HAIKU)]["note"] == "zero events"
    logistic = t[("logistic-slope-B4", HAIKU)]
    assert logistic["verdict"] == "estimated" and logistic["estimate"] > 0 and logistic["p"] < 0.05
    assert t[("H2", SONNET)]["verdict"] == "not run"
    cells = list(csv.DictReader(open(tmp_path / "cells.csv")))
    b3_1000 = next(c for c in cells if c["arm"] == "B3" and c["rho"] == "1000")
    assert b3_1000["HR"] == "0" and b3_1000["HR_lo"] == "0" and b3_1000["HR_hi"] == "0" and b3_1000["n_questions"] == "96"
    figure = list(csv.DictReader(open(tmp_path / "figure_data.csv")))
    assert len(figure) == 9 and {f["arm"] for f in figure} == {"B3", "B4", "L"}
    assert next(f for f in figure if f["arm"] == "B4" and f["rho"] == "10")["HR"] == "0.166667"


def test_perfect_separation_keeps_the_ols_slope_and_marks_the_logistic_not_estimable(tmp_path):
    scores, calls = grid({"B3": lambda rho, e: 3 if rho == 1000 else 0, "B4": lambda rho, e: 0, "L": lambda rho, e: 0})
    t = by_test(analyze(scores, calls, [], tmp_path, resamples=R))
    assert t[("H1-B3", HAIKU)]["estimate"] == Fraction(1, 4)  # rates 0, 0, 1/2 on x = 1, 2, 3
    assert t[("logistic-slope-B3", HAIKU)]["verdict"] == "not_estimable" and t[("logistic-slope-B3", HAIKU)]["note"] == "separation"
    assert logistic_slope([(1, False, 1), (2, True, 1)])["reason"] == "separation"
    assert logistic_slope([(1, True, 1)] * 3)["reason"] == "only events"


def test_mcnemar_h3_non_inferiority_and_h4(tmp_path):
    scores, calls = grid(
        {"B4": lambda rho, e: 3, "L": lambda rho, e: 1 if e <= 6 else 2, "B3": lambda rho, e: 2},
        cost={"B4": Fraction(1, 100), "L": Fraction(12, 1000), "B3": Fraction(1, 200)},
    )
    t = by_test(analyze(scores, calls, [], tmp_path, resamples=R))
    m = t[("H2-McNemar", HAIKU)]
    b, c, n = 0, 12 * 3 - (6 * 1 + 6 * 2), 72  # L wrong on q0 (and q1 in episodes 7-12), B4 on q0..q2
    assert m["p"] == mcnemar_exact(b, c) and m["estimate"] == Fraction(b - c, n) and m["p_kind"] == "two-sided"
    h3 = t[("H3", HAIKU)]
    assert h3["estimate"] == Fraction(36, 144) - Fraction(1, 2) and h3["verdict"] == "supported"  # rho 100 and 1000 pooled
    ni = t[("H3-NI", HAIKU)]
    assert ni["estimate"] == Fraction(108, 144) - Fraction(1, 2) and ni["verdict"] == "met"
    h4 = t[("H4", HAIKU)]
    assert h4["estimate"] == Fraction(6, 5) and h4["ci_high"] == Fraction(6, 5) and h4["verdict"] == "supported"
    costly = by_test(analyze(*grid({"B4": lambda r, e: 3, "L": lambda r, e: 3}, cost={"B4": Fraction(1, 100), "L": Fraction(2, 100)}), [], tmp_path, resamples=R))
    assert costly[("H4", HAIKU)]["estimate"] == 2 and costly[("H4", HAIKU)]["verdict"] == "not supported" and costly[("H4", HAIKU)]["p"] == 1


def test_holm_is_step_down_and_capped():
    adjusted = holm({"a": Fraction(1, 100), "b": Fraction(4, 100), "c": Fraction(3, 100), "d": None})
    assert adjusted == {"a": Fraction(3, 100), "c": Fraction(6, 100), "b": Fraction(6, 100), "d": None}
    assert holm({"a": Fraction(9, 10), "b": Fraction(1, 2)}) == {"b": 1, "a": 1}


def test_exclusions_and_pairing(tmp_path):
    scores, calls = grid({"B4": lambda rho, e: PLANTED[rho], "L": lambda rho, e: 1, "B3": lambda rho, e: 1})
    scores = [s for s in scores if not (s["arm"] == "L" and s["rho"] == 100 and s["episode"] == 5)]
    scores += rows_for(HAIKU, "L", 100, 5, 1, partial=True)[:4]  # budget-stopped after 4 questions
    scores = [dict(s, error=True) if (s["arm"], s["rho"], s["episode"], s["qid"]) == ("B4", 10, 2, "e02q0") else s for s in scores]
    result = analyze(scores, calls, [], tmp_path, resamples=R)
    reasons = {(x["arm"], x["rho"], x["episode"], x["reason"]) for x in result["exclusions"]}
    assert ("B4", 10, 2, "infrastructure_error") in reasons and ("L", 100, 5, "incomplete_episode") in reasons
    t = by_test(result)
    assert t[("H2", HAIKU)]["n_episodes"] == 11  # episode 5 is incomplete in L: out of the paired tests
    cell = next(c for c in result["cells"] if (c["arm"], c["rho"]) == ("B4", 10))
    assert cell["n_questions"] == 95 and cell["HR"].estimate == Fraction(11, 71)  # the excluded question is in no metric
    l100 = next(c for c in result["cells"] if (c["arm"], c["rho"]) == ("L", 100))
    assert l100["n_episodes"] == 12 and l100["n_complete_episodes"] == 11 and l100["n_questions"] == 11 * 8 + 4  # as far as it went
    rows = list(csv.DictReader(open(tmp_path / "exclusions.csv")))
    assert {r["reason"] for r in rows} == {"infrastructure_error", "incomplete_episode"}
    doc = json.loads((tmp_path / "tests.json").read_text())
    assert doc["meta"]["resamples"] == R and all(isinstance(v, (str, int, type(None))) for t in doc["tests"] for v in t.values())


def test_bootstrap_draws_are_seeded_and_shared():
    a, b = Draws([1, 2, 3, 4], 50), Draws([4, 3, 2, 1], 50)
    assert a.weights == b.weights and all(sum(w.values()) == 4 for w in a.weights)
    assert Draws([1, 2, 3], 50, seed=1).weights != Draws([1, 2, 3], 50, seed=2).weights
    assert median([Fraction(3), Fraction(1), Fraction(2), Fraction(10)]) == Fraction(5, 2) and median([]) is None


def test_sonnet_rows_are_secondary_and_uncorrected(tmp_path):
    haiku = grid({"B4": lambda r, e: PLANTED[r], "L": lambda r, e: 1, "B3": lambda r, e: PLANTED[r]})
    sonnet = grid({"B4": lambda r, e: 2 if r == 10 else 4, "L": lambda r, e: 1}, model=SONNET, rhos=(10, 1000))
    t = by_test(analyze(haiku[0] + sonnet[0], haiku[1] + sonnet[1], [], tmp_path, resamples=R))
    for name in ("H2", "H2-McNemar", "H4"):
        row = t[(name, SONNET)]
        assert row["family"] == "secondary" and row["p_holm"] is None
    assert t[("H2", SONNET)]["estimate"] == ols_slope([(1, Fraction(1, 6)), (3, Fraction(1, 6))]) - ols_slope([(1, Fraction(1, 3)), (3, Fraction(2, 3))])
    assert all(t[(name, HAIKU)]["p_holm"] is not None for name in ("H1-B3", "H1-B4", "H2", "H3", "H4"))


def test_build_ignores_calls_of_units_without_results():
    scores = rows_for(HAIKU, "L", 10, 1, 1)
    data = build(scores, calls_for(HAIKU, "L", 10, 1, Fraction(1, 10)) + calls_for(HAIKU, "L", 10, 9, Fraction(5)))
    assert data.cells[(HAIKU, "L", 10)][1].cost_pusd == PICO_PER_USD // 10 and 9 not in data.cells[(HAIKU, "L", 10)]


def test_a_degenerate_sandwich_is_not_estimable_and_mcnemar_is_supporting(tmp_path):
    t = by_test(analyze(*grid({"B4": lambda r, e: PLANTED[r], "L": lambda r, e: 1 + e % 2}), [], tmp_path, resamples=R))
    assert t[("logistic-slope-L", HAIKU)]["verdict"] == "not_estimable" and "degenerate" in t[("logistic-slope-L", HAIKU)]["note"]
    assert t[("H2-McNemar", HAIKU)]["family"] == "supporting" and t[("H2-McNemar", HAIKU)]["p_holm"] is None
