"""BUILD-V3 item 4: `eval analyze-v3` on synthetic fixtures with known slopes, known McNemar counts and
known cost ratios, the forced L variants, the cells, and one run of the command on offline runs."""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import shutil
import sqlite3
import sys
import types
from fractions import Fraction
from pathlib import Path

import pytest

from context_ledger.eval import cli
from context_ledger.eval.analyze import HAIKU, holm
from context_ledger.eval.analyze_v3 import (
    CELL_COLUMNS,
    FIGURE_COLUMNS,
    TEST_COLUMNS,
    analyze_v3,
    cell_rows,
    hypothesis_tests,
    load,
    main_figure,
    mcnemar_one_sided,
    with_variants,
)
from context_ledger.eval.stats import nearest_rank, resample_indices
from context_ledger.pricing import PICO_PER_USD
from pilot_fixture import v2_fixture_files, write_fixture_tarball

SONNET = "claude-sonnet-5-5"


def row(arm, rho, episode, qid, outcome, *, condition="allowed", model=HAIKU, window=6000, answerable=None, lookalike=False, stale=False, cost=10**6,
        forced_violation=False, forced_fallback=False, gate_reason=None, gate_reasons=(), partial=False, error=False):  # fmt: skip
    answerable = outcome in ("correct", "wrong", "abstain") if answerable is None else answerable
    return {"model": model, "window": window, "condition": condition, "arm": arm, "rho": rho, "episode": episode, "qid": qid, "answerable": answerable,
            "outcome": outcome, "template": "constant", "lookalike": lookalike, "stale": stale, "cost_pusd": cost, "forced_violation": forced_violation,
            "forced_fallback": forced_fallback, "gate_reason": gate_reason, "gate_reasons": list(gate_reasons), "partial": partial, "error": error}  # fmt: skip


def by_name(tests, name, model=HAIKU):
    return next(t for t in tests if t["test"] == name and t["model"] == model)


def slope_rows(arm, wrong_by, condition="forced", outcome="wrong", n=4):
    """`n` answerable questions per episode and rho; wrong_by[episode][rho] of them get `outcome`."""
    return [row(arm, rho, e, f"e{e}q{k}", outcome if k < wrong else "correct", condition=condition)
            for e, per in wrong_by.items() for rho, wrong in per.items() for k in range(n)]  # fmt: skip


def test_known_slopes_and_their_bootstrap():
    same = slope_rows("B4", {e: {10: 0, 100: 2, 1000: 4} for e in (1, 2, 3)})  # HR 0, 1/2, 1 in every episode
    mixed = slope_rows("B3", {1: {10: 0, 100: 1, 1000: 2}, 2: {10: 0, 100: 3, 1000: 4}})  # mean HR 0, 1/2, 3/4
    tests = hypothesis_tests(same + mixed, resamples=500)
    h1b4, h1b3 = by_name(tests, "H1-B4"), by_name(tests, "H1-B3")
    assert (h1b4["estimate"], h1b4["ci_low"], h1b4["ci_high"], h1b4["p"]) == (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), 0)
    assert h1b3["estimate"] == Fraction(3, 8) and h1b3["p"] == 0 and h1b3["valid_resamples"] == 500  # OLS of (1, 0), (2, 1/2), (3, 3/4)
    flat = hypothesis_tests(slope_rows("B4", {e: {10: 2, 100: 2, 1000: 2} for e in (1, 2)}), resamples=50)
    assert by_name(flat, "H1-B4")["estimate"] == 0 and by_name(flat, "H1-B4")["p"] == 1  # ties count against the hypothesis


def test_fabrication_counts_unanswerable_and_abstention_slopes_use_the_allowed_condition():
    rows = []
    for e in (1, 2):
        for rho, false in ((10, 0), (100, 1), (1000, 2)):
            rows += [row("B4", rho, e, f"e{e}a{k}", "correct", condition="forced") for k in range(2)]
            rows += [row("B4", rho, e, f"e{e}u{k}", "false_answer" if k < false else "correct_abstain", condition="forced") for k in range(2)]
    rows += slope_rows("B3", {e: {10: 0, 100: 1, 1000: 2} for e in (1, 2)}, condition="allowed", outcome="abstain", n=2)
    tests = hypothesis_tests(rows, resamples=100)
    assert by_name(tests, "H1sec-B4")["estimate"] == Fraction(1, 4)  # fabrication 0, 1/4, 1/2 per episode
    assert by_name(tests, "H1-B4")["estimate"] == 0  # no wrong answer on answerable
    assert by_name(tests, "H1a-B3")["estimate"] == Fraction(1, 2) and by_name(tests, "H1a-B3")["condition"] == "allowed"
    assert by_name(tests, "H1a-B4")["verdict"] == "not run"  # B4 ran only forced


def mcnemar_rows(b, c, both=0, neither=0, *, other="B4", condition="allowed", rho=10, lookalike=False, start=0):
    """Items spread over episodes 1-3: b only L errs on, c only `other`, then both, then neither."""
    rows, k = [], start
    for l_err, o_err, count in ((True, False, b), (False, True, c), (True, True, both), (False, False, neither)):
        for _ in range(count):
            e = 1 + k % 3
            rows += [row("L", rho, e, f"q{k}", "wrong" if l_err else "correct", condition=condition, lookalike=lookalike),
                     row(other, rho, e, f"q{k}", "wrong" if o_err else "correct", condition=condition, lookalike=lookalike)]  # fmt: skip
            k += 1
    return rows


def test_known_mcnemar_counts_and_subgroups():
    assert mcnemar_one_sided(1, 5) == Fraction(7, 64) and mcnemar_one_sided(0, 3) == Fraction(1, 8) and mcnemar_one_sided(0, 0) == 1
    rows = mcnemar_rows(1, 5, both=2, neither=4) + mcnemar_rows(0, 3, neither=1, lookalike=True, start=100)
    rows += mcnemar_rows(2, 2, other="B3", start=200)
    tests = hypothesis_tests(rows, resamples=300)
    h2 = by_name(tests, "H2-vsB4")
    assert (h2["b"], h2["c"], h2["n_items"], h2["p"]) == (1, 8, 16, mcnemar_one_sided(1, 8))
    assert h2["estimate"] == Fraction(1 - 8, 16) and h2["ci_low"] <= h2["estimate"] <= h2["ci_high"]
    look = by_name(tests, "H2-vsB4-lookalike")
    assert (look["b"], look["c"], look["n_items"], look["p"]) == (0, 3, 4, Fraction(1, 8)) and look["p_holm"] is None
    assert by_name(tests, "H2-vsB4-stale")["verdict"] == "not run"
    vs_b3 = by_name(tests, "H2-vsB3")
    assert (vs_b3["b"], vs_b3["c"], vs_b3["p"]) == (2, 2, Fraction(11, 16))


def test_h3_counts_fallback_answers_as_answers_and_the_gated_variant_as_abstentions():
    rows = []
    for k in range(6):
        e = 1 + k % 3
        fallback = k < 2  # L's wrong answers on q0, q1 are fallbacks; on q2 a gated answer
        rows.append(row("L", 100, e, f"q{k}", "wrong" if k < 3 else "correct", condition="forced", forced_fallback=fallback))
        rows.append(row("B4", 100, e, f"q{k}", "wrong" if k in (2, 3, 4) else "correct", condition="forced"))
    rows += [row("L", 10, 1, "low", "wrong", condition="forced"), row("B4", 10, 1, "low", "correct", condition="forced")]  # rho < 100: out
    tests = hypothesis_tests(rows, resamples=200)
    primary, gated = by_name(tests, "H3"), by_name(tests, "H3-gated")
    assert (primary["b"], primary["c"], primary["n_items"]) == (2, 2, 6) and primary["p"] == mcnemar_one_sided(2, 2)
    assert (gated["b"], gated["c"], gated["n_items"]) == (0, 2, 6) and gated["p"] == Fraction(1, 4)
    assert primary["p_holm"] is not None and gated["p_holm"] is None


def test_h4_is_judged_by_the_upper_bound_of_its_interval():
    cheap = [row(arm, 10, e, f"e{e}q{k}", "correct", cost=10**6) for arm in ("L", "B4") for e in (1, 2, 3) for k in range(3)]
    supported = by_name(hypothesis_tests(cheap, resamples=200), "H4")
    assert (supported["estimate"], supported["ci_high"], supported["verdict"]) == (1, 1, "supported")
    dear = [{**r, "cost_pusd": 2 * 10**6} if r["arm"] == "L" else r for r in cheap]
    refused = by_name(hypothesis_tests(dear, resamples=200), "H4")
    assert (refused["estimate"], refused["verdict"], refused["p"]) == (2, "not supported", 1)


def test_holm_runs_over_the_five_per_model_family():
    rows = mcnemar_rows(0, 6, neither=2) + mcnemar_rows(0, 6, neither=2, other="B3", start=200)
    rows += [{**r, "model": SONNET} for r in mcnemar_rows(0, 6, neither=2)]
    tests = hypothesis_tests(rows, resamples=100)
    haiku = {t["test"]: t for t in tests if t["model"] == HAIKU}
    expected = holm({k: haiku[k]["p"] for k in ("H1-B3", "H1-B4", "H2-vsB4", "H2-vsB3", "H3")})
    assert all(haiku[k]["p_holm"] == v for k, v in expected.items()) and haiku["H2-vsB4"]["family"] == "primary"
    assert haiku["H2-vsB4"]["p_holm"] == haiku["H2-vsB3"]["p_holm"] == 2 * Fraction(1, 64)  # two of the five have a p of 1/64
    assert haiku["H2-vsB4"]["verdict"] == "supported" and haiku["H1-B4"]["verdict"] == "not run"
    sonnet = by_name(tests, "H2-vsB4", SONNET)
    assert sonnet["family"] == "secondary" and sonnet["p_holm"] == Fraction(1, 64)  # its own family: one test with a p


def test_cells_variants_rates_and_exclusions(tmp_path):
    rows = [
        row("L", 10, 1, "a", "correct", condition="forced", gate_reason="anchored", gate_reasons=["anchored"], cost=2 * 10**6),
        row("L", 10, 1, "b", "wrong", condition="forced", forced_fallback=True, gate_reasons=["target_not_witnessed"], lookalike=True, cost=4 * 10**6),
        row("L", 10, 1, "c", "false_answer", condition="forced", forced_fallback=True, stale=False, cost=6 * 10**6),
        row("L", 10, 1, "d", "abstain", condition="forced", forced_violation=True, stale=True, cost=8 * 10**6),
        row("L", 10, 2, "e", "wrong", condition="forced", error=True),
        row("L", 10, 3, "f", "correct", condition="forced", partial=True, cost=10 * 10**6),
    ]
    result = analyze_v3(rows, tmp_path, resamples=20)
    cells = {(c["variant"]): c for c in result["cells"]}
    p, g = cells["primary"], cells["gated"]
    assert (p["n_answerable"], p["correct"], p["wrong"], p["abstain"], p["n_unanswerable"], p["false_answer"]) == (4, 2, 1, 1, 1, 1)
    assert (p["forced_fallback"], p["forced_fallback_wrong"], p["forced_fallback_false_answer"], p["forced_violation"]) == (2, 1, 1, 1)
    assert (p["hr"], p["fabrication"], p["abstain_rate"]) == (Fraction(1, 4), Fraction(2, 5), Fraction(1, 4))
    assert (p["n_lookalike"], p["lookalike_confident_error_rate"], p["n_stale"], p["stale_confident_error_rate"]) == (1, 1, 1, 0)
    six = Fraction(6 * 10**6, PICO_PER_USD)
    assert (p["cost_per_question_median_usd"], p["cost_per_question_mean_usd"]) == (six, six)  # the error row is out, the partial row in
    assert p["gate_reasons"] == "anchored:1;not_run:4" and p["gate_refusals"] == "target_not_witnessed:1"
    assert (g["wrong"], g["abstain"], g["false_answer"], g["correct_abstain"], g["fabrication"]) == (0, 2, 0, 1, 0)
    # the gated cell still reports the grader's grades of the fallback answers
    assert (g["forced_fallback"], g["forced_fallback_wrong"], g["forced_fallback_false_answer"], g["forced_violation_correct_abstain"]) == (2, 1, 1, 0)
    assert len(with_variants([row("B4", 10, 1, "a", "correct", condition="forced")])) == 1  # only L has a gated variant
    with open(tmp_path / "figure_data.csv") as f:
        figure = {r["variant"]: r for r in csv.DictReader(f)}
    assert (figure["primary"]["correct_share"], figure["primary"]["false_answer_share"], figure["gated"]["abstain_share"]) == ("0.5", "1", "0.5")
    tests = json.loads((tmp_path / "tests.json").read_text())
    assert tests["meta"]["seed"] == 0 and {t["test"] for t in tests["tests"]} >= {"H1-B3", "H2-vsB4", "H3", "H3-gated", "H4"}
    assert cell_rows(with_variants([r for r in rows if not r["error"]])) == result["cells"]


def test_holm_can_leave_a_raw_p_below_alpha_not_detected():
    # every item look-alike: H2-vsB4 and its look-alike subgroup see the same five discordant items (raw p 1/32)
    rows = mcnemar_rows(0, 5, neither=1, lookalike=True) + mcnemar_rows(0, 5, neither=1, other="B3", start=200)
    tests = {t["test"]: t for t in hypothesis_tests(rows, resamples=100)}
    for name in ("H2-vsB4", "H2-vsB3"):
        assert (tests[name]["p"], tests[name]["p_holm"], tests[name]["verdict"]) == (Fraction(1, 32), Fraction(1, 16), "not detected at this n")
    look = tests["H2-vsB4-lookalike"]
    assert (look["p"], look["p_holm"], look["verdict"]) == (Fraction(1, 32), None, "significant at 0.05 (uncorrected)")
    assert {k for k, t in tests.items() if t["p_holm"] is not None} == {"H2-vsB4", "H2-vsB3"}  # the other three of the five did not run


def test_bootstrap_is_seed_0_by_default_and_deterministic(tmp_path):
    for fn in (hypothesis_tests, analyze_v3):
        params = inspect.signature(fn).parameters
        assert (params["seed"].default, params["resamples"].default) == (0, 10_000)
    rows = mcnemar_rows(3, 7, both=2, neither=6) + slope_rows("B4", {1: {10: 0, 100: 1, 1000: 3}, 2: {10: 1, 100: 1, 1000: 2}, 3: {10: 0, 100: 2, 1000: 2}})
    first, again = hypothesis_tests(rows, resamples=400), hypothesis_tests(rows, resamples=400)
    assert first == again
    # H2-vsB4 by hand: the draws are stats.resample_indices(3 episodes, 400, seed 0), the statistic (b - c) / items
    h2 = by_name(first, "H2-vsB4")
    per = {e: [0, 0, 0] for e in (1, 2, 3)}  # b, c, items per episode
    for k, (l_err, o_err) in enumerate([(1, 0)] * 3 + [(0, 1)] * 7 + [(1, 1)] * 2 + [(0, 0)] * 6):
        per[1 + k % 3][0] += l_err and not o_err
        per[1 + k % 3][1] += o_err and not l_err
        per[1 + k % 3][2] += 1

    def by_hand(seed):
        values = []
        for draw in resample_indices(3, 400, seed):
            b, c, n = (sum(per[1 + i][j] for i in draw) for j in range(3))
            values.append(Fraction(b - c, n))
        return nearest_rank(values, Fraction(5, 2)), nearest_rank(values, Fraction(195, 2)), 400

    assert (h2["ci_low"], h2["ci_high"], h2["valid_resamples"]) == by_hand(0)
    reseeded = by_name(hypothesis_tests(rows, resamples=400, seed=1), "H2-vsB4")
    assert resample_indices(3, 400, 1) != resample_indices(3, 400, 0) and (reseeded["ci_low"], reseeded["ci_high"], reseeded["valid_resamples"]) == by_hand(1)
    assert reseeded["p"] == h2["p"]  # the exact McNemar p does not resample
    a, b = analyze_v3(rows, tmp_path / "a", resamples=200), analyze_v3(rows, tmp_path / "b", resamples=200)
    assert a["tests"] == b["tests"]
    for name in ("cells.csv", "tests.csv", "tests.json", "figure_data.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_forced_l_primary_counts_fallbacks_as_answers_and_gated_as_abstentions():
    rows = [
        row("L", 100, 1, "a", "correct", condition="forced", forced_fallback=True),
        row("L", 100, 1, "b", "wrong", condition="forced", forced_fallback=True),
        row("L", 100, 1, "c", "wrong", condition="forced"),  # an answer the gate passed: wrong in both variants
        row("L", 100, 1, "u1", "false_answer", condition="forced", forced_fallback=True),
        row("L", 100, 1, "u2", "correct_abstain", condition="forced", forced_violation=True),
        row("L", 100, 1, "a", "correct", condition="allowed"),
        row("B4", 100, 1, "a", "wrong", condition="forced", forced_violation=False),
    ]
    cells = {(c["condition"], c["arm"], c["variant"]): c for c in cell_rows(with_variants(rows))}
    assert set(cells) == {("forced", "L", "primary"), ("forced", "L", "gated"), ("allowed", "L", "primary"), ("forced", "B4", "primary")}
    p, g = cells[("forced", "L", "primary")], cells[("forced", "L", "gated")]
    assert (p["correct"], p["wrong"], p["abstain"], p["false_answer"], p["correct_abstain"]) == (1, 2, 0, 1, 1)
    assert (g["correct"], g["wrong"], g["abstain"], g["false_answer"], g["correct_abstain"]) == (0, 1, 2, 0, 2)
    assert (p["hr"], p["fabrication"], g["hr"], g["fabrication"]) == (Fraction(2, 3), Fraction(3, 5), Fraction(1, 3), Fraction(1, 5))
    for c in (p, g):  # the same flags and the grader's grades of the fallback answers in both variants
        assert (c["forced_fallback"], c["forced_fallback_correct"], c["forced_fallback_wrong"], c["forced_fallback_false_answer"]) == (3, 1, 1, 1)
        assert (c["forced_violation"], c["forced_violation_correct_abstain"]) == (1, 1)
    assert cells[("allowed", "L", "primary")]["forced_fallback"] == 0
    # the grader's outcome is untouched on the primary rows
    assert [r["outcome"] for r in with_variants(rows) if r["variant"] == "primary"] == [r["outcome"] for r in rows]


class _Axes:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a, k))


class _Figure:
    def tight_layout(self):
        pass

    def savefig(self, path, **kw):
        path.write_bytes(b"stub png")


def _fake_matplotlib(monkeypatch) -> dict:
    """A stand-in for matplotlib recording what main_figure draws (the real one is no dependency)."""
    seen: dict = {}
    mpl, plt = types.ModuleType("matplotlib"), types.ModuleType("matplotlib.pyplot")
    mpl.use = lambda backend: seen.setdefault("backend", backend)

    def subplots(nrows, ncols, **kw):
        seen["grid"] = [[_Axes() for _ in range(ncols)] for _ in range(nrows)]
        return _Figure(), seen["grid"]

    plt.subplots, plt.close = subplots, lambda fig: seen.setdefault("closed", True)
    mpl.pyplot = plt
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)
    return seen


def test_main_figure_is_drawn_from_figure_data_csv_alone(tmp_path, monkeypatch):
    lines = [",".join(FIGURE_COLUMNS),
             f"{HAIKU},6000,forced,L,primary,10,4,0.5,0.25,0.25,2,0.5", f"{HAIKU},6000,forced,L,primary,100,4,0.25,0.5,0.25,2,1",
             f"{HAIKU},6000,forced,L,gated,10,4,0.5,0,0.5,2,0", f"{HAIKU},6000,allowed,B4,primary,10,4,0.75,0,0.25,0,"]  # fmt: skip
    (tmp_path / "figure_data.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    seen = _fake_matplotlib(monkeypatch)
    assert main_figure(tmp_path) == tmp_path / "main_figure.png" and (tmp_path / "main_figure.png").exists()
    assert seen["backend"] == "Agg" and seen["closed"] and len(seen["grid"]) == 2 and len(seen["grid"][0]) == 4  # allowed, forced x four shares
    plots = lambda ax: [(c[1], c[2]["label"]) for c in ax.calls if c[0] == "plot"]  # noqa: E731
    allowed, forced = seen["grid"]
    assert plots(forced[1]) == [(((10, 100), (0.25, 0.5)), "L"), (((10,), (0.0,)), "L (gated)")]  # wrong share, from the CSV
    assert plots(forced[3]) == [(((10, 100), (0.5, 1.0)), "L"), (((10,), (0.0,)), "L (gated)")]
    assert plots(allowed[3]) == [] and plots(allowed[2]) == [(((10,), (0.25,)), "B4")]  # no unanswerable question: no false-answer point


def test_main_figure_is_skipped_without_matplotlib(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # import matplotlib now raises ImportError
    result = analyze_v3([row("B4", 10, 1, "a", "correct", cost=1_234_567)], tmp_path, resamples=10)
    assert result["figure"] is None and (tmp_path / "figure_data.csv").exists() and not (tmp_path / "main_figure.png").exists()
    with open(tmp_path / "cells.csv") as f:
        (cell,) = csv.DictReader(f)
    assert (cell["cost_per_question_median_usd"], cell["cost_per_question_mean_usd"], cell["abstain_rate"]) == ("0.000001235", "0.000001235", "0")


def test_output_columns():
    assert CELL_COLUMNS == ["model", "window", "condition", "arm", "variant", "rho", "n_episodes", "n_answerable", "correct", "wrong", "abstain", "n_unanswerable",
                            "false_answer", "correct_abstain", "forced_violation", "forced_violation_correct_abstain", "forced_fallback",
                            "forced_fallback_correct", "forced_fallback_wrong", "forced_fallback_false_answer", "hr", "fabrication", "abstain_rate",
                            "n_lookalike", "lookalike_confident_error_rate", "n_stale", "stale_confident_error_rate", "cost_per_question_median_usd",
                            "cost_per_question_mean_usd", "gate_reasons", "gate_refusals"]  # fmt: skip
    assert TEST_COLUMNS == ["test", "model", "window", "family", "condition", "estimate", "ci_low", "ci_high", "p", "p_kind", "p_holm", "verdict", "n_episodes",
                            "n_items", "b", "c", "valid_resamples", "note"]  # fmt: skip
    assert FIGURE_COLUMNS == ["model", "window", "condition", "arm", "variant", "rho", "n_answerable", "correct_share", "wrong_share", "abstain_share",
                              "n_unanswerable", "false_answer_share"]  # fmt: skip


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    work = tmp_path_factory.mktemp("analyze-v3")
    tar, sha = write_fixture_tarball(work, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    mix = "anchors=5,probes=3,multihop=2,unanswerable=2,notes=4,supersessions=1,session=3,session_unanswerable=1"
    assert e("questions", "build", "--version", "v3", "--seed", "2", "--episodes", "2", "--questions-per-episode", "12", "--mix", mix, "--probe-gap", "4") == 0
    for condition in ("allowed", "forced"):
        assert e("run", "--questions", "v3", "--rho", "10,20", "--episodes", "1-2", "--window", "3000", "--budget", "2", "--run-id", condition,
                 "--condition", condition) == 0  # fmt: skip
    return work, e


def test_the_command_analyzes_both_conditions_read_only(runs, capsys, monkeypatch):
    work, e = runs
    before = {p: p.read_bytes() for p in (work / "runs").rglob("*") if p.is_file()}
    opened, connect = [], sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda database, *a, **k: opened.append(str(database)) or connect(database, *a, **k))
    capsys.readouterr()
    assert e("analyze-v3", str(work / "runs" / "allowed"), str(work / "runs" / "forced"), "--out", str(work / "out"), "--resamples", "50") == 0
    shown = capsys.readouterr().out
    assert "wrote cells.csv" in shown
    assert {p: p.read_bytes() for p in (work / "runs").rglob("*") if p.is_file()} == before  # read-only on runs/
    assert all(d.startswith("file:") and "mode=ro&immutable=1" in d for d in opened)  # any ledger read-only (it reads none: results.jsonl has all)
    # the png only where matplotlib is importable (it is not a dependency; this .venv has none)
    plotted = importlib.util.find_spec("matplotlib") is not None
    assert (work / "out" / "main_figure.png").exists() == plotted and ("wrote main_figure.png" in shown) == plotted
    with open(work / "out" / "cells.csv") as f:
        cells = list(csv.DictReader(f))
    assert list(cells[0]) == CELL_COLUMNS
    keys = {(c["condition"], c["arm"], c["variant"], c["rho"]) for c in cells}
    assert keys == {(cond, arm, "primary", rho) for cond in ("allowed", "forced") for arm in ("B3", "B4", "L") for rho in ("10", "20")} | {
        ("forced", "L", "gated", rho) for rho in ("10", "20")}
    assert all(c["window"] == "3000" and c["gate_reasons"] for c in cells if c["arm"] == "L")
    tests = {t["test"]: t for t in json.loads((work / "out" / "tests.json").read_text())["tests"]}
    # the fixture corpus holds no two powers of ten of rho, nor rho >= 100 (the synthetic tests above cover those)
    assert tests["H1-B4"]["verdict"] == "not run" and "powers of ten" in tests["H1-B4"]["note"] and tests["H3"]["verdict"] == "not run"
    assert tests["H2-vsB4"]["n_items"] == 64 and tests["H2-vsB4"]["verdict"] != "not run" and tests["H4"]["n_episodes"] == 2


# BUILD-V3FIX: reruns. A unit incomplete in a run is dropped; a unit complete in several runs is taken from the later.

UNIT_QS = [{"qid": f"e{e}q{k}", "episode": e, "pos": k, "template": "constant", "answerable": True, "fold_probe": False, "answer": "1"}
           for e in (6, 7) for k in range(3)]  # fmt: skip


def write_run(root: Path, run_id: str, units: dict[tuple[str, int], int], *, stopped=None, partial=(), cost=1) -> Path:
    """runs/<run_id>: Haiku, W 3000, rho 10, allowed, asked every UNIT_QS question in each arm of
    `units`; results: the first n questions of each (arm, episode) unit, abstaining, those units
    in `partial` flagged partial."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    arms = sorted({arm for arm, _ in units})
    run = {"run_id": run_id, "model": HAIKU, "window": 3000, "rho": 10, "condition": "allowed", "questions": {a: [q["qid"] for q in UNIT_QS] for a in arms},
           "incomplete": stopped is not None, "stopped": stopped and {"arm": stopped[0], "episode": stopped[1], "reason": "api_errors"}}  # fmt: skip
    (run_dir / "run.json").write_text(json.dumps(run))
    rows = [{"arm": arm, "qid": q["qid"], "episode": e, "rho": 10, "model": HAIKU, "abstained": True, "answer": None, "citations": [], "cost_pusd": cost,
             "bound_hit": False, "calls": 1, "condition": "allowed", **({"partial": True} if (arm, e) in partial else {})}
            for (arm, e), n in units.items() for q in [q for q in UNIT_QS if q["episode"] == e][:n]]  # fmt: skip
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run_dir


def loaded_units(dirs):
    rows, superseded, dropped = load(dirs, lambda run: (UNIT_QS, {}))
    taken = sorted({(r["arm"], r["episode"], r["cost_pusd"]) for r in rows})
    return taken, len(rows), superseded, dropped


def unit(arm, episode):
    return [HAIKU, 3000, "allowed", arm, 10, episode]


def test_load_drops_the_unit_a_run_stopped_in_and_takes_its_rerun(tmp_path):
    stopped = write_run(tmp_path, "v3-grid-x-e6-7", {("B4", 6): 3, ("B4", 7): 3, ("L", 6): 2}, stopped=("L", 6), partial={("L", 6)})
    rerun = write_run(tmp_path, "v3-grid-x-e6-7-Lrerun", {("L", 6): 3, ("L", 7): 3}, cost=2)
    assert sorted([rerun, stopped]) == [stopped, rerun]  # the shell glob gives the rerun after the run it reruns
    for order in ([stopped, rerun], [rerun, stopped]):  # a dropped unit never competes, so the order does not matter here
        taken, n, superseded, dropped = loaded_units(order)
        assert taken == [("B4", 6, 1), ("B4", 7, 1), ("L", 6, 2), ("L", 7, 2)] and n == 12 and superseded == []
        assert dropped == [{"unit": unit("L", 6), "run": "v3-grid-x-e6-7", "rows": 2, "questions": 3, "reason": "run_stopped"}]


def test_load_takes_a_unit_complete_in_two_runs_from_the_later(tmp_path):
    first = write_run(tmp_path, "first", {("L", 6): 3, ("L", 7): 3})
    later = write_run(tmp_path, "later", {("L", 6): 3}, cost=2)
    taken, n, superseded, dropped = loaded_units([first, later])
    assert taken == [("L", 6, 2), ("L", 7, 1)] and n == 6 and dropped == []
    assert superseded == [{"unit": unit("L", 6), "run": "first", "replaced_by": "later"}]
    taken, _, superseded, _ = loaded_units([later, first])
    assert taken == [("L", 6, 1), ("L", 7, 1)] and superseded == [{"unit": unit("L", 6), "run": "later", "replaced_by": "first"}]


def test_load_drops_a_unit_short_of_rows_even_unflagged(tmp_path):
    short = write_run(tmp_path, "short", {("L", 6): 2, ("L", 7): 3})
    taken, n, superseded, dropped = loaded_units([short])
    assert taken == [("L", 7, 1)] and n == 3 and superseded == []
    assert dropped == [{"unit": unit("L", 6), "run": "short", "rows": 2, "questions": 3, "reason": "missing_rows"}]


def test_a_stopped_later_run_never_replaces_a_complete_unit(tmp_path):
    complete = write_run(tmp_path, "complete", {("L", 6): 3})
    stopped = write_run(tmp_path, "stopped", {("L", 6): 3}, stopped=("L", 6), cost=2)  # every row there, but the run stopped in it
    taken, _, superseded, dropped = loaded_units([complete, stopped])
    assert taken == [("L", 6, 1)] and superseded == [] and [d["reason"] for d in dropped] == ["run_stopped"]


def test_the_command_on_a_stopped_run_and_its_rerun_matches_the_run_that_never_stopped(runs, tmp_path, capsys):
    """The allowed fixture run copied twice: once stopped in L episode 2 of its rho-10 part (rows
    cut short and flagged partial), once as a rerun of that unit alone. analyze-v3 over the forced
    run, the stopped run and the rerun writes exactly what it writes over the complete runs."""
    work, e = runs
    source = work / "runs" / "allowed"
    stopped, rerun = tmp_path / "v3-grid-allowed", tmp_path / "v3-grid-allowed-Lrerun"
    shutil.copytree(source, stopped, ignore=shutil.ignore_patterns("*.sqlite", "B4"))
    part = stopped / "rho-10"
    run = json.loads((part / "run.json").read_text())
    results = [json.loads(x) for x in (part / "results.jsonl").read_text().splitlines()]
    redo = [r for r in results if (r["arm"], r["episode"]) == ("L", 2)]
    assert len(redo) > 3
    kept = [r for r in results if (r["arm"], r["episode"]) != ("L", 2)] + [{**r, "partial": True} for r in redo[:3]]
    (part / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    (part / "run.json").write_text(json.dumps({**run, "run_id": "v3-grid-allowed", "incomplete": True, "stopped": {"arm": "L", "episode": 2, "reason": "api_errors"}}))
    (rerun / "rho-10").mkdir(parents=True)
    (rerun / "rho-10" / "run.json").write_text(json.dumps({**run, "run_id": "v3-grid-allowed-Lrerun", "arms": ["L"], "questions": {"L": run["questions"]["L"]}}))
    (rerun / "rho-10" / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in redo))
    forced = work / "runs" / "forced"
    assert e("analyze-v3", str(source), str(forced), "--out", str(tmp_path / "complete"), "--resamples", "50") == 0
    assert e("analyze-v3", str(forced), *map(str, sorted([rerun, stopped])), "--out", str(tmp_path / "merged"), "--resamples", "50") == 0
    capsys.readouterr()
    for name in ("cells.csv", "tests.csv", "figure_data.csv"):
        assert (tmp_path / "merged" / name).read_bytes() == (tmp_path / "complete" / name).read_bytes(), name
    complete, merged = (json.loads((tmp_path / d / "tests.json").read_text()) for d in ("complete", "merged"))
    assert merged["tests"] == complete["tests"] and complete["meta"]["dropped_units"] == complete["meta"]["superseded_units"] == []
    assert merged["meta"]["superseded_units"] == []
    assert merged["meta"]["dropped_units"] == [{"unit": [run["model"], run["window"], "allowed", "L", 10, 2], "run": "v3-grid-allowed", "rows": 3,
                                                "questions": len(redo), "reason": "run_stopped"}]  # fmt: skip
