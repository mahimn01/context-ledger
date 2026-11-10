"""BUILD-V4 item 9 and BUILD-V4B decisions 4-5: analyze-v4 on synthetic rows with known answers. The
six confirmatory tests per model (L4 non-inferior to B3 within 3 pp; L4 fewer confident errors
than B3 and than B4; L4's accuracy flat over rho by TOST in both conditions; forced, rho >= 100,
L4 fewer unlabelled fabrications than B4), Holm within each model's family, the exploratory
rows (L vs L4 on corpus questions and on every turn, the gate and index effects, the cost
ratio), the gated variant of L4's unverified guesses, subgroup cells and the files."""

from __future__ import annotations

import csv
import json
from fractions import Fraction

from context_ledger.eval.analyze import holm
from context_ledger.eval.analyze_v3 import mcnemar_one_sided
from context_ledger.eval.analyze_v4 import CONFIRMATORY, HOLM_V4, analyze_v4, cell_rows, hypothesis_tests, with_variants

RHOS = (10, 100, 1000, 10000)
EPISODES = range(1, 9)
N = len(EPISODES) * len(RHOS)


def row(arm: str, rho: int, episode: int, k: int, outcome: str, condition: str = "allowed", model: str = "claude-haiku-5-5", **extra) -> dict:
    answerable = outcome in ("correct", "wrong", "abstain")
    return {"model": model, "window": 3000, "condition": condition, "arm": arm, "rho": rho, "episode": episode, "qid": f"q{episode}-{k}",
            "answerable": answerable, "outcome": outcome, "template": "constant", "lookalike": k % 2 == 0, "stale": False, "cost_pusd": 1000 * (2 if arm == "L4" else 3),
            "forced_violation": False, "forced_fallback": False, "gate_reason": None, "gate_reasons": [], "partial": False, "error": False,
            "unverified_guess": False, "repairs": 0, "repair_reasons": [], "preresolved": "exact", "reason": "answered", "kind": "corpus", "origin": "generator",
            **extra}  # fmt: skip


def synthetic(model: str = "claude-haiku-5-5") -> list[dict]:
    """10 answerable items per episode and rho. L4: 8 correct at every rho (flat), 2 abstentions.
    B3: 9, 7, 5, 3 correct at rho 10..10000 (-20 pp per decade), its misses wrong answers. B4:
    L4's outcomes but item 8 wrong (L4 has fewer confident errors). B5: B3's outcomes."""
    rows = []
    for e in EPISODES:
        for n, rho in enumerate(RHOS):
            for k in range(10):
                rows.append(row("L4", rho, e, k, "correct" if k < 8 else "abstain", model=model))
                rows.append(row("B4", rho, e, k, "correct" if k < 8 else "wrong" if k == 8 else "abstain", model=model))
                rows.append(row("B3", rho, e, k, "correct" if k < 9 - 2 * n else "wrong", model=model))
                rows.append(row("B5", rho, e, k, "correct" if k < 9 - 2 * n else "wrong", model=model))
    return rows


def by_name(rows, **kw):
    return {t["test"]: t for t in hypothesis_tests(rows, resamples=kw.get("resamples", 300), seed=0)}


def test_the_confirmatory_family_is_six_tests_with_holm_over_them_only():
    assert HOLM_V4 == CONFIRMATORY and len(CONFIRMATORY) == 6
    tests = by_name(synthetic())
    confirmatory = [t for t in tests.values() if t["role"] == "confirmatory"]
    assert [t["test"] for t in confirmatory] == list(CONFIRMATORY)  # all six, in order, even those the rows cannot run
    assert [t["test"] for t in confirmatory if t["verdict"] == "not run"] == ["TOST-ACC-L4-forced", "UFAB-L4vsB4-forced"]  # no forced rows here
    adjusted = holm({t["test"]: t["p"] for t in confirmatory})
    assert all(t["p_holm"] == adjusted[t["test"]] for t in confirmatory)
    assert all(t["p_holm"] is None and t["role"] == "exploratory" for t in tests.values() if t["test"] not in CONFIRMATORY)
    assert all(t["family"] == "primary" for t in tests.values())


def test_non_inferiority_to_b3_within_three_points():
    tests = by_name(synthetic())
    ni = tests["NI-ACC-L4vsB3"]
    assert ni["estimate"] == Fraction(8, 10) - Fraction(9 + 7 + 5 + 3, 40) and ni["p"] == 0 and ni["verdict"] == "non-inferior within 3 pp"
    assert ni["n_items"] == 10 * N and ni["ci90_low"] > Fraction(-3, 100)
    worse = [row("L4", rho, e, k, "correct" if k < 7 else "abstain") for e in EPISODES for rho in RHOS for k in range(10)]
    worse += [row("B3", rho, e, k, "correct" if k < 8 else "wrong") for e in EPISODES for rho in RHOS for k in range(10)]
    ni = by_name(worse)["NI-ACC-L4vsB3"]
    assert ni["estimate"] == Fraction(-1, 10) and ni["p"] == 1 and ni["verdict"] == "non-inferiority not shown at this n"
    tie = [r for r in worse if r["arm"] == "B3"] + [{**r, "arm": "L4"} for r in worse if r["arm"] == "B3"]
    ni = by_name(tie)["NI-ACC-L4vsB3"]
    assert ni["estimate"] == 0 and ni["p"] == 0  # a 0 difference is inside the margin


def test_confident_errors_against_b3_and_b4_are_item_paired_mcnemar():
    tests = by_name(synthetic())
    ce4 = tests["CE-L4vsB4"]
    assert (ce4["b"], ce4["c"], ce4["n_items"]) == (0, N, 10 * N) and ce4["p"] == mcnemar_one_sided(0, N) and ce4["verdict"] == "supported"
    ce3 = tests["CE-L4vsB3"]
    assert (ce3["b"], ce3["c"]) == (0, len(EPISODES) * (1 + 3 + 5 + 7)) and "only L4 errs on" in ce3["note"] and "only B3 errs on" in ce3["note"]


def test_accuracy_slopes_and_tost():
    tests = by_name(synthetic(), resamples=400)
    flat, falling = tests["TOST-ACC-L4-allowed"], tests["TOST-ACC-B3-allowed"]
    assert flat["estimate"] == 0 and flat["p"] == 0 and flat["verdict"] == "equivalent within +/-2 pp per decade"
    assert falling["estimate"] == Fraction(-1, 5) and falling["p"] == 1 and falling["verdict"] == "equivalence not shown at this n (exploratory, uncorrected)"
    assert tests["SLOPE-FAB-B3-allowed"]["estimate"] == Fraction(1, 5) and tests["SLOPE-FAB-B3-allowed"]["p"] == 0  # fabrication rises 20 pp per decade


def forced_rows() -> list[dict]:
    """Forced, 10 items per unit: item 0 answered wrong by both; L4's is flagged unverified_guess.
    Item 1 wrong for B4 only at rho 10 (outside the rho >= 100 test)."""
    rows = []
    for e in EPISODES:
        for rho in RHOS:
            for k in range(10):
                l4 = "wrong" if k == 0 else "correct"
                b4 = "wrong" if k == 0 or (k == 1 and rho == 10) else "correct"
                rows.append(row("L4", rho, e, k, l4, "forced", unverified_guess=k == 0, forced_fallback=k == 0, reason="unverified_guess" if k == 0 else "answered"))
                rows.append(row("B4", rho, e, k, b4, "forced"))
    return rows


def test_unlabelled_fabrications_at_rho_100_and_above():
    tests = by_name(forced_rows())
    u = tests["UFAB-L4vsB4-forced"]
    far = len(EPISODES) * 3  # rho 100, 1000, 10000
    assert u["role"] == "confirmatory" and u["n_items"] == 10 * far and (u["b"], u["c"]) == (0, far) and u["p"] == mcnemar_one_sided(0, far)
    g = tests["UFAB-L4vsB4-forced-with-guesses"]
    assert g["role"] == "exploratory" and (g["b"], g["c"]) == (0, 0)  # counted as answers, L4's guesses are as wrong as B4's
    assert tests["TOST-ACC-L4-forced"]["role"] == "confirmatory" and tests["TOST-ACC-L4-forced"]["estimate"] == 0


def test_sonnet_is_its_own_secondary_family():
    tests = hypothesis_tests(synthetic() + synthetic("claude-sonnet-5-5"), resamples=100, seed=0)
    for model, family in (("claude-haiku-5-5", "primary"), ("claude-sonnet-5-5", "secondary")):
        mine = [t for t in tests if t["model"] == model]
        confirmatory = [t for t in mine if t["role"] == "confirmatory"]
        assert {t["family"] for t in mine} == {family} and len(confirmatory) == 6 and sum(t["p"] is not None for t in confirmatory) == 4
        adjusted = holm({t["test"]: t["p"] for t in confirmatory})
        assert all(t["p_holm"] == adjusted[t["test"]] for t in confirmatory)


def test_l_vs_l4_is_on_corpus_questions_first_and_every_turn_second():
    rows = []
    for e in EPISODES:
        for rho in (10, 100):
            for k in range(10):
                kind = "session" if k >= 6 else "corpus"
                rows.append(row("L4", rho, e, k, "correct", kind=kind, origin=None if kind == "session" else "generator"))
                rows.append(row("L", rho, e, k, "abstain" if kind == "session" else "correct", kind=kind))
    tests = by_name(rows)
    corpus, every = tests["ACC-L4vsL-corpus"], tests["ACC-L4vsL-all"]
    assert corpus["n_items"] == len(EPISODES) * 2 * 6 and (corpus["b"], corpus["c"]) == (0, 0) and "primary L vs L4 contrast" in corpus["note"]
    assert every["n_items"] == len(EPISODES) * 2 * 10 and every["c"] == len(EPISODES) * 2 * 4 and "secondary" in every["note"]
    assert corpus["role"] == every["role"] == "exploratory"


def test_gate_and_index_effects_and_the_cost_ratio_are_exploratory():
    rows = synthetic() + [{**r, "arm": "L4-nogate", "outcome": "wrong" if r["qid"].endswith("-9") else r["outcome"]} for r in synthetic() if r["arm"] == "L4"]
    tests = by_name(rows)
    gate = tests["CE-L4vsL4-nogate"]
    assert gate["role"] == "exploratory" and (gate["b"], gate["c"]) == (0, N) and gate["verdict"] == "significant at 0.05 (exploratory, uncorrected)"
    assert tests["ACC-B5vsB3"]["n_items"] == 10 * N and (tests["ACC-B5vsB3"]["b"], tests["ACC-B5vsB3"]["c"]) == (0, 0)
    cost = tests["COST-L4/B4-allowed"]
    assert cost["estimate"] == Fraction(2, 3) and cost["verdict"] == "descriptive: 95% interval at or under 1.5x" and cost["p"] is None


def test_unverified_guesses_count_as_answers_primary_and_abstentions_gated():
    rows = [row("L4", 10, 1, 0, "wrong", "forced", forced_fallback=True, unverified_guess=True, reason="unverified_guess"),
            row("L4", 10, 1, 1, "correct", "forced"), row("L4", 10, 1, 2, "false_answer", "forced", forced_fallback=True, unverified_guess=True, kind="near_miss")]  # fmt: skip
    cells = {c["variant"]: c for c in cell_rows(with_variants(rows)) if c["subgroup"] == "all"}
    primary, gated = cells["primary"], cells["gated"]
    assert (primary["wrong"], primary["false_answer"], primary["unverified_guess_wrong"]) == (1, 1, 1)
    assert (gated["wrong"], gated["abstain"], gated["correct_abstain"], gated["unverified_guess_wrong"]) == (0, 1, 1, 1)
    assert primary["coverage"] == 1 and gated["coverage"] == Fraction(1, 2)
    assert primary["confident_error"] == Fraction(2, 3) and primary["fabrication"] == Fraction(2, 3)


def test_subgroup_cells_by_kind_and_origin():
    rows = [row("L4", 10, 1, 0, "correct", origin="generator"), row("L4", 10, 1, 1, "wrong", origin="constant_unperturbed"),
            row("L4", 10, 1, 2, "correct", kind="session", origin=None, template="session")]  # fmt: skip
    cells = {c["subgroup"]: c for c in cell_rows(with_variants(rows))}
    assert set(cells) == {"all", "corpus", "session", "origin:generator", "origin:constant_unperturbed"}
    assert (cells["all"]["n_answerable"], cells["corpus"]["n_answerable"], cells["session"]["n_answerable"]) == (3, 2, 1)
    assert cells["origin:constant_unperturbed"]["accuracy"] == 0 and cells["origin:generator"]["accuracy"] == 1


def test_the_files_are_written(tmp_path):
    result = analyze_v4(synthetic(), tmp_path, resamples=100)
    with open(tmp_path / "figure_data.csv") as f:
        figure = list(csv.DictReader(f))
    assert {r["arm"] for r in figure} == {"B3", "B4", "B5", "L4"} and {r["rho"] for r in figure} == {str(r) for r in RHOS} and "subgroup" not in figure[0]
    assert len(figure) == 4 * len(RHOS)  # subgroup "all" only
    meta = json.loads((tmp_path / "tests.json").read_text())["meta"]
    assert meta["confirmatory"] == list(CONFIRMATORY) and meta["ni_margin"] == "0.03" and meta["ufab_min_rho"] == 100 and "PREREG-v4" in meta["status"]
    with open(tmp_path / "tests.csv") as f:
        assert {r["role"] for r in csv.DictReader(f)} == {"confirmatory", "exploratory"}
    assert {c["subgroup"] for c in result["cells"]} == {"all", "corpus", "origin:generator"}


# BUILD-V4FIX: reruns (as analyze-v3's, 47abe39). A unit (model, W, condition, arm, rho, episode) incomplete in a run is
# dropped (dropped_units); a unit complete in several runs is taken from the run given last (superseded_units).


def v4_units(dirs):
    from context_ledger.eval.analyze_v4 import load
    from test_analyze_v3 import UNIT_QS

    rows, superseded, dropped = load(dirs, lambda run: (UNIT_QS, {}))
    return sorted({(r["arm"], r["episode"], r["cost_pusd"]) for r in rows}), len(rows), superseded, dropped


def test_load_v4_drops_the_unit_a_run_stopped_in_and_takes_it_from_the_fix1_run(tmp_path):
    from test_analyze_v3 import unit, write_run

    stopped = write_run(tmp_path, "v4-grid-sonnet-forced-r10-e6-7-c260", {("B3", 6): 3, ("B3", 7): 3, ("L", 6): 2}, stopped=("L", 6), partial={("L", 6)})
    rerun = write_run(tmp_path, "v4-grid-sonnet-forced-r10-e6-7-c260-fix1", {("L", 6): 3, ("L", 7): 3, ("L4", 6): 3, ("L4", 7): 3}, cost=2)
    assert sorted([rerun, stopped]) == [stopped, rerun]  # the shell glob runs/v4-grid-*-c260* gives a -fix1 run after the run it reruns
    for order in ([stopped, rerun], [rerun, stopped]):
        taken, n, superseded, dropped = v4_units(order)
        assert taken == [("B3", 6, 1), ("B3", 7, 1), ("L", 6, 2), ("L", 7, 2), ("L4", 6, 2), ("L4", 7, 2)] and n == 18 and superseded == []
        assert dropped == [{"unit": unit("L", 6), "run": "v4-grid-sonnet-forced-r10-e6-7-c260", "rows": 2, "questions": 3, "reason": "run_stopped"}]


def test_load_v4_takes_a_unit_complete_in_two_runs_from_the_later(tmp_path):
    from test_analyze_v3 import unit, write_run

    first = write_run(tmp_path, "v4-grid-x-c260", {("L4", 6): 3, ("L4", 7): 3})
    later = write_run(tmp_path, "v4-grid-x-c260-fix1", {("L4", 6): 3}, cost=2)
    taken, n, superseded, dropped = v4_units([first, later])
    assert taken == [("L4", 6, 2), ("L4", 7, 1)] and n == 6 and dropped == []
    assert superseded == [{"unit": unit("L4", 6), "run": "v4-grid-x-c260", "replaced_by": "v4-grid-x-c260-fix1"}]


def test_load_v4_drops_a_short_unit_and_a_stopped_later_run_never_replaces_a_complete_one(tmp_path):
    from test_analyze_v3 import unit, write_run

    short = write_run(tmp_path, "short", {("L", 6): 2, ("L", 7): 3})
    taken, _, _, dropped = v4_units([short])
    assert taken == [("L", 7, 1)] and dropped == [{"unit": unit("L", 6), "run": "short", "rows": 2, "questions": 3, "reason": "missing_rows"}]
    complete = write_run(tmp_path, "complete", {("L4", 6): 3})
    stopped = write_run(tmp_path, "complete-fix1", {("L4", 6): 3}, stopped=("L4", 6), cost=2)  # every row there, but the run stopped in it
    taken, _, superseded, dropped = v4_units([complete, stopped])
    assert taken == [("L4", 6, 1)] and superseded == [] and [d["reason"] for d in dropped] == ["run_stopped"]


# PREREG-v4-AMENDMENT-2: Sonnet's L4-nogate arm. Its rows give Sonnet the gate ablation Haiku has (the same tests, exploratory,
# secondary family), and leave every other row, the six confirmatory tests of both families with Holm among them included, as it was.

HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
GATE_TESTS = {"ACC-L4vsL4-nogate", "CE-L4vsL4-nogate", "UFAB-L4vsL4-nogate-forced", "TOST-ACC-L4-nogate-allowed", "TOST-ACC-L4-nogate-forced",
              "SLOPE-FAB-L4-nogate-allowed", "SLOPE-FAB-L4-nogate-forced"}  # fmt: skip


def nogate_rows(model: str) -> list[dict]:
    """L4's rows as L4-nogate's: allowed, item 9 answered wrong (more confident errors); forced, item 0 a wrong answer not flagged
    unverified_guess (an unlabelled fabrication, where L4's is a labelled guess)."""
    rows = [r for r in synthetic(model) + [{**r, "model": model} for r in forced_rows()] if r["arm"] == "L4"]
    out = []
    for r in rows:
        wrong = r["qid"].endswith("-9") if r["condition"] == "allowed" else r["qid"].endswith("-0")
        out.append({**r, "arm": "L4-nogate", **({"outcome": "wrong", "unverified_guess": False, "forced_fallback": False} if wrong else {})})
    return out


def grid_rows(model: str) -> list[dict]:
    return synthetic(model) + [{**r, "model": model} for r in forced_rows()]


def test_sonnet_l4_nogate_adds_the_gate_ablation_and_changes_no_other_row():
    before = grid_rows(HAIKU) + nogate_rows(HAIKU) + grid_rows(SONNET)  # as analysed so far: Haiku has L4-nogate, Sonnet not
    after = before + nogate_rows(SONNET)
    old = {(t["model"], t["test"]): t for t in hypothesis_tests(before, resamples=200, seed=0)}
    new = {(t["model"], t["test"]): t for t in hypothesis_tests(after, resamples=200, seed=0)}
    assert set(new) - set(old) == {(SONNET, name) for name in GATE_TESTS}
    assert all(new[k] == old[k] for k in old)  # every row as it was, both confirmatory families and their Holm-adjusted p included
    confirmatory = [new[(SONNET, name)] for name in CONFIRMATORY]
    assert all(t["p_holm"] is not None for t in confirmatory if t["p"] is not None) and {t["family"] for t in confirmatory} == {"secondary"}
    gate = {name: new[(SONNET, name)] for name in GATE_TESTS}
    assert all(t["role"] == "exploratory" and t["family"] == "secondary" and t["p_holm"] is None for t in gate.values())
    assert {name for model, name in new if model == HAIKU} >= GATE_TESTS  # Haiku's gate ablation: the same tests
    assert (gate["CE-L4vsL4-nogate"]["b"], gate["CE-L4vsL4-nogate"]["c"]) == (0, N) and gate["CE-L4vsL4-nogate"]["verdict"] == "significant at 0.05 (exploratory, uncorrected)"
    assert (gate["UFAB-L4vsL4-nogate-forced"]["b"], gate["UFAB-L4vsL4-nogate-forced"]["c"]) == (0, N) and gate["UFAB-L4vsL4-nogate-forced"]["n_items"] == 10 * N
    assert gate["ACC-L4vsL4-nogate"]["n_items"] == 10 * N and (gate["ACC-L4vsL4-nogate"]["b"], gate["ACC-L4vsL4-nogate"]["c"]) == (0, 0)  # item 9 abstains in L4: a miss in both
    assert gate["TOST-ACC-L4-nogate-allowed"]["estimate"] == 0 and gate["TOST-ACC-L4-nogate-allowed"]["verdict"] == "equivalent within +/-2 pp per decade (exploratory, uncorrected)"


def test_sonnet_l4_nogate_cells_are_haikus_cells_and_leave_the_others_unchanged():
    before = grid_rows(HAIKU) + nogate_rows(HAIKU) + grid_rows(SONNET)
    key = lambda c: tuple(c[k] for k in ("model", "window", "condition", "arm", "variant", "subgroup", "rho"))  # noqa: E731
    old = {key(c): c for c in cell_rows(with_variants(before))}
    new = {key(c): c for c in cell_rows(with_variants(before + nogate_rows(SONNET)))}
    added = set(new) - set(old)
    assert {k[3] for k in added} == {"L4-nogate"} and {k[0] for k in added} == {SONNET} and all(new[k] == old[k] for k in old)
    shape = lambda model: {k[2:] for k in new if k[0] == model and k[3] == "L4-nogate"}  # noqa: E731
    assert shape(SONNET) == shape(HAIKU) and {k[4] for k in added} == {"primary"}  # as Haiku's: no gated variant (the gate is off)


def test_load_v4_adds_the_nogate1_units_beside_the_original_and_fix1_units(tmp_path):
    from test_analyze_v3 import unit, write_run

    original = write_run(tmp_path, "v4-grid-sonnet-allowed-r10-e6-7-c260", {("B3", 6): 3, ("B3", 7): 3, ("L4", 6): 2}, stopped=("L4", 6), partial={("L4", 6)})
    fix1 = write_run(tmp_path, "v4-grid-sonnet-allowed-r10-e6-7-c260-fix1", {("L4", 6): 3, ("L4", 7): 3}, cost=2)
    nogate1 = write_run(tmp_path, "v4-grid-sonnet-allowed-r10-e6-7-c260-nogate1", {("L4-nogate", 6): 3, ("L4-nogate", 7): 3}, cost=3)
    taken, n, superseded, dropped = v4_units([original, fix1, nogate1])  # run-sonnet-nogate.sh's order
    assert taken == [("B3", 6, 1), ("B3", 7, 1), ("L4", 6, 2), ("L4", 7, 2), ("L4-nogate", 6, 3), ("L4-nogate", 7, 3)] and n == 18
    assert superseded == [] and dropped == [{"unit": unit("L4", 6), "run": "v4-grid-sonnet-allowed-r10-e6-7-c260", "rows": 2, "questions": 3, "reason": "run_stopped"}]
    assert v4_units([original, fix1]) == (taken[:4], 12, superseded, dropped)  # the -nogate1 run adds its units and changes no other
    (tmp_path / "halted").mkdir()  # a partial arm: the -nogate1 run stopped in episode 7; that unit is dropped, never half-counted
    halted = write_run(tmp_path / "halted", nogate1.name, {("L4-nogate", 6): 3, ("L4-nogate", 7): 1}, stopped=("L4-nogate", 7), partial={("L4-nogate", 7)}, cost=3)
    taken, n, superseded, dropped = v4_units([original, fix1, halted])
    assert taken[-1] == ("L4-nogate", 6, 3) and n == 15 and superseded == []
    assert [(d["unit"], d["reason"]) for d in dropped] == [(unit("L4", 6), "run_stopped"), (unit("L4-nogate", 7), "run_stopped")]
