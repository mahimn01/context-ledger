"""`eval analyze-v4`: the analysis of v4 runs (BUILD-V4 item 9; its confirmatory family is BUILD-V4B
decisions 4-5, drafted in PREREG-v4). Stdlib only, read-only on runs/; analyze-v3 is unchanged and
its statistics are reused here.

Rows. Every results row scored by the one unchanged grader (score.py), tagged with model, W
(run.json's window), condition, arm and rho, plus what v4 recorded (unverified_guess, repairs,
the pre-resolution status) and the question's kind (corpus, session or near_miss) and origin
(questions v4: the v3 generator or an extra). Units (model, W, condition, arm, rho, episode) are
loaded as analyze-v3 loads them: a unit incomplete in a run is dropped, a unit complete in several
runs is taken from the run given last. Infrastructure errors are left out.

Variants. L and L4 under the forced condition are reported twice in the cells: "primary", the
fallback answers (L's forced_fallback, L4's unverified_guess) counted as answers, graded by the
grader; "gated", those rows counted as abstentions. Every other cell has the primary variant only.

Outcomes, per cell (cells.csv; per rho, and per subgroup: all, corpus, session, origin:<x>), on
answerable (A) and unanswerable (U) items:
    accuracy           correct / A
    coverage           (correct + wrong) / A: the share of answerable questions answered
    confident_error    (wrong + false_answer) / answered: the share of emitted answers that are wrong
    fabrication        (wrong + false_answer) / (A + U), as in v3
    lookalike / stale  the confident-error rate on look-alike items (confusers > 0 at the row's rho,
                       plus unanswerable corpus near misses) and on stale items (superseded session probes)
    cost               median and mean USD per question
and the counts behind them, the forced flags, unverified guesses with their grades, repairs by
reason and the pre-resolution outcomes.

CONFIRMATORY family, per model and W (Holm within it; Haiku's family primary, any other model's,
Sonnet's, secondary). Items are (rho, question) pairs both arms ran; bootstraps resample episodes
(the clusters every arm, rho and condition share; 10,000 resamples, seed 0):
    NI-ACC-L4vsB3          allowed, rho pooled: acc(L4) - acc(B3) on the answerable items both
                           ran, item-paired; one-sided non-inferiority with margin 3 pp: p = the
                           share of resamples whose paired difference is <= -0.03 (non-inferior
                           when p < 0.05: the 90% interval's lower bound above -0.03)
    CE-L4vsB3, CE-L4vsB4   allowed, rho pooled: confident errors (a wrong answer to an answerable
                           item, a false answer to an unanswerable one; v3's fabrication over all
                           items), exact item-paired McNemar, one-sided for L4 fewer
    TOST-ACC-L4-allowed,   the OLS slope of L4's accuracy (per-episode accuracy, mean over episodes
    TOST-ACC-L4-forced     per rho) on log10 rho, its episode-cluster bootstrap; TOST within +/-2 pp
                           per decade: p = max(share <= -0.02, share >= +0.02), equivalent when
                           p < 0.05 (the 90% interval inside the margin); forced counts unverified
                           guesses as answers (the primary variant)
    UFAB-L4vsB4-forced     forced, rho >= 100: UNLABELLED fabrications (a confident error not flagged
                           unverified_guess; B4 flags none), exact item-paired McNemar, one-sided for
                           L4 fewer
EXPLORATORY (labelled so, raw p): UFAB-L4vsB4-forced-with-guesses (unverified guesses counted);
COST-L4/B4-<cond>, the median per-question cost ratio with its interval, descriptive against 1.5x;
the index effect ACC-B5vsB3 and CE-B5vsB3; the gate effect ACC-, CE- and UFAB-L4vsL4-nogate; L vs
L4 (decision 4): ACC-L4vsL-corpus and CE-L4vsL-corpus on corpus questions (the asked target is not
a session note; the primary L vs L4 contrast) and -all on every turn (secondary); TOST-ACC and
SLOPE-FAB for every arm and condition; CE-L4vsB3 and CE-L4vsB4 on look-alike and stale items;
and every per-rho and subgroup cell.

figure_data.csv: per cell of subgroup "all", the correct, wrong and abstain shares on answerable
items, the false-answer share on unanswerable ones, accuracy and coverage. main_figure.png as in
analyze-v3 (only if matplotlib is importable).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

from ..pricing import PICO_PER_USD
from .analyze import ALPHA, HAIKU, Draws, _csv, holm, log10_exact, median, ols_slope, resampled
from .analyze_v3 import RESAMPLES_V3, SEED_V3, UNIT_KEYS, _cell, _counts, _rate, confident_error, main_figure, mcnemar_test, prepare
from .stats import nearest_rank

TOST_MARGIN = Fraction(2, 100)  # 2 percentage points per decade of rho
NI_MARGIN = Fraction(3, 100)  # L4's accuracy may be at most 3 pp below B3's
UFAB_MIN_RHO = 100  # the unlabelled-fabrication test pools rho >= 100
COST_REFERENCE = Fraction(3, 2)  # the cost ratio is read against 1.5x (descriptive)
CONFIRMATORY = ("NI-ACC-L4vsB3", "CE-L4vsB3", "CE-L4vsB4", "TOST-ACC-L4-allowed", "TOST-ACC-L4-forced", "UFAB-L4vsB4-forced")
HOLM_V4 = CONFIRMATORY
CELL_KEYS = ("model", "window", "condition", "arm", "variant", "subgroup", "rho")
GATED_ARMS = ("L", "L4")


def prepare_v4(run: dict, results: list[dict], questions: list[dict], texts: dict[str, str]) -> list[dict]:
    """analyze_v3.prepare's rows plus the v4 fields of each results row and its question's kind and
    origin."""
    rows = prepare(run, results, questions, texts)
    by_qid = {q["qid"]: q for q in questions}
    for row, r in zip(rows, results):
        q = by_qid[r["qid"]]
        kind = "session" if q["template"] == "session" else "corpus" if q["answerable"] else "near_miss"
        row.update({"unverified_guess": bool(r.get("unverified_guess")), "repairs": r.get("repairs") or 0, "repair_reasons": list(r.get("repair_reasons") or []),
                    "preresolved": (r.get("preresolved") or {}).get("status"), "reason": r.get("reason"), "kind": kind, "origin": q.get("origin")})  # fmt: skip
    return rows


def load(run_dirs: Sequence[Path], questions_for: Callable[[dict], tuple[list[dict], dict[str, str]]]) -> tuple[list[dict], list[dict], list[dict]]:
    """(analysis rows, superseded units, dropped units): analyze_v3.load's rules with prepare_v4."""
    units: dict[tuple, tuple[str, list[dict]]] = {}
    superseded, dropped = [], []
    for run_dir in run_dirs:
        for path in sorted({Path(run_dir) / "run.json", *Path(run_dir).glob("*/run.json")}):
            if not path.exists():
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            results_path = path.parent / "results.jsonl"
            results = [json.loads(x) for x in results_path.read_text(encoding="utf-8").splitlines()] if results_path.exists() else []
            questions, texts = questions_for(run)
            rows = prepare_v4(run, results, questions, texts)
            episode_of = {q["qid"]: q["episode"] for q in questions}
            stopped = run.get("stopped") or {}
            by_unit: dict[tuple, list[dict]] = defaultdict(list)
            for r in rows:
                by_unit[tuple(r[k] for k in UNIT_KEYS)].append(r)
            for unit, mine in sorted(by_unit.items(), key=lambda kv: str(kv[0])):
                arm, episode = unit[3], unit[5]
                asked = {qid for qid in (run.get("questions") or {}).get(arm, []) if episode_of.get(qid) == episode}
                stopped_in = (stopped.get("arm"), stopped.get("episode")) == (arm, episode) or any(r["partial"] for r in mine)
                if stopped_in or asked - {r["qid"] for r in mine}:
                    dropped.append({"unit": list(unit), "run": run["run_id"], "rows": len(mine), "questions": len(asked),
                                    "reason": "run_stopped" if stopped_in else "missing_rows"})  # fmt: skip
                    continue
                if unit in units:
                    superseded.append({"unit": list(unit), "run": units[unit][0], "replaced_by": run["run_id"]})
                units[unit] = (run["run_id"], mine)
    return [r for _, rows in units.values() for r in rows], superseded, dropped


def with_variants(rows: list[dict]) -> list[dict]:
    """Every row as "primary"; L's and L4's forced rows also as "gated", fallback answers abstaining."""
    out = []
    for r in rows:
        out.append({**r, "variant": "primary"})
        if r["arm"] in GATED_ARMS and r["condition"] == "forced":
            gated = ("abstain" if r["answerable"] else "correct_abstain") if r["forced_fallback"] else r["outcome"]
            out.append({**r, "variant": "gated", "outcome": gated, "graded": r["outcome"]})
    return out


def subgroups(r: dict) -> list[str]:
    """The cell subgroups a row is in: all, its kind's (corpus questions and their near misses as
    "corpus", session probes as "session"), and origin:<x> for a questions v4 corpus item."""
    return ["all", "session" if r.get("kind") == "session" else "corpus", *([f"origin:{r['origin']}"] if r.get("origin") else [])]


def cell_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        for sub in subgroups(r):
            groups[tuple(sub if k == "subgroup" else r[k] for k in CELL_KEYS)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        n = Counter(r["outcome"] for r in rs)
        answerable = n["correct"] + n["wrong"] + n["abstain"]
        unanswerable = n["false_answer"] + n["correct_abstain"]
        answered = n["correct"] + n["wrong"] + n["false_answer"]
        look, stale = [r for r in rs if r["lookalike"]], [r for r in rs if r["stale"]]
        guesses = Counter(r.get("graded", r["outcome"]) for r in rs if r["unverified_guess"])
        costs = [Fraction(r["cost_pusd"], PICO_PER_USD) for r in rs]
        out.append({
            **dict(zip(CELL_KEYS, key)), "n_episodes": len({r["episode"] for r in rs}), "n_answerable": answerable, "correct": n["correct"], "wrong": n["wrong"],
            "abstain": n["abstain"], "n_unanswerable": unanswerable, "false_answer": n["false_answer"], "correct_abstain": n["correct_abstain"],
            "accuracy": _rate(n["correct"], answerable), "coverage": _rate(n["correct"] + n["wrong"], answerable),
            "confident_error": _rate(n["wrong"] + n["false_answer"], answered), "fabrication": _rate(n["wrong"] + n["false_answer"], len(rs)),
            "n_lookalike": len(look), "lookalike_confident_error_rate": _rate(sum(map(confident_error, look)), len(look)),
            "n_stale": len(stale), "stale_confident_error_rate": _rate(sum(map(confident_error, stale)), len(stale)),
            "forced_violation": sum(r["forced_violation"] for r in rs), "forced_fallback": sum(r["forced_fallback"] for r in rs),
            "unverified_guess": sum(r["unverified_guess"] for r in rs), "unverified_guess_correct": guesses["correct"],
            "unverified_guess_wrong": guesses["wrong"], "unverified_guess_false_answer": guesses["false_answer"],
            "repaired_questions": sum(bool(r["repairs"]) for r in rs), "repairs": _counts(Counter(x for r in rs for x in r["repair_reasons"])),
            "preresolved": _counts(Counter(r["preresolved"] for r in rs if r["preresolved"])),
            "cost_per_question_median_usd": median(costs), "cost_per_question_mean_usd": sum(costs, Fraction(0)) / len(costs),
        })  # fmt: skip
    return out


def figure_rows(cells: list[dict]) -> list[dict]:
    return [{**{k: c[k] for k in CELL_KEYS if k != "subgroup"}, "n_answerable": c["n_answerable"], "correct_share": _rate(c["correct"], c["n_answerable"]),
             "wrong_share": _rate(c["wrong"], c["n_answerable"]), "abstain_share": _rate(c["abstain"], c["n_answerable"]),
             "n_unanswerable": c["n_unanswerable"], "false_answer_share": _rate(c["false_answer"], c["n_unanswerable"]),
             "accuracy": c["accuracy"], "coverage": c["coverage"]} for c in cells if c["subgroup"] == "all"]  # fmt: skip


def _test(name: str, model, window, family: str, condition: str, **kw) -> dict:
    row = {"test": name, "role": "confirmatory" if name in CONFIRMATORY else "exploratory", "model": model, "window": window, "family": family,
           "condition": condition, "estimate": None, "ci_low": None, "ci_high": None, "ci90_low": None, "ci90_high": None, "p": None, "p_kind": "one-sided",
           "p_holm": None, "verdict": "", "n_episodes": 0, "n_items": None, "b": None, "c": None, "valid_resamples": 0, "note": ""}  # fmt: skip
    row.update(kw)
    return row


def _ci90(values: Sequence[Fraction]) -> dict:
    return {"ci90_low": nearest_rank(values, Fraction(5)), "ci90_high": nearest_rank(values, Fraction(95))} if values else {}


def slope(name: str, rows: list[dict], metric: Callable[[list[dict]], tuple[int, int]], model, window, family, condition, resamples, seed, tost: bool) -> dict:
    """The OLS slope over log10 rho of a per-episode metric (mean over episodes per rho), its
    bootstrap intervals, and the TOST p (`tost`) or the one-sided p for a rise."""
    per: dict[tuple[int, int], Fraction] = {}
    by_unit: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_unit[(r["episode"], r["rho"])].append(r)
    for unit, rs in by_unit.items():
        num, den = metric(rs)
        if den:
            per[unit] = Fraction(num, den)
    rhos = sorted({rho for _, rho in by_unit})
    ids = sorted({e for e, _ in by_unit})
    kind = "TOST, margin +/-0.02 per decade" if tost else "one-sided"
    base = _test(name, model, window, family, condition, n_episodes=len(ids), p_kind=kind, note=f"OLS slope per decade of rho over rho {rhos}")
    if len(rhos) < 2 or not ids:
        return {**base, "note": "not run: fewer than two rho"}
    try:
        xs = {rho: log10_exact(rho) for rho in rhos}
    except ValueError:
        return {**base, "note": f"not run: rho {rhos} are not all powers of ten"}

    def stat(w: Counter) -> Fraction | None:
        cells = []
        for rho in rhos:
            got = [(k, per[(e, rho)]) for e, k in w.items() if (e, rho) in per]
            cells.append((xs[rho], sum(k * v for k, v in got) / sum(k for k, _ in got) if got else None))
        return ols_slope(cells)

    result = resampled(Draws(ids, resamples, seed), stat)
    values = sorted(result.values)
    row = {**base, "estimate": result.estimate, "ci_low": result.low, "ci_high": result.high, "valid_resamples": result.valid, **_ci90(values)}
    if not values:
        return row
    if tost:
        row["p"] = max(Fraction(sum(v <= -TOST_MARGIN for v in values), len(values)), Fraction(sum(v >= TOST_MARGIN for v in values), len(values)))
    else:
        row["p"] = Fraction(sum(v <= 0 for v in values), len(values))
    return row


def pairs(rows: list[dict], first: str, second: str, bad: Callable[[dict], bool], keep: Callable[[dict], bool] = lambda r: True) -> list[tuple[int, bool, bool]]:
    """(episode, bad under `first`, bad under `second`) per item (question and rho) both arms ran."""
    a = {(r["rho"], r["qid"]): r for r in rows if r["arm"] == first and keep(r)}
    b = {(r["rho"], r["qid"]): r for r in rows if r["arm"] == second and keep(r)}
    return [(a[k]["episode"], bad(a[k]), bad(b[k])) for k in sorted(set(a) & set(b), key=str)]


def miss(r: dict) -> bool:
    return r["answerable"] and r["outcome"] != "correct"


def unlabelled(r: dict) -> bool:
    """A confident error not flagged unverified_guess (a labelled guess is not an unlabelled fabrication)."""
    return confident_error(r) and not r["unverified_guess"]


def mcnemar(name: str, rows: list[dict], first: str, second: str, bad: Callable[[dict], bool], model, window, family, condition, note, resamples, seed,
            keep: Callable[[dict], bool] = lambda r: True) -> dict:  # fmt: skip
    """analyze_v3's exact item-paired McNemar (one-sided for `first` fewer) with its bootstrap of the paired difference."""
    t = mcnemar_test(name, pairs(rows, first, second, bad, keep), model, window, family, condition, note, resamples, seed)
    return {**t, "note": t["note"].replace("only L errs on", f"only {first} errs on").replace("only the other arm errs on", f"only {second} errs on")}


def noninferiority(name: str, rows: list[dict], first: str, second: str, model, window, family, condition, resamples, seed) -> dict:
    """acc(first) - acc(second) on the answerable items both ran (rho pooled), item-paired; the
    episode-cluster bootstrap of that difference; p = the share of resamples <= -NI_MARGIN."""
    items = pairs(rows, first, second, lambda r: r["outcome"] == "correct", lambda r: r["answerable"])
    by_episode: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for e, x, y in items:
        by_episode[e][0] += int(x) - int(y)
        by_episode[e][1] += 1
    base = _test(name, model, window, family, condition, n_episodes=len(by_episode), n_items=len(items), p_kind="one-sided non-inferiority, margin 0.03",
                 b=sum(x and not y for _, x, y in items), c=sum(y and not x for _, x, y in items),
                 note=f"accuracy {first} - {second} on the answerable items both ran, rho pooled; b = items only {first} gets right, c = only {second}")  # fmt: skip
    if not items:
        return {**base, "note": f"not run: no paired answerable items of {first} and {second}"}

    def stat(w: Counter) -> Fraction | None:
        num, den = (sum(k * by_episode[e][i] for e, k in w.items() if e in by_episode) for i in range(2))
        return Fraction(num, den) if den else None

    result = resampled(Draws(sorted(by_episode), resamples, seed), stat)
    values = sorted(result.values)
    return {**base, "estimate": result.estimate, "ci_low": result.low, "ci_high": result.high, "valid_resamples": result.valid, **_ci90(values),
            "p": Fraction(sum(v <= -NI_MARGIN for v in values), len(values)) if values else None}  # fmt: skip


def cost_ratio(name: str, rows: list[dict], first: str, second: str, model, window, family, condition, resamples, seed) -> dict:
    a_rows, b_rows = ([r for r in rows if r["arm"] == arm] for arm in (first, second))
    ids = sorted({r["episode"] for r in a_rows} & {r["episode"] for r in b_rows})
    base = _test(name, model, window, family, condition, n_episodes=len(ids), p_kind="",
                 note=f"median per-question cost {first} / {second}, rho pooled; descriptive, read against {float(COST_REFERENCE)}x")  # fmt: skip
    if not ids:
        return {**base, "note": f"not run: no episode in both {first} and {second}"}
    costs = {arm: defaultdict(list) for arm in (first, second)}
    for r in a_rows + b_rows:
        costs[r["arm"]][r["episode"]].append(r["cost_pusd"])

    def stat(w: Counter) -> Fraction | None:
        meds = [median([c for e, k in w.items() for _ in range(k) for c in costs[arm].get(e, [])]) for arm in (first, second)]
        return None if None in meds or meds[1] == 0 else meds[0] / meds[1]

    result = resampled(Draws(ids, resamples, seed), stat)
    return {**base, "estimate": result.estimate, "ci_low": result.low, "ci_high": result.high, "valid_resamples": result.valid}


def hypothesis_tests(rows: list[dict], resamples: int = RESAMPLES_V3, seed: int = SEED_V3) -> list[dict]:
    rows = [r for r in rows if not r["partial"]]
    out = []
    for model, window in sorted({(r["model"], r["window"]) for r in rows}, key=str):
        family = "primary" if model == HAIKU else "secondary"
        mine = [r for r in rows if (r["model"], r["window"]) == (model, window)]
        arms = {r["arm"] for r in mine}
        allowed = [r for r in mine if r["condition"] == "allowed"]
        forced = [r for r in mine if r["condition"] == "forced"]
        far = [r for r in forced if r["rho"] >= UFAB_MIN_RHO]
        acc = lambda rs: (sum(r["outcome"] == "correct" for r in rs), sum(r["answerable"] for r in rs))  # noqa: E731
        fab = lambda rs: (sum(map(confident_error, rs)), len(rs))  # noqa: E731
        both = lambda a, b: a in arms and b in arms  # noqa: E731
        kw = {"model": model, "window": window, "family": family, "resamples": resamples, "seed": seed}
        tests = []
        # the confirmatory family (module docstring)
        if both("L4", "B3"):
            tests.append(noninferiority("NI-ACC-L4vsB3", allowed, "L4", "B3", condition="allowed", **kw))
        for other in ("B3", "B4"):
            if both("L4", other):
                tests.append(mcnemar(f"CE-L4vs{other}", allowed, "L4", other, confident_error, condition="allowed",
                                     note=f"confident errors (wrong or false answer, all items), L4 vs {other}, rho pooled", **kw))  # fmt: skip
        for condition, these in (("allowed", allowed), ("forced", forced)):
            if "L4" in arms:  # listed (as not run) when a condition has no L4 rows
                tests.append(slope(f"TOST-ACC-L4-{condition}", [r for r in these if r["arm"] == "L4"], acc, condition=condition, tost=True, **kw))
        if both("L4", "B4"):
            note = f"unlabelled fabrications (confident errors not flagged unverified_guess), L4 vs B4, rho >= {UFAB_MIN_RHO}"
            tests.append(mcnemar("UFAB-L4vsB4-forced", far, "L4", "B4", unlabelled, condition="forced", note=note, **kw))
            # exploratory from here on
            tests.append(mcnemar("UFAB-L4vsB4-forced-with-guesses", far, "L4", "B4", confident_error, condition="forced",
                                 note=f"fabrications with L4's unverified guesses counted, L4 vs B4, rho >= {UFAB_MIN_RHO}", **kw))  # fmt: skip
            for condition, these in (("allowed", allowed), ("forced", forced)):
                tests.append(cost_ratio(f"COST-L4/B4-{condition}", these, "L4", "B4", condition=condition, **kw))
        for first, second, label in (("B5", "B3", "the index effect"), ("L4", "L4-nogate", "the gate effect")):
            if both(first, second):
                tests.append(mcnemar(f"ACC-{first}vs{second}", allowed, first, second, miss, condition="allowed", keep=lambda r: r["answerable"],
                                     note=f"{label}: answerable items not answered correctly, rho pooled", **kw))  # fmt: skip
                tests.append(mcnemar(f"CE-{first}vs{second}", allowed, first, second, confident_error, condition="allowed", note=f"{label}: confident errors, rho pooled", **kw))
        if both("L4", "L4-nogate"):
            tests.append(mcnemar("UFAB-L4vsL4-nogate-forced", forced, "L4", "L4-nogate", unlabelled, condition="forced",
                                 note="the gate effect: unlabelled fabrications, rho pooled", **kw))  # fmt: skip
        if both("L4", "L"):  # decision 4: corpus questions are the primary L vs L4 contrast, every turn the secondary
            for suffix, keep, which in (("-corpus", lambda r: r["kind"] != "session", "primary L vs L4 contrast: corpus questions"),
                                        ("-all", lambda r: True, "secondary L vs L4 contrast: every turn")):  # fmt: skip
                tests.append(mcnemar(f"ACC-L4vsL{suffix}", allowed, "L4", "L", miss, condition="allowed", keep=lambda r, keep=keep: r["answerable"] and keep(r),
                                     note=f"{which}; answerable items not answered correctly, rho pooled", **kw))  # fmt: skip
                tests.append(mcnemar(f"CE-L4vsL{suffix}", allowed, "L4", "L", confident_error, condition="allowed", keep=keep, note=f"{which}; confident errors, rho pooled", **kw))
        for other in ("B3", "B4"):
            if both("L4", other):
                for suffix, keep in (("-lookalike", lambda r: r["lookalike"]), ("-stale", lambda r: r["stale"])):
                    tests.append(mcnemar(f"CE-L4vs{other}{suffix}", allowed, "L4", other, confident_error, condition="allowed", keep=keep,
                                         note=f"confident errors on {suffix[1:]} items, L4 vs {other}, rho pooled", **kw))  # fmt: skip
        for condition, these in (("allowed", allowed), ("forced", forced)):
            for arm in sorted({r["arm"] for r in these}):
                mine_arm = [r for r in these if r["arm"] == arm]
                if arm != "L4":
                    tests.append(slope(f"TOST-ACC-{arm}-{condition}", mine_arm, acc, condition=condition, tost=True, **kw))
                tests.append(slope(f"SLOPE-FAB-{arm}-{condition}", mine_arm, fab, condition=condition, tost=False, **kw))
        tests = [{**_test(t["test"], model, window, family, t["condition"]), **t} for t in tests]
        adjusted = holm({t["test"]: t["p"] for t in tests if t["role"] == "confirmatory"})
        for t in tests:
            if t["role"] == "confirmatory":
                t["p_holm"] = adjusted[t["test"]]
            t["verdict"] = _verdict(t)
        out += tests
    return out


def _verdict(t: dict) -> str:
    if t["note"].startswith("not run"):
        return "not run"
    if t["test"].startswith("COST-"):
        if t["ci_high"] is None:
            return "not estimable"
        return f"descriptive: 95% interval {'at or under' if t['ci_high'] <= COST_REFERENCE else 'reaches above'} {float(COST_REFERENCE)}x"
    confirmatory = t["role"] == "confirmatory"
    p = t["p_holm"] if confirmatory else t["p"]
    tail = "" if confirmatory else " (exploratory, uncorrected)"
    if p is None:
        return "not estimable"
    if t["test"].startswith("TOST-"):
        return ("equivalent within +/-2 pp per decade" if p < ALPHA else "equivalence not shown at this n") + tail
    if t["test"].startswith("NI-"):
        return ("non-inferior within 3 pp" if p < ALPHA else "non-inferiority not shown at this n") + tail
    if p < ALPHA:
        return "supported" if confirmatory else "significant at 0.05 (exploratory, uncorrected)"
    return "not detected at this n" + tail


CELL_COLUMNS = [*CELL_KEYS, "n_episodes", "n_answerable", "correct", "wrong", "abstain", "n_unanswerable", "false_answer", "correct_abstain", "accuracy", "coverage",
                "confident_error", "fabrication", "n_lookalike", "lookalike_confident_error_rate", "n_stale", "stale_confident_error_rate", "forced_violation",
                "forced_fallback", "unverified_guess", "unverified_guess_correct", "unverified_guess_wrong", "unverified_guess_false_answer", "repaired_questions",
                "repairs", "preresolved", "cost_per_question_median_usd", "cost_per_question_mean_usd"]  # fmt: skip
TEST_COLUMNS = ["test", "role", "model", "window", "family", "condition", "estimate", "ci_low", "ci_high", "ci90_low", "ci90_high", "p", "p_kind", "p_holm",
                "verdict", "n_episodes", "n_items", "b", "c", "valid_resamples", "note"]  # fmt: skip
FIGURE_COLUMNS = [k for k in CELL_KEYS if k != "subgroup"] + ["n_answerable", "correct_share", "wrong_share", "abstain_share", "n_unanswerable", "false_answer_share",
                                                              "accuracy", "coverage"]  # fmt: skip
COST_COLUMNS = ("cost_per_question_median_usd", "cost_per_question_mean_usd")


def analyze_v4(rows: list[dict], out: Path, *, resamples: int = RESAMPLES_V3, seed: int = SEED_V3, meta: dict | None = None) -> dict:
    """Write cells.csv, tests.csv, tests.json and figure_data.csv (and main_figure.png if
    matplotlib is importable) from rows of `load`; returns the cells, tests and figure path."""
    rows = [r for r in rows if not r["error"]]
    cells = cell_rows(with_variants(rows))
    tests = hypothesis_tests(rows, resamples, seed)
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / "cells.csv", CELL_COLUMNS, [[_cell(c[k], 9 if k in COST_COLUMNS else 6) for k in CELL_COLUMNS] for c in cells])
    _csv(out / "tests.csv", TEST_COLUMNS, [[_cell(t[k]) for k in TEST_COLUMNS] for t in tests])
    shown = [{k: _cell(t[k]) for k in TEST_COLUMNS} for t in tests]
    status = "PREREG-v4 (draft): the confirmatory family is six tests per model and W, Holm within it (Haiku primary, Sonnet secondary); every other row is exploratory"
    (out / "tests.json").write_text(json.dumps({"meta": {"resamples": resamples, "seed": seed, "alpha": "0.05", "tost_margin": "0.02", "ni_margin": "0.03",
                                                         "ufab_min_rho": UFAB_MIN_RHO, "confirmatory": list(CONFIRMATORY), "status": status, **(meta or {})},
                                                "tests": shown}, indent=1, sort_keys=True) + "\n", encoding="utf-8")  # fmt: skip
    _csv(out / "figure_data.csv", FIGURE_COLUMNS, [[_cell(f[k]) for k in FIGURE_COLUMNS] for f in figure_rows(cells)])
    return {"cells": cells, "tests": tests, "figure": main_figure(out)}


__all__ = ["CONFIRMATORY", "HOLM_V4", "NI_MARGIN", "TOST_MARGIN", "UFAB_MIN_RHO", "analyze_v4", "cell_rows", "hypothesis_tests", "load", "noninferiority", "pairs",
           "prepare_v4", "slope", "unlabelled", "with_variants"]
