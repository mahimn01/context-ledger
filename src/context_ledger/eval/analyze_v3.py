"""`eval analyze-v3`: the PREREG-v3 analysis of v3 runs (BUILD-V3 item 4), stdlib only, read-only on runs/.

Inputs are the results rows of one or more runs, each scored by the one unchanged grader (score.py)
and tagged with its model, W (run.json's window), condition, arm and rho. A question whose call
failed for an infrastructure reason (the row's error) is left out of everything. `load` drops a
unit (model, W, condition, arm, rho, episode) that is incomplete in a run, the one it stopped in
or one short of rows (listed in the meta's dropped_units), and takes a unit complete in several
runs from the run given last (the earlier ones in superseded_units). Rows flagged partial that
reach `analyze_v3` some other way count in the cells and are left out of the tests.

Variants. L under the forced condition is reported two ways: "primary", its fallback answers
(forced_fallback) counted as answers, graded as the grader graded them; and "gated", those rows
counted as abstentions (abstain on answerable, correct_abstain on unanswerable). Every other cell
has only the primary variant.

Item sets. A confident error is a wrong answer on an answerable question or a false answer on an
unanswerable one. Look-alike items: questions with confusers > 0 at the row's rho (a session
probe's confusers are the earlier notes sharing its service or attribute), plus every unanswerable
corpus question (a near miss of a real name). Stale items: session probes whose fact was updated
(superseded) before the probe.

cells.csv, per model, W, condition, arm, variant and rho: n answerable, correct, wrong, abstain;
n unanswerable, false_answer, correct_abstain; forced_violation and how many of those
abstentions were on unanswerable questions (correct_abstain); forced_fallback and the grader's
grades of the fallback answers (in both variants); hr = wrong / answerable; fabrication =
(wrong + false_answer) / all; abstain rate = abstain / answerable; the confident-error rate on
look-alike and on stale items; the median and mean cost per question (USD); and L's final gate
reasons and gate refusals.

tests.csv and tests.json (PREREG-v3), per model and W; bootstraps resample episodes, the clusters
shared by every arm, rho and condition (10,000 resamples, seed 0, 95% percentile intervals by
nearest rank):
    H1-B3, H1-B4          forced: per-episode HR on answerable at each rho, its mean over the
                          episodes per rho, and the OLS slope of those cells on log10 rho; one-sided
                          p = share of resamples with slope <= 0
    H1sec-B3, H1sec-B4    the same for fabrication (unanswerable questions included)
    H1a-B3, H1a-B4        allowed: the same for the abstention rate
    H2-vsB4, H2-vsB3      allowed: confident errors of L against the baseline on the same items
                          (question and rho), pooled over rho; exact McNemar, one-sided for L fewer:
                          p = P(X <= b), X ~ Binomial(b + c, 1/2), b = items only L gets wrong,
                          c = items only the baseline gets wrong; estimate (b - c) / items with its
                          paired episode-cluster bootstrap interval
    H2-vsB4-lookalike, H2-vsB4-stale, H2-vsB3-lookalike, H2-vsB3-stale   the same on those items
    H3                    forced, rho >= 100: fabrication of L (primary) against B4, McNemar as H2
    H3-gated              the same with L's gated variant
    H4                    allowed: median over questions of the per-question cost, L / B4 (the
                          episodes both ran, rho pooled); judged by its interval: supported if the
                          upper bound <= 1.5; p = share of resamples >= 1.5
Holm's step-down adjustment runs over {H1-B3, H1-B4, H2-vsB4, H2-vsB3, H3} at alpha 0.05 per model
and W: Haiku is the primary family, any other model (Sonnet) a separate secondary family. A Holm
test with adjusted p < 0.05 is "supported", else "not detected at this n"; the other tests report
raw p ("significant at 0.05 (uncorrected)" or "not detected at this n").

figure_data.csv, per model, W, condition, arm, variant and rho: the correct, wrong and abstain
shares on answerable questions and the false-answer share on unanswerable ones. main_figure.png
is drawn from figure_data.csv alone, and only if matplotlib is already importable (no
dependency); otherwise it is skipped.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from fractions import Fraction
from math import comb
from pathlib import Path

from ..pricing import PICO_PER_USD
from ..tokens import decimal_str
from .analyze import ALPHA, H4_BOUND, HAIKU, Draws, _csv, holm, log10_exact, median, ols_slope, resampled
from .score import score

SEED_V3 = 0
RESAMPLES_V3 = 10_000
HOLM_V3 = ("H1-B3", "H1-B4", "H2-vsB4", "H2-vsB3", "H3")
CELL_KEYS = ("model", "window", "condition", "arm", "variant", "rho")


def prepare(run: dict, results: list[dict], questions: list[dict], texts: dict[str, str]) -> list[dict]:
    """One analysis row per results row of a run part: the grader's outcome plus the tags."""
    by_qid = {q["qid"]: q for q in questions}
    rows = []
    for r, s in zip(results, score(results, questions, texts)):
        q = by_qid[r["qid"]]
        confusers = (q.get("confusers") or {}).get(str(run["rho"]), r.get("confusers")) or 0
        rows.append({
            "model": run["model"], "window": run.get("window"), "condition": r.get("condition") or run.get("condition", "allowed"), "arm": r["arm"],
            "rho": run["rho"], "episode": r["episode"], "qid": r["qid"], "answerable": q["answerable"], "outcome": s["outcome"],
            "template": q["template"], "lookalike": confusers > 0 or (not q["answerable"] and q["template"] != "session"),
            "stale": q["template"] == "session" and bool(q.get("superseded")), "cost_pusd": r["cost_pusd"],
            "forced_violation": bool(r.get("forced_violation")), "forced_fallback": bool(r.get("forced_fallback")),
            "gate_reason": r.get("gate_reason"), "gate_reasons": list(r.get("gate_reasons") or []), "partial": bool(r.get("partial")),
            "error": r.get("error") is not None,
        })  # fmt: skip
    return rows


UNIT_KEYS = ("model", "window", "condition", "arm", "rho", "episode")


def load(run_dirs: Sequence[Path], questions_for: Callable[[dict], tuple[list[dict], dict[str, str]]]) -> tuple[list[dict], list[dict], list[dict]]:
    """(analysis rows, superseded units, dropped units) of every run part under `run_dirs`, in the
    order given (a shell glob sorts a rerun, runs/<id>-Lrerun, after the run it reruns). Reads
    run.json and results.jsonl only.

    A unit is (model, W, condition, arm, rho, episode). A unit incomplete in a run part, the one
    the run stopped in (run.json's `stopped`, its rows flagged partial) or with fewer rows than
    the episode's questions in run.json, is dropped from that part and listed in dropped units
    (PREREG-v3 section 8). A unit complete in several parts is taken from the part given last;
    each earlier one is listed in superseded units."""
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
            rows = prepare(run, results, questions, texts)
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
    """Every row as variant "primary"; L's forced rows also as "gated", fallback answers abstaining
    (a gated row's "graded" keeps the grader's outcome)."""
    out = []
    for r in rows:
        out.append({**r, "variant": "primary"})
        if r["arm"] == "L" and r["condition"] == "forced":
            gated = ("abstain" if r["answerable"] else "correct_abstain") if r["forced_fallback"] else r["outcome"]
            out.append({**r, "variant": "gated", "outcome": gated, "graded": r["outcome"]})
    return out


def confident_error(r: dict) -> bool:
    return r["outcome"] in ("wrong", "false_answer")


def _rate(num: int, den: int) -> Fraction | None:
    return Fraction(num, den) if den else None


# Cells


def cell_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in CELL_KEYS)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        n = Counter(r["outcome"] for r in rs)
        answerable = n["correct"] + n["wrong"] + n["abstain"]
        unanswerable = n["false_answer"] + n["correct_abstain"]
        look, stale = [r for r in rs if r["lookalike"]], [r for r in rs if r["stale"]]
        fallback = Counter(r.get("graded", r["outcome"]) for r in rs if r["forced_fallback"])
        costs = [Fraction(r["cost_pusd"], PICO_PER_USD) for r in rs]
        l_rows = rs[0]["arm"] == "L"
        out.append({
            **dict(zip(CELL_KEYS, key)), "n_episodes": len({r["episode"] for r in rs}), "n_answerable": answerable, "correct": n["correct"],
            "wrong": n["wrong"], "abstain": n["abstain"], "n_unanswerable": unanswerable, "false_answer": n["false_answer"],
            "correct_abstain": n["correct_abstain"], "forced_violation": sum(r["forced_violation"] for r in rs),
            "forced_violation_correct_abstain": sum(r["forced_violation"] and not r["answerable"] for r in rs),
            "forced_fallback": sum(r["forced_fallback"] for r in rs), "forced_fallback_correct": fallback["correct"],
            "forced_fallback_wrong": fallback["wrong"], "forced_fallback_false_answer": fallback["false_answer"],
            "hr": _rate(n["wrong"], answerable), "fabrication": _rate(n["wrong"] + n["false_answer"], len(rs)), "abstain_rate": _rate(n["abstain"], answerable),
            "n_lookalike": len(look), "lookalike_confident_error_rate": _rate(sum(map(confident_error, look)), len(look)),
            "n_stale": len(stale), "stale_confident_error_rate": _rate(sum(map(confident_error, stale)), len(stale)),
            "cost_per_question_median_usd": median(costs), "cost_per_question_mean_usd": sum(costs, Fraction(0)) / len(costs),
            "gate_reasons": _counts(Counter(r["gate_reason"] or "not_run" for r in rs)) if l_rows else "",
            "gate_refusals": _counts(Counter(g for r in rs for g in r["gate_reasons"] if g not in ("anchored", "no_target"))) if l_rows else "",
        })  # fmt: skip
    return out


def _counts(c: Counter) -> str:
    return ";".join(f"{k}:{v}" for k, v in sorted(c.items()))


def figure_rows(cells: list[dict]) -> list[dict]:
    return [{**{k: c[k] for k in CELL_KEYS}, "n_answerable": c["n_answerable"], "correct_share": _rate(c["correct"], c["n_answerable"]),
             "wrong_share": _rate(c["wrong"], c["n_answerable"]), "abstain_share": _rate(c["abstain"], c["n_answerable"]),
             "n_unanswerable": c["n_unanswerable"], "false_answer_share": _rate(c["false_answer"], c["n_unanswerable"])} for c in cells]  # fmt: skip


# Tests


def mcnemar_one_sided(b: int, c: int) -> Fraction:
    """Exact one-sided McNemar p for "the first arm errs less": P(X <= b), X ~ Binomial(b + c, 1/2)."""
    n = b + c
    return Fraction(1) if n == 0 else Fraction(sum(comb(n, i) for i in range(b + 1)), 2**n)


def _test(name: str, model: str, window, family: str, condition: str, **kw) -> dict:
    row = {"test": name, "model": model, "window": window, "family": family, "condition": condition, "estimate": None, "ci_low": None, "ci_high": None,
           "p": None, "p_kind": "one-sided", "p_holm": None, "verdict": "", "n_episodes": 0, "n_items": None, "b": None, "c": None, "valid_resamples": 0,
           "note": ""}  # fmt: skip
    row.update(kw)
    return row


def _with(row: dict, result) -> dict:
    return {**row, "estimate": result.estimate, "ci_low": result.low, "ci_high": result.high, "valid_resamples": result.valid}


def slope_test(name: str, rows: list[dict], metric: Callable[[list[dict]], tuple[int, int]], model, window, family, condition, resamples, seed) -> dict:
    """The per-episode metric at each rho, its mean over episodes per rho, and the OLS slope on log10 rho."""
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
    base = _test(name, model, window, family, condition, n_episodes=len(ids), note=f"OLS slope of the per-episode mean on log10 rho over rho {rhos}")
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
    p = Fraction(sum(v <= 0 for v in result.values), len(result.values)) if result.values else None
    return {**_with(base, result), "p": p}


def _pairs(rows: list[dict], first: str, second: str, keep: Callable[[dict], bool]) -> list[tuple[int, bool, bool]]:
    """(episode, confident error under `first`, under `second`) per item (question and rho) both ran."""
    a = {(r["rho"], r["qid"]): r for r in rows if r["arm"] == first and keep(r)}
    b = {(r["rho"], r["qid"]): r for r in rows if r["arm"] == second and keep(r)}
    return [(a[k]["episode"], confident_error(a[k]), confident_error(b[k])) for k in sorted(set(a) & set(b), key=str)]


def mcnemar_test(name: str, pairs: list[tuple[int, bool, bool]], model, window, family, condition, note, resamples, seed) -> dict:
    by_episode: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    for e, x, y in pairs:
        by_episode[e][0] += x and not y
        by_episode[e][1] += y and not x
        by_episode[e][2] += 1
    b, c = sum(v[0] for v in by_episode.values()), sum(v[1] for v in by_episode.values())
    base = _test(name, model, window, family, condition, n_episodes=len(by_episode), n_items=len(pairs), b=b, c=c,
                 note=f"{note}; b = items only L errs on, c = items only the other arm errs on")  # fmt: skip
    if not pairs:
        return {**base, "note": f"not run: no paired items ({note})"}

    def stat(w: Counter) -> Fraction | None:
        sums = [sum(k * by_episode[e][i] for e, k in w.items() if e in by_episode) for i in range(3)]
        return Fraction(sums[0] - sums[1], sums[2]) if sums[2] else None

    result = resampled(Draws(sorted(by_episode), resamples, seed), stat)
    return {**_with(base, result), "p": mcnemar_one_sided(b, c)}


def cost_test(rows: list[dict], model, window, family, resamples, seed) -> dict:
    l_rows, b4_rows = ([r for r in rows if r["arm"] == arm] for arm in ("L", "B4"))
    ids = sorted({r["episode"] for r in l_rows} & {r["episode"] for r in b4_rows})
    base = _test("H4", model, window, family, "allowed", n_episodes=len(ids), note="median per-question cost L / B4, rho pooled; supported if CI upper <= 1.5")
    if not ids:
        return {**base, "note": "not run: no episode in both L and B4"}
    costs = {arm: defaultdict(list) for arm in ("L", "B4")}  # integer pico-USD: the ratio is unit-free and int medians are fast
    for r in l_rows + b4_rows:
        costs[r["arm"]][r["episode"]].append(r["cost_pusd"])

    def stat(w: Counter) -> Fraction | None:
        meds = [median([c for e, k in w.items() for _ in range(k) for c in costs[arm].get(e, [])]) for arm in ("L", "B4")]
        return None if None in meds or meds[1] == 0 else meds[0] / meds[1]

    result = resampled(Draws(ids, resamples, seed), stat)
    p = Fraction(sum(v >= H4_BOUND for v in result.values), len(result.values)) if result.values else None
    return {**_with(base, result), "p": p}


def hypothesis_tests(rows: list[dict], resamples: int = RESAMPLES_V3, seed: int = SEED_V3) -> list[dict]:
    """The PREREG-v3 tests per model and W (module docstring), with Holm and verdicts."""
    rows = [r for r in rows if not r["partial"]]
    out = []
    for model, window in sorted({(r["model"], r["window"]) for r in rows}, key=str):
        family = "primary" if model == HAIKU else "secondary"
        mine = [r for r in rows if (r["model"], r["window"]) == (model, window)]
        cond = lambda c, arm=None: [r for r in mine if r["condition"] == c and (arm is None or r["arm"] == arm)]  # noqa: E731
        hr = lambda rs: (sum(r["outcome"] == "wrong" for r in rs), sum(r["answerable"] for r in rs))  # noqa: E731
        fab = lambda rs: (sum(map(confident_error, rs)), len(rs))  # noqa: E731
        abst = lambda rs: (sum(r["outcome"] == "abstain" for r in rs), sum(r["answerable"] for r in rs))  # noqa: E731
        tests = []
        for arm in ("B3", "B4"):
            tests.append(slope_test(f"H1-{arm}", cond("forced", arm), hr, model, window, family, "forced", resamples, seed))
            tests.append(slope_test(f"H1sec-{arm}", cond("forced", arm), fab, model, window, family, "forced", resamples, seed))
            tests.append(slope_test(f"H1a-{arm}", cond("allowed", arm), abst, model, window, family, "allowed", resamples, seed))
        allowed = cond("allowed")
        for other in ("B4", "B3"):
            for suffix, keep in (("", lambda r: True), ("-lookalike", lambda r: r["lookalike"]), ("-stale", lambda r: r["stale"])):
                tests.append(mcnemar_test(f"H2-vs{other}{suffix}", _pairs(allowed, "L", other, keep), model, window, family, "allowed",
                                          f"confident errors, L vs {other}, rho pooled{' on ' + suffix[1:] + ' items' if suffix else ''}", resamples, seed))  # fmt: skip
        forced = [r for r in with_variants(cond("forced")) if r["rho"] >= 100]
        for name, variant in (("H3", "primary"), ("H3-gated", "gated")):
            chosen = [r for r in forced if r["arm"] == "B4" or (r["arm"] == "L" and r["variant"] == variant)]
            tests.append(mcnemar_test(name, _pairs(chosen, "L", "B4", lambda r: True), model, window, family, "forced",
                                      f"fabrication, L ({variant}) vs B4, rho >= 100", resamples, seed))  # fmt: skip
        tests.append(cost_test(allowed, model, window, family, resamples, seed))
        adjusted = holm({t["test"]: t["p"] for t in tests if t["test"] in HOLM_V3})
        for t in tests:
            if t["test"] in HOLM_V3:
                t["p_holm"] = adjusted[t["test"]]
            t["verdict"] = _verdict(t)
        out += tests
    return out


def _verdict(t: dict) -> str:
    if t["note"].startswith("not run"):
        return "not run"
    if t["test"] == "H4":
        return "not estimable" if t["ci_high"] is None else "supported" if t["ci_high"] <= H4_BOUND else "not supported"
    p = t["p_holm"] if t["test"] in HOLM_V3 else t["p"]
    if p is None:
        return "not estimable"
    if p < ALPHA:
        return "supported" if t["test"] in HOLM_V3 else "significant at 0.05 (uncorrected)"
    return "not detected at this n"


# Writing

CELL_COLUMNS = [*CELL_KEYS, "n_episodes", "n_answerable", "correct", "wrong", "abstain", "n_unanswerable", "false_answer", "correct_abstain", "forced_violation",
                "forced_violation_correct_abstain", "forced_fallback", "forced_fallback_correct", "forced_fallback_wrong", "forced_fallback_false_answer", "hr", "fabrication", "abstain_rate",
                "n_lookalike", "lookalike_confident_error_rate", "n_stale", "stale_confident_error_rate", "cost_per_question_median_usd",
                "cost_per_question_mean_usd", "gate_reasons", "gate_refusals"]  # fmt: skip
TEST_COLUMNS = ["test", "model", "window", "family", "condition", "estimate", "ci_low", "ci_high", "p", "p_kind", "p_holm", "verdict", "n_episodes", "n_items",
                "b", "c", "valid_resamples", "note"]  # fmt: skip
FIGURE_COLUMNS = [*CELL_KEYS, "n_answerable", "correct_share", "wrong_share", "abstain_share", "n_unanswerable", "false_answer_share"]


COST_COLUMNS = ("cost_per_question_median_usd", "cost_per_question_mean_usd")  # nine places: a question costs about $0.0001-0.01


def _cell(v, places: int = 6):
    return decimal_str(v, places) if isinstance(v, Fraction) else v


FIGURE_SHARES = (("correct_share", "correct, answerable"), ("wrong_share", "wrong, answerable"), ("abstain_share", "abstain, answerable"),
                 ("false_answer_share", "false answer, unanswerable"))  # fmt: skip


def main_figure(out: Path) -> Path | None:
    """main_figure.png drawn from out/figure_data.csv alone, if matplotlib is importable (it is no
    dependency and is never installed for this); None, and no file, if it is not or the CSV has
    no rows. A row of panels per model, W and condition, a panel per share, a line per arm (and
    L's gated variant) over rho on a log axis."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    with open(out / "figure_data.csv", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f))
    panels = sorted({(r["model"], r["window"], r["condition"]) for r in data})
    if not panels:
        return None
    fig, axes = plt.subplots(len(panels), len(FIGURE_SHARES), figsize=(4 * len(FIGURE_SHARES), 3 * len(panels)), squeeze=False)
    for i, panel in enumerate(panels):
        mine = [r for r in data if (r["model"], r["window"], r["condition"]) == panel]
        for j, (column, label) in enumerate(FIGURE_SHARES):
            ax = axes[i][j]
            for arm, variant in sorted({(r["arm"], r["variant"]) for r in mine}, key=lambda av: (av[0], av[1] != "primary")):
                points = sorted((int(r["rho"]), float(r[column])) for r in mine if (r["arm"], r["variant"]) == (arm, variant) and r[column])
                if points:
                    ax.plot(*zip(*points), marker="o", label=arm if variant == "primary" else f"{arm} ({variant})")
            ax.set(xscale="log", ylim=(0, 1), xlabel="rho")
            ax.set_title(f"{panel[0]}, W {panel[1]}, {panel[2]}: {label}", fontsize=8)
            if j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "main_figure.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def analyze_v3(rows: list[dict], out: Path, *, resamples: int = RESAMPLES_V3, seed: int = SEED_V3, meta: dict | None = None) -> dict:
    """Write cells.csv, tests.csv, tests.json and figure_data.csv from analysis rows (`prepare`),
    then main_figure.png from figure_data.csv if matplotlib is importable; infrastructure errors
    are left out. Returns the cells, the tests and the figure's path (None if skipped)."""
    rows = [r for r in rows if not r["error"]]
    cells = cell_rows(with_variants(rows))
    tests = hypothesis_tests(rows, resamples, seed)
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / "cells.csv", CELL_COLUMNS, [[_cell(c[k], 9 if k in COST_COLUMNS else 6) for k in CELL_COLUMNS] for c in cells])
    _csv(out / "tests.csv", TEST_COLUMNS, [[_cell(t[k]) for k in TEST_COLUMNS] for t in tests])
    shown = [{k: _cell(t[k]) for k in TEST_COLUMNS} for t in tests]
    (out / "tests.json").write_text(json.dumps({"meta": {"resamples": resamples, "seed": seed, "alpha": "0.05", **(meta or {})}, "tests": shown},
                                               indent=1, sort_keys=True) + "\n", encoding="utf-8")  # fmt: skip
    _csv(out / "figure_data.csv", FIGURE_COLUMNS, [[_cell(f[k]) for k in FIGURE_COLUMNS] for f in figure_rows(cells)])
    return {"cells": cells, "tests": tests, "figure": main_figure(out)}


__all__ = ["analyze_v3", "cell_rows", "hypothesis_tests", "load", "main_figure", "mcnemar_one_sided", "prepare", "with_variants"]
