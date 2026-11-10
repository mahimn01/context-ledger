"""`eval analyze`: the pre-registered analysis (PREREG.md sections 4-6), stdlib only.

Inputs are scored result rows (score.py, the one arm-independent grader), call rows and fold
rows of one or more runs. A cell is (model, arm, rho); a cluster is an episode, the same eight
questions in every arm, rho and model, so resampling episodes pairs every comparison.

Cells (cells.csv). Per cell, each ratio metric is sum(numerator) / sum(denominator) over its
episodes, with a 95% cluster-bootstrap percentile interval (stats.bootstrap: 10,000 resamples,
fixed seed, nearest rank):
    HR                   wrong / answerable                         (primary)
    false_answer_rate    false_answer / unanswerable
    total_hallucination  (wrong + false_answer) / all
    abstention           abstain / answerable
    accuracy             correct / answerable
    selective_accuracy   correct / (correct + wrong)
    fold_probe_accuracy  correct / answerable fold probes
    leakage              answers equal to the unperturbed value / answerable default+constant
    cache_hit_share      cache read / (input + cache read + cache write) tokens
and, over COMPLETE episodes only (a per-episode figure needs the whole episode): the median
cost per episode (USD settled), calls, folds (L) and compactions (B4) per episode, and the
predicted vs actual fold savings (L/folds.jsonl) summed.

Tests (tests.csv, tests.json). The slope is the OLS slope of the cell rate on log10 rho (a
linear probability model; exact, defined with zero-event cells). Every bootstrap resample
draws episodes once and applies the draw to every cell involved, so arms are paired. The
one-sided p-value is the fraction of valid resamples on the wrong side (ties count against the
hypothesis); resamples where a statistic is undefined are left out and counted.
    H1-B3, H1-B4   HR slope > 0 (Haiku), over each arm's episodes
    H2             slope(L) - slope(B4) < 0 (Haiku), episodes complete in both arms at every rho
    H2-McNemar     exact two-sided McNemar on paired answerable questions, wrong vs not wrong,
                   L vs B4 at rho = 1000 (family "supporting", outside Holm; estimate (b - c) / n with b = wrong only
                   under L; its interval is the paired cluster bootstrap)
    H3             HR(L) - HR(B4) < 0, Haiku, rho >= 100 pooled, paired episodes
    H3-NI          non-inferiority: the 95% CI lower bound of accuracy(L) - accuracy(B4) (same
                   pooling) must exceed -3 percentage points; H3 is supported only if both hold
    H4             median episode cost L / B4, Haiku pooled over rho, paired episodes; judged by
                   its CI (upper bound <= 1.5); its p is the fraction of resamples >= 1.5
Holm's step-down adjustment runs over the five primary Haiku tests (H1-B3, H1-B4, H2, H3, H4).
Sonnet rows (H2 at rho 10 and 1000, its McNemar at 1000, H4) are a secondary replication,
uncorrected. A primary test with Holm-adjusted p < 0.05 is "supported"; otherwise "not detected
at this n" (PREREG section 5's power caveat), never "no effect".
Secondary logistic slopes (per arm and model): logistic regression of wrong on log10 rho over
answerable questions by IRLS, with a cluster-robust (episode) sandwich standard error, a Wald
95% interval and a one-sided p for a positive slope; "not_estimable" when there are zero events
(or only events), on complete or quasi-complete separation, if IRLS does not converge, or if the
sandwich degenerates (residuals constant within every episode).

Exclusions (exclusions.csv; PREREG section 6). A question whose call failed for an
infrastructure reason (an API error, a timeout or a harness exception, recorded as the result's
error) is excluded from every metric and listed. Episodes stopped by the budget (rows flagged
partial) are kept in the cells, as far as they went, left out of paired tests and listed. When
the same unit appears in more than one run (an infrastructure re-run), the run given last on
the command line is used and the other is listed as superseded.

Figure (figure_data.csv): model, arm, rho, HR and its interval, for HR against log10 rho.

BUILD-V2 adds two files outside the pre-registered plan, each labelled in a header comment line:
by_template.csv (exploratory: outcomes per model, arm, rho and template) and pressure.csv (the
manipulation check: per model, arm and rho, the p50 and p95 of the actual prompt minus the arm's
tool definitions over W on answer calls, the share of answer calls made with the window at least
half full, and folds and compactions per episode). H1-H4 are unchanged.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from ..ids import frac_str
from ..pricing import PICO_PER_USD
from ..tokens import decimal_str
from .stats import RESAMPLES, SEED, mcnemar_exact, nearest_rank, resample_indices

HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
ALPHA = Fraction(1, 20)
NI_MARGIN = Fraction(-3, 100)
H4_BOUND = Fraction(3, 2)
PRIMARY = ("H1-B3", "H1-B4", "H2", "H3", "H4")

RATIOS: dict[str, tuple[Callable[[dict], bool], Callable[[dict], bool]]] = {
    "HR": (lambda r: r["outcome"] == "wrong", lambda r: r["answerable"]),
    "false_answer_rate": (lambda r: r["outcome"] == "false_answer", lambda r: not r["answerable"]),
    "total_hallucination": (lambda r: r["outcome"] in ("wrong", "false_answer"), lambda r: True),
    "abstention": (lambda r: r["outcome"] == "abstain", lambda r: r["answerable"]),
    "accuracy": (lambda r: r["outcome"] == "correct", lambda r: r["answerable"]),
    "selective_accuracy": (lambda r: r["outcome"] == "correct", lambda r: r["outcome"] in ("correct", "wrong")),
    "fold_probe_accuracy": (lambda r: r["outcome"] == "correct", lambda r: r["answerable"] and r["fold_probe"]),
    "leakage": (lambda r: r["leaked"], lambda r: r["answerable"] and r["has_upstream"]),
}
PER_EPISODE = ("cost_usd", "calls", "folds", "compactions")


def log10_exact(rho: int) -> int:
    """log10 of a power of ten, exactly (the design's ratios are 10, 100 and 1000)."""
    k, n = 0, rho
    while n > 1 and n % 10 == 0:
        n, k = n // 10, k + 1
    if n != 1:
        raise ValueError(f"rho {rho} is not a power of ten")
    return k


@dataclass
class Episode:
    """One (model, arm, rho, episode) unit: per-metric (numerator, denominator) counts and the
    per-episode figures."""

    counts: dict[str, list[int]] = field(default_factory=lambda: {m: [0, 0] for m in RATIOS})
    cache: list[int] = field(default_factory=lambda: [0, 0])  # cache read, prompt tokens
    cost_pusd: int = 0
    calls: int = 0
    folds: int = 0
    compactions: int = 0
    complete: bool = True
    wrong: dict[str, bool] = field(default_factory=dict)  # qid -> wrong (answerable questions)
    correct: dict[str, bool] = field(default_factory=dict)


@dataclass
class Data:
    cells: dict[tuple[str, str, int], dict[int, Episode]]
    exclusions: list[dict]
    folds: dict[tuple[str, str, int], list[int]]  # predicted, actual delta sums (L)


def build(scores: Iterable[dict], calls: Iterable[dict], folds: Iterable[dict] = ()) -> Data:
    """Group scored rows (each with "partial" if the episode was budget-stopped) and call rows
    into cells and episodes, applying the exclusions."""
    cells: dict = defaultdict(dict)
    exclusions = []
    for r in scores:
        ep = cells[(r["model"], r["arm"], r["rho"])].setdefault(r["episode"], Episode())
        if r.get("partial"):
            ep.complete = False
        if r["error"]:
            exclusions.append({"model": r["model"], "arm": r["arm"], "rho": r["rho"], "episode": r["episode"], "qid": r["qid"],
                               "reason": "infrastructure_error", "detail": r.get("error_type") or "api_error"})  # fmt: skip
            continue
        for name, (num, den) in RATIOS.items():
            if den(r):
                ep.counts[name][1] += 1
                ep.counts[name][0] += bool(num(r))
        if r["answerable"]:
            ep.wrong[r["qid"]] = r["outcome"] == "wrong"
            ep.correct[r["qid"]] = r["outcome"] == "correct"
        ep.folds += r.get("folds", 0)
        ep.compactions += r.get("compactions", 0)
    for key, eps in cells.items():
        for e, ep in eps.items():
            if not ep.complete:
                exclusions.append({"model": key[0], "arm": key[1], "rho": key[2], "episode": e, "qid": "",
                                   "reason": "incomplete_episode", "detail": "budget-stopped: in the cells as far as it went, not in paired tests"})  # fmt: skip
    for c in calls:
        ep = cells.get((c["model"], c["arm"], c["rho"]), {}).get(c["episode"])
        if ep is None:
            continue
        u = c["usage"]
        ep.cost_pusd += c["cost_pusd"]
        ep.calls += 1
        ep.cache[0] += u["cache_read_input_tokens"]
        ep.cache[1] += u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"]
    fold_sums: dict = defaultdict(lambda: [0, 0])
    for f in folds:
        if f.get("actual_delta") is not None:
            fold_sums[(f["model"], "L", f["rho"])][0] += f["predicted_delta"]
            fold_sums[(f["model"], "L", f["rho"])][1] += f["actual_delta"]
    return Data(dict(cells), exclusions, dict(fold_sums))


# The cluster bootstrap


class Draws:
    """`resamples` draws of episode ids with replacement (seeded), as multiplicities, shared by
    every statistic computed from them: that is what pairs arms, rhos and models."""

    def __init__(self, ids: Sequence[int], resamples: int = RESAMPLES, seed: int = SEED) -> None:
        self.ids = sorted(ids)
        self.weights = [Counter(self.ids[i] for i in draw) for draw in resample_indices(len(self.ids), resamples, seed)] if self.ids else []
        self.point = Counter(self.ids)


def ratio_at(eps: dict[int, Episode], get: Callable[[Episode], Sequence[int]], weights: Counter) -> Fraction | None:
    """sum(weight * numerator) / sum(weight * denominator), or None if the denominator is 0."""
    num = den = 0
    for e, w in weights.items():
        ep = eps.get(e)
        if ep is not None:
            n, d = get(ep)
            num += w * n
            den += w * d
    return Fraction(num, den) if den else None


def median(values: Sequence[Fraction]) -> Fraction | None:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return None
    return Fraction(ordered[n // 2]) if n % 2 else (Fraction(ordered[n // 2 - 1]) + ordered[n // 2]) / 2


@dataclass(frozen=True)
class Result:
    estimate: Fraction | None
    low: Fraction | None
    high: Fraction | None
    valid: int
    values: tuple[Fraction, ...] = ()


def resampled(draws: Draws, stat: Callable[[Counter], Fraction | None]) -> Result:
    """Point estimate on the observed episodes and the 95% percentile interval."""
    point = stat(draws.point)
    values = tuple(v for v in (stat(w) for w in draws.weights) if v is not None)
    if not values:
        return Result(point, None, None, 0)
    return Result(point, nearest_rank(values, Fraction(5, 2)), nearest_rank(values, Fraction(195, 2)), len(values), values)


def ols_slope(points: Sequence[tuple[int, Fraction | None]]) -> Fraction | None:
    """OLS slope of y on x, exactly; None if any y is undefined or fewer than two distinct x."""
    if any(y is None for _, y in points) or len({x for x, _ in points}) < 2:
        return None
    n = len(points)
    mx = Fraction(sum(x for x, _ in points), n)
    my = sum(y for _, y in points) / n
    return sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)


def p_wrong_side(values: Sequence[Fraction], wrong: Callable[[Fraction], bool]) -> Fraction | None:
    """The fraction of valid resamples on the wrong side of the hypothesis."""
    return Fraction(sum(wrong(v) for v in values), len(values)) if values else None


def holm(pvalues: dict[str, Fraction | float | None]) -> dict[str, Fraction | float | None]:
    """Holm's step-down adjusted p-values (monotone, capped at 1); None stays None."""
    present = sorted(((p, k) for k, p in pvalues.items() if p is not None), key=lambda t: t[0])
    m, running, out = len(present), 0, {k: None for k in pvalues}
    for j, (p, k) in enumerate(present):
        running = max(running, min(1, (m - j) * p))
        out[k] = running
    return out


# Logistic slope (secondary)


def _phi_upper(z: float) -> float:
    """P(Z > z) for a standard normal Z."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def logistic_slope(points: Sequence[tuple[int, bool, int]], max_iter: int = 100) -> dict:
    """Logistic regression of y on x with an intercept by IRLS over (x, y, cluster) points.
    Returns {"status": "estimated", "slope", "low", "high", "p_positive"} with a cluster-robust
    sandwich SE and a Wald interval, or {"status": "not_estimable", "reason"} on zero events, only
    events, (quasi-)complete separation or no convergence."""
    ys = [y for _, y, _ in points]
    if not ys or not any(ys):
        return {"status": "not_estimable", "reason": "zero events"}
    if all(ys):
        return {"status": "not_estimable", "reason": "only events"}
    x1 = [x for x, y, _ in points if y]
    x0 = [x for x, y, _ in points if not y]
    if max(x0) <= min(x1) or max(x1) <= min(x0):
        return {"status": "not_estimable", "reason": "separation"}
    a, b = math.log(len(x1) / len(x0)), 0.0
    for _ in range(max_iter):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y, _ in points:
            p = 1 / (1 + math.exp(-(a + b * x)))
            w = p * (1 - p)
            g0, g1 = g0 + (y - p), g1 + (y - p) * x
            h00, h01, h11 = h00 + w, h01 + w * x, h11 + w * x * x
        det = h00 * h11 - h01 * h01
        if det <= 0:
            return {"status": "not_estimable", "reason": "singular information"}
        da, db = (h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-12:
            break
    else:
        return {"status": "not_estimable", "reason": "no convergence"}
    inv = ((h11 / det, -h01 / det), (-h01 / det, h00 / det))
    scores: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for x, y, c in points:
        p = 1 / (1 + math.exp(-(a + b * x)))
        scores[c][0] += y - p
        scores[c][1] += (y - p) * x
    meat = [[sum(u[i] * u[j] for u in scores.values()) for j in (0, 1)] for i in (0, 1)]
    left = [[sum(inv[i][k] * meat[k][j] for k in (0, 1)) for j in (0, 1)] for i in (0, 1)]
    var_b = sum(left[1][k] * inv[k][1] for k in (0, 1))
    se = math.sqrt(var_b) if var_b > 0 else 0.0
    if se < 1e-9:  # residuals constant within every episode: the sandwich degenerates
        return {"status": "not_estimable", "reason": "degenerate cluster-robust standard error"}
    return {"status": "estimated", "slope": b, "low": b - 1.959963984540054 * se, "high": b + 1.959963984540054 * se, "p_positive": _phi_upper(b / se)}


# The analysis


def _cell_rate(eps: dict[int, Episode], metric: str) -> Callable[[Counter], Fraction | None]:
    return lambda w: ratio_at(eps, lambda ep: ep.counts[metric], w)


def _per_episode(ep: Episode, what: str) -> Fraction:
    return {"cost_usd": Fraction(ep.cost_pusd, PICO_PER_USD), "calls": Fraction(ep.calls), "folds": Fraction(ep.folds), "compactions": Fraction(ep.compactions)}[what]


def _per_episode_stat(eps: dict[int, Episode], what: str, summary: Callable) -> Callable[[Counter], Fraction | None]:
    def stat(w: Counter) -> Fraction | None:
        values = [_per_episode(eps[e], what) for e, k in w.items() for _ in range(k) if e in eps and eps[e].complete]
        return summary(values) if values else None

    return stat


def _mean(values: Sequence[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


def cell_table(data: Data, resamples: int = RESAMPLES, seed: int = SEED) -> list[dict]:
    """One row per cell: every metric with its interval."""
    rows = []
    for (model, arm, rho), eps in sorted(data.cells.items()):
        draws = Draws(list(eps), resamples, seed)
        row: dict = {"model": model, "arm": arm, "rho": rho, "n_episodes": len(eps), "n_complete_episodes": sum(ep.complete for ep in eps.values()),
                     "n_questions": sum(ep.counts["total_hallucination"][1] for ep in eps.values())}  # fmt: skip
        for metric in RATIOS:
            row[metric] = resampled(draws, _cell_rate(eps, metric))
        row["cache_hit_share"] = resampled(draws, lambda w: ratio_at(eps, lambda ep: ep.cache, w))
        row["cost_per_episode_median"] = resampled(draws, _per_episode_stat(eps, "cost_usd", median))
        for what in ("calls", "folds", "compactions"):
            row[f"{what}_per_episode"] = resampled(draws, _per_episode_stat(eps, what, _mean))
        sums = data.folds.get((model, arm, rho))
        row["fold_savings_predicted"], row["fold_savings_actual"] = (sums[0], sums[1]) if sums else (None, None)
        rows.append(row)
    return rows


def _paired(data: Data, model: str, arms: Sequence[str], rhos: Sequence[int]) -> list[int]:
    """Episodes complete in every (arm, rho) cell given."""
    sets = [{e for e, ep in data.cells.get((model, arm, rho), {}).items() if ep.complete} for arm in arms for rho in rhos]
    return sorted(set.intersection(*sets)) if sets and all(sets) else []


def _rhos(data: Data, model: str, arms: Sequence[str]) -> list[int]:
    return sorted({rho for (m, a, rho) in data.cells if m == model and a in arms})


def _slope_stat(data: Data, model: str, arm: str, rhos: Sequence[int]) -> Callable[[Counter], Fraction | None]:
    cells = [(log10_exact(rho), data.cells.get((model, arm, rho), {})) for rho in rhos]
    return lambda w: ols_slope([(x, ratio_at(eps, lambda ep: ep.counts["HR"], w)) for x, eps in cells])


def _pooled(data: Data, model: str, arm: str, rhos: Sequence[int], metric: str) -> Callable[[Counter], Fraction | None]:
    def stat(w: Counter) -> Fraction | None:
        num = den = 0
        for rho in rhos:
            eps = data.cells.get((model, arm, rho), {})
            for e, k in w.items():
                if e in eps:
                    num += k * eps[e].counts[metric][0]
                    den += k * eps[e].counts[metric][1]
        return Fraction(num, den) if den else None

    return stat


def _minus(f: Callable[[Counter], Fraction | None], g: Callable[[Counter], Fraction | None]) -> Callable[[Counter], Fraction | None]:
    """The statistic f - g on the same draw (None if either is undefined)."""

    def stat(w: Counter) -> Fraction | None:
        a, b = f(w), g(w)
        return None if a is None or b is None else a - b

    return stat


def _test(name: str, model: str, family: str, result: Result | None, p: Fraction | float | None, n_episodes: int, sided: str, note: str = "") -> dict:
    return {"test": name, "model": model, "family": family, "estimate": None if result is None else result.estimate,
            "ci_low": None if result is None else result.low, "ci_high": None if result is None else result.high,
            "p": p, "p_kind": sided, "p_holm": None, "verdict": "", "n_episodes": n_episodes,
            "valid_resamples": 0 if result is None else result.valid, "note": note}  # fmt: skip


def hypothesis_tests(data: Data, resamples: int = RESAMPLES, seed: int = SEED) -> list[dict]:
    """The PREREG section 5 tests, Holm-adjusted over the primary Haiku family, with verdicts."""
    rows: list[dict] = []
    # H1: HR slope > 0 for B3 and B4 (Haiku), over each arm's episodes.
    for arm in ("B3", "B4"):
        rhos = _rhos(data, HAIKU, [arm])
        ids = sorted({e for rho in rhos for e in data.cells.get((HAIKU, arm, rho), {})})
        if len(rhos) < 2 or not ids:
            rows.append(_test(f"H1-{arm}", HAIKU, "primary", None, None, 0, "one-sided", "not run"))
            continue
        result = resampled(Draws(ids, resamples, seed), _slope_stat(data, HAIKU, arm, rhos))
        rows.append(_test(f"H1-{arm}", HAIKU, "primary", result, p_wrong_side(result.values, lambda v: v <= 0), len(ids), "one-sided",
                          f"OLS slope of HR on log10 rho over rho {rhos}"))  # fmt: skip
    # H2 and its replication: slope(L) - slope(B4) < 0, paired episodes; McNemar at rho 1000.
    for model, family in ((HAIKU, "primary"), (SONNET, "secondary")):
        rhos = sorted(set(_rhos(data, model, ["L"])) & set(_rhos(data, model, ["B4"])))
        ids = _paired(data, model, ["L", "B4"], rhos)
        if len(rhos) < 2 or not ids:
            rows.append(_test("H2", model, family, None, None, 0, "one-sided", "not run"))
        else:
            result = resampled(Draws(ids, resamples, seed), _minus(_slope_stat(data, model, "L", rhos), _slope_stat(data, model, "B4", rhos)))
            rows.append(_test("H2", model, family, result, p_wrong_side(result.values, lambda v: v >= 0), len(ids), "one-sided",
                              f"slope(L) - slope(B4) over rho {rhos}, paired episodes"))  # fmt: skip
        rows.append(_mcnemar(data, model, "supporting" if family == "primary" else family, resamples, seed))
    # H3: HR(L) < HR(B4), Haiku, rho >= 100 pooled; non-inferiority on accuracy.
    rhos = [r for r in sorted(set(_rhos(data, HAIKU, ["L"])) & set(_rhos(data, HAIKU, ["B4"]))) if r >= 100]
    ids = _paired(data, HAIKU, ["L", "B4"], rhos)
    if not rhos or not ids:
        rows += [_test("H3", HAIKU, "primary", None, None, 0, "one-sided", "not run"), _test("H3-NI", HAIKU, "primary", None, None, 0, "none", "not run")]
    else:
        draws = Draws(ids, resamples, seed)
        gap = lambda metric: _minus(_pooled(data, HAIKU, "L", rhos, metric), _pooled(data, HAIKU, "B4", rhos, metric))  # noqa: E731
        hr = resampled(draws, gap("HR"))
        rows.append(_test("H3", HAIKU, "primary", hr, p_wrong_side(hr.values, lambda v: v >= 0), len(ids), "one-sided", f"HR(L) - HR(B4), rho {rhos} pooled"))
        ni = resampled(draws, gap("accuracy"))
        rows.append(_test("H3-NI", HAIKU, "primary", ni, None, len(ids), "none", "accuracy(L) - accuracy(B4); its CI lower bound must exceed -0.03"))
    # H4: median episode cost L / B4 pooled over rho, judged by its CI.
    for model, family in ((HAIKU, "primary"), (SONNET, "secondary")):
        rhos = sorted(set(_rhos(data, model, ["L"])) & set(_rhos(data, model, ["B4"])))
        ids = _paired(data, model, ["L", "B4"], rhos)
        if not ids:
            rows.append(_test("H4", model, family, None, None, 0, "one-sided", "not run"))
            continue

        def ratio(w: Counter, model=model, rhos=rhos) -> Fraction | None:
            meds = []
            for arm in ("L", "B4"):
                costs = [_per_episode(data.cells[(model, arm, rho)][e], "cost_usd") for e, k in w.items() for _ in range(k) for rho in rhos]
                meds.append(median(costs))
            return None if None in meds or meds[1] == 0 else meds[0] / meds[1]

        result = resampled(Draws(ids, resamples, seed), ratio)
        rows.append(_test("H4", model, family, result, p_wrong_side(result.values, lambda v: v >= H4_BOUND), len(ids), "one-sided",
                          f"median episode cost L / B4 over rho {rhos}; judged by CI upper <= 1.5"))  # fmt: skip
    # Holm over the primary Haiku family, then verdicts.
    primary = {r["test"]: r["p"] for r in rows if r["model"] == HAIKU and r["test"] in PRIMARY}
    adjusted = holm(primary)
    for r in rows:
        if r["model"] == HAIKU and r["test"] in PRIMARY:
            r["p_holm"] = adjusted[r["test"]]
        r["verdict"] = _verdict(r, rows)
    rows += logistic_rows(data)
    return rows


def _mcnemar(data: Data, model: str, family: str, resamples: int, seed: int) -> dict:
    ids = _paired(data, model, ["L", "B4"], [1000])
    if not ids:
        return _test("H2-McNemar", model, family, None, None, 0, "two-sided", "not run at rho 1000")
    lc, bc = data.cells[(model, "L", 1000)], data.cells[(model, "B4", 1000)]

    def counts(w: Counter) -> tuple[int, int, int]:
        b = c = n = 0
        for e, k in w.items():
            shared = set(lc[e].wrong) & set(bc[e].wrong)
            n += k * len(shared)
            b += k * sum(lc[e].wrong[q] and not bc[e].wrong[q] for q in shared)
            c += k * sum(bc[e].wrong[q] and not lc[e].wrong[q] for q in shared)
        return b, c, n

    def stat(w: Counter) -> Fraction | None:
        b, c, n = counts(w)
        return Fraction(b - c, n) if n else None

    b, c, n = counts(Counter(ids))
    result = resampled(Draws(ids, resamples, seed), stat)
    return _test("H2-McNemar", model, family, result, mcnemar_exact(b, c), len(ids), "two-sided",
                 f"rho 1000: wrong only under L b={b}, only under B4 c={c}, paired questions n={n}")  # fmt: skip


def _verdict(r: dict, rows: list[dict]) -> str:
    if r["estimate"] is None and r["test"] != "H3-NI":
        return "not run" if "not run" in r["note"] else "not estimable"
    if r["test"] == "H4":
        return "supported" if r["ci_high"] is not None and r["ci_high"] <= H4_BOUND else "not supported"
    if r["test"] == "H3-NI":
        if r["ci_low"] is None:
            return "not run" if "not run" in r["note"] else "not estimable"
        return "met" if r["ci_low"] > NI_MARGIN else "not met"
    p = r["p_holm"] if r["p_holm"] is not None else r["p"]
    if p is None:
        return "not estimable"
    significant = p < ALPHA
    if r["test"] == "H3":
        ni = next((x for x in rows if x["test"] == "H3-NI" and x["model"] == r["model"]), None)
        if significant and ni is not None and (ni["ci_low"] is None or ni["ci_low"] <= NI_MARGIN):
            return "not supported: non-inferiority not met"
    if r["p_kind"] == "two-sided":
        return "significant at 0.05" if significant else "not detected at this n"
    return "supported" if significant else "not detected at this n"


def logistic_rows(data: Data) -> list[dict]:
    """Secondary: the logistic slope of wrong on log10 rho, per model and arm."""
    rows = []
    for model, arm in sorted({(m, a) for m, a, _ in data.cells}):
        points = [(log10_exact(rho), w, e) for (m, a, rho), eps in sorted(data.cells.items()) if (m, a) == (model, arm)
                  for e, ep in eps.items() for w in ep.wrong.values()]  # fmt: skip
        fit = logistic_slope(points) if len({x for x, _, _ in points}) >= 2 else {"status": "not_estimable", "reason": "fewer than two rho"}
        estimated = fit["status"] == "estimated"
        rows.append({"test": f"logistic-slope-{arm}", "model": model, "family": "secondary", "estimate": fit.get("slope"), "ci_low": fit.get("low"),
                     "ci_high": fit.get("high"), "p": fit.get("p_positive"), "p_kind": "one-sided (Wald, cluster-robust)", "p_holm": None,
                     "verdict": "estimated" if estimated else "not_estimable", "n_episodes": len({e for _, _, e in points}), "valid_resamples": 0,
                     "note": "log-odds of wrong per decade of rho" if estimated else fit["reason"]})  # fmt: skip
    return rows


# Writing


def _show(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, Fraction):
        return decimal_str(v, 6)
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


CELL_METRICS = [*RATIOS, "cache_hit_share", "cost_per_episode_median", "calls_per_episode", "folds_per_episode", "compactions_per_episode"]
TEST_KEYS = ["test", "model", "family", "estimate", "ci_low", "ci_high", "p", "p_kind", "p_holm", "verdict", "n_episodes", "valid_resamples", "note"]


def write_outputs(out: Path, data: Data, cells: list[dict], tests: list[dict], meta: dict) -> None:
    """cells.csv, tests.csv, tests.json, exclusions.csv and figure_data.csv in `out`."""
    out.mkdir(parents=True, exist_ok=True)
    header = ["model", "arm", "rho", "n_episodes", "n_complete_episodes", "n_questions"]
    for m in CELL_METRICS:
        header += [m, f"{m}_lo", f"{m}_hi"]
    header += ["fold_savings_predicted", "fold_savings_actual"]
    body = []
    for c in cells:
        row = [c["model"], c["arm"], c["rho"], c["n_episodes"], c["n_complete_episodes"], c["n_questions"]]
        for m in CELL_METRICS:
            row += [_show(c[m].estimate), _show(c[m].low), _show(c[m].high)]
        body.append(row + [c["fold_savings_predicted"], c["fold_savings_actual"]])
    _csv(out / "cells.csv", header, body)
    _csv(out / "tests.csv", TEST_KEYS, [[_show(t[k]) if k not in ("n_episodes", "valid_resamples") else t[k] for k in TEST_KEYS] for t in tests])
    shown = [{k: (_show(t[k]) if k not in ("n_episodes", "valid_resamples") else t[k]) for k in TEST_KEYS} for t in tests]
    (out / "tests.json").write_text(json.dumps({"meta": meta, "tests": shown}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    keys = ["model", "arm", "rho", "episode", "qid", "reason", "detail"]
    _csv(out / "exclusions.csv", keys, [[x[k] for k in keys] for x in data.exclusions])
    _csv(out / "figure_data.csv", ["model", "arm", "rho", "HR", "lo", "hi"],
         [[c["model"], c["arm"], c["rho"], _show(c["HR"].estimate), _show(c["HR"].low), _show(c["HR"].high)] for c in cells])  # fmt: skip


def _commented_csv(path: Path, comment: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {comment}\n")
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


OUTCOMES = ("correct", "wrong", "abstain", "correct_abstain", "false_answer")


def by_template_rows(scores: Iterable[dict]) -> list[list]:
    """Per (model, arm, rho, template): questions, each outcome, and wrong / answerable."""
    groups: dict = defaultdict(Counter)
    for r in scores:
        if not r.get("error"):
            groups[(r["model"], r["arm"], r["rho"], r.get("template") or "")][r["outcome"]] += 1
    rows = []
    for (model, arm, rho, template), c in sorted(groups.items()):
        answerable = c["correct"] + c["wrong"] + c["abstain"]
        rows.append([model, arm, rho, template, sum(c.values()), *(c[o] for o in OUTCOMES), _show(Fraction(c["wrong"], answerable)) if answerable else None])
    return rows


def pressure_rows(scores: Iterable[dict], calls: Iterable[dict]) -> list[list]:
    """Per (model, arm, rho): the manipulation check (see the module docstring). Each call row
    carries its run's window and the arm's tool-definition tokens (load_runs adds them)."""
    episodes: dict = defaultdict(set)
    folds: Counter = Counter()
    compactions: Counter = Counter()
    for r in scores:
        key = (r["model"], r["arm"], r["rho"])
        episodes[key].add(r["episode"])
        folds[key] += r.get("folds", 0)
        compactions[key] += r.get("compactions", 0)
    ratios: dict = defaultdict(list)
    for c in calls:
        if c.get("kind", "answer") == "answer" and c.get("window"):
            u = c["usage"]
            actual = u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"] - (c.get("tool_definitions") or 0)
            ratios[(c["model"], c["arm"], c["rho"])].append(Fraction(actual, c["window"]))
    rows = []
    for key in sorted(set(episodes) | set(ratios)):
        values, n = ratios.get(key, []), len(episodes.get(key, ())) or 1
        rank = lambda q: nearest_rank(values, q) if values else None  # noqa: E731
        half = Fraction(sum(v >= Fraction(1, 2) for v in values), len(values)) if values else None
        rows.append([*key, len(values), _show(rank(50)), _show(rank(95)), _show(half), _show(Fraction(folds[key], n)), _show(Fraction(compactions[key], n))])
    return rows


def write_v2_outputs(out: Path, scores: list[dict], calls: list[dict]) -> None:
    """by_template.csv (exploratory) and pressure.csv (manipulation check)."""
    _commented_csv(out / "by_template.csv", "exploratory (not pre-registered): outcomes by template", ["model", "arm", "rho", "template", "n", *OUTCOMES, "HR"],
                   by_template_rows(scores))  # fmt: skip
    _commented_csv(out / "pressure.csv", "manipulation check (not a hypothesis test): window pressure per cell",
                   ["model", "arm", "rho", "answer_calls", "prompt_over_w_p50", "prompt_over_w_p95", "share_calls_half_full", "folds_per_episode",
                    "compactions_per_episode"], pressure_rows(scores, calls))  # fmt: skip


def analyze(scores: list[dict], calls: list[dict], folds: list[dict], out: Path, *, resamples: int = RESAMPLES, seed: int = SEED,
            superseded: Sequence[dict] = (), meta: dict | None = None) -> dict:  # fmt: skip
    """Build cells and tests from scored rows and write every output. Returns the tests."""
    data = build(scores, calls, folds)
    data.exclusions += list(superseded)
    cells = cell_table(data, resamples, seed)
    tests = hypothesis_tests(data, resamples, seed)
    write_outputs(out, data, cells, tests, {"resamples": resamples, "seed": seed, "alpha": frac_str(ALPHA), **(meta or {})})
    write_v2_outputs(out, scores, calls)
    return {"cells": cells, "tests": tests, "exclusions": data.exclusions}


def load_runs(run_dirs: Sequence[Path], score_fn: Callable[[dict, list[dict]], list[dict]]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """(scored rows, call rows, fold rows, superseded exclusions) of every run under `run_dirs`
    (grid subdirectories included). `score_fn(run, results)` scores one run's results. A unit
    (model, arm, rho, episode) found in several runs is taken from the one given last."""
    units: dict[tuple, tuple[dict, list[dict], list[dict], list[dict]]] = {}
    superseded = []
    for run_dir in run_dirs:
        for path in sorted({Path(run_dir) / "run.json", *Path(run_dir).glob("*/run.json")}):
            if not path.exists():
                continue
            d = path.parent
            run = json.loads(path.read_text(encoding="utf-8"))
            read = lambda name: [json.loads(x) for x in (d / name).read_text(encoding="utf-8").splitlines()] if (d / name).exists() else []  # noqa: E731
            results = read("results.jsonl")
            scored = score_fn(run, results)
            for s, r in zip(scored, results):
                s.update(partial=bool(r.get("partial")), error_type=(r.get("error") or {}).get("error_type"), folds=r.get("folds", 0), compactions=r.get("compactions", 0))
            tools = (run.get("tool_definitions") or {}).get("tokens") or {}
            calls = [{**c, "rho": run["rho"], "window": run.get("window"), "tool_definitions": tools.get(c["arm"])} for c in read("calls.jsonl")]
            folds = [{**f, "rho": run["rho"], "model": run["model"]} for f in read("L/folds.jsonl")]
            for unit in sorted({(s["model"], s["arm"], s["rho"], s["episode"]) for s in scored}):
                if unit in units:
                    superseded.append({"model": unit[0], "arm": unit[1], "rho": unit[2], "episode": unit[3], "qid": "", "reason": "superseded_by_rerun",
                                       "detail": f"replaced by {run['run_id']}"})  # fmt: skip
                m, a, rho, e = unit
                units[unit] = (
                    run,
                    [s for s in scored if (s["model"], s["arm"], s["rho"], s["episode"]) == unit],
                    [c for c in calls if (c["model"], c["arm"], c["episode"]) == (m, a, e)],
                    [f for f in folds if a == "L" and f["episode"] == e],
                )
    scores = [s for _, rows, _, _ in units.values() for s in rows]
    calls = [c for _, _, rows, _ in units.values() for c in rows]
    folds = [f for _, _, _, rows in units.values() for f in rows]
    return scores, calls, folds, superseded
