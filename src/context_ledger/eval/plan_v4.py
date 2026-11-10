"""`eval plan --v4`: the cost of the v4 pilot and grids (BUILD-V4 item 11, BUILD-V4B decisions 1 and
7), read-only.

Episode bound. Every v4 run gives every arm 260 model calls per episode (26 questions x 10;
`run --max-calls-episode 260`), the default here (`--max-calls-episode` prices another).

Cost per episode-arm unit, per (model, condition, arm), from the best source there is:
1. pilot: measured in the runs given with `--measured` (the v4 pilot), at the same episode bound,
   averaged over its episodes and rho (at a fixed W the per-episode cost does not grow with rho:
   every arm's prompt is held to W); only units that ran to the end count (BUILD-V4FIX: not the
   unit a run stopped in, nor one with a partial or API-error row, as the grid's Sonnet L units
   the 400s cut short would underprice a rerun);
2. pilot x ratio: for a model the pilot did not run (Sonnet), Haiku's pilot cost times that arm's
   Sonnet/Haiku cost ratio in the v3 grid (B3's for B5, L's for L4 and L4-nogate; about 20x);
3. v3 estimate: the v3 grid's measured cost of B3, B4 and L at W = 3,000 (runs/v3-grid-*, or
   `--measured-v3`; else the built-in means below), each agent arm scaled to the episode bound by
   its measured calls per asked question x 26 / 90 (at most bound / 90), with L4 = L4-nogate = L x
   L4_FACTOR and B5 = B3 x B5_FACTOR (stated assumptions, replaced by the pilot).

Items: the pilot (questions v3 episodes 11-12, Haiku, rho 10, 1000 and 10000, both conditions,
six arms), the grids PREREG-v4 names, and a custom grid (`--model`, `--episodes`, `--rhos`,
`--arms`, `--conditions`). Each grid is run in runs of one condition, one rho and half of its
episodes (all its arms), as scripts/v4/grid.sh runs it; per run the plan gives the expected cost and a suggested
--budget (CAP_FACTOR x expected, rounded up to $0.05). Everything is exact (pico-USD,
Fractions); plan.py's 1.2 safety factor applies to the totals.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from ..pricing import PICO_PER_USD, usd
from .analyze import HAIKU, SONNET
from .plan import SAFETY, PlanError

L4_FACTOR = Fraction(1)
B5_FACTOR = Fraction(2)
QUESTIONS = 26
V3_MAX_CALLS = 90
DEFAULT_MAX_CALLS = 260
CAP_FACTOR = Fraction(2)
ARMS = ("B3", "B4", "B5", "L", "L4", "L4-nogate")
AGENTS = ("B4", "L")  # the measured arms whose cost grows with the episode bound (L4 and L4-nogate follow L)
BASE = {"B3": "B3", "B4": "B4", "L": "L", "B5": "B3", "L4": "L", "L4-nogate": "L"}
FACTOR = {"B5": B5_FACTOR, "L4": L4_FACTOR, "L4-nogate": L4_FACTOR}
CONDITIONS = ("allowed", "forced")
RHOS = (10, 100, 1000, 10000)
PILOT = {"model": HAIKU, "episodes": 2, "rhos": (10, 1000, 10000), "conditions": CONDITIONS, "arms": ARMS}
GRIDS = {
    "grid Haiku: 6 arms, 10 episodes": {"model": HAIKU, "episodes": 10, "rhos": RHOS, "conditions": CONDITIONS, "arms": ARMS},
    "grid Sonnet: 5 arms (no L4-nogate), 8 episodes": {"model": SONNET, "episodes": 8, "rhos": RHOS, "conditions": CONDITIONS, "arms": ARMS[:5]},
    "grid Sonnet: L4-nogate only, 8 episodes (if budget allows)": {"model": SONNET, "episodes": 8, "rhos": RHOS, "conditions": CONDITIONS, "arms": ("L4-nogate",)},
}
# The prior, if no v3 run is given: the v3 grid's means at W = 3,000 (2026-11-02), pico-USD per episode-arm unit and calls per asked question.
PRIOR_PUSD = {
    (HAIKU, "allowed", "B3"): 7_000_000_000, (HAIKU, "allowed", "B4"): 25_200_000_000, (HAIKU, "allowed", "L"): 19_300_000_000,
    (HAIKU, "forced", "B3"): 7_800_000_000, (HAIKU, "forced", "B4"): 25_500_000_000, (HAIKU, "forced", "L"): 20_000_000_000,
    (SONNET, "allowed", "B3"): 142_200_000_000, (SONNET, "allowed", "B4"): 533_000_000_000, (SONNET, "allowed", "L"): 370_100_000_000,
    (SONNET, "forced", "B3"): 159_400_000_000, (SONNET, "forced", "B4"): 540_800_000_000, (SONNET, "forced", "L"): 373_600_000_000,
}  # fmt: skip
PRIOR_CALLS = {
    (HAIKU, "allowed", "B4"): Fraction(441, 100), (HAIKU, "allowed", "L"): Fraction(414, 100), (HAIKU, "forced", "B4"): Fraction(432, 100),
    (HAIKU, "forced", "L"): Fraction(560, 100), (SONNET, "allowed", "B4"): Fraction(453, 100), (SONNET, "allowed", "L"): Fraction(426, 100),
    (SONNET, "forced", "B4"): Fraction(450, 100), (SONNET, "forced", "L"): Fraction(554, 100),
}  # fmt: skip


def complete_units(run: dict, results: list[dict]) -> set[tuple[str, int]]:
    """The (arm, episode) units of a run part that ran to the end: every unit with a results row
    but the one the run stopped in and any with a partial or an API-error row."""
    stopped = run.get("stopped") or {}
    cut = {(r["arm"], r["episode"]) for r in results if r.get("partial") or r.get("error")} | {(stopped.get("arm"), stopped.get("episode"))}
    return {(r["arm"], r["episode"]) for r in results} - cut


def measured(run_dirs: list[Path]) -> tuple[dict, dict, dict]:
    """(mean pico-USD per episode-arm unit by (model, condition, arm), mean calls per asked
    question by the same, the episode bound by the same) over the complete units of the run
    directories (rho parts included)."""
    by = defaultdict(list)
    calls: dict = defaultdict(lambda: [0, 0])
    bound: dict = {}
    for d in run_dirs:
        for path in sorted({Path(d) / "run.json", *Path(d).glob("*/run.json")}):
            if not path.exists():
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            results = [json.loads(x) for x in _lines(path.parent / "results.jsonl")]
            done = complete_units(run, results)
            cost: Counter = Counter()
            for c in map(json.loads, _lines(path.parent / "calls.jsonl")):
                cost[(c["arm"], c["episode"])] += c["cost_pusd"]
            model, condition = run["model"], run.get("condition", "allowed")
            for arm, episode in done:
                by[(model, condition, arm)].append(cost[(arm, episode)])
            for r in results:
                key = (model, condition, r["arm"])
                bound[key] = run.get("max_calls_episode", V3_MAX_CALLS)
                if r.get("calls") and (r["arm"], r["episode"]) in done:
                    calls[key][0] += r["calls"]
                    calls[key][1] += 1
    return ({k: Fraction(sum(v), len(v)) for k, v in by.items()}, {k: Fraction(c, n) for k, (c, n) in calls.items() if n}, bound)


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def bound_factor(calls_per_question: Fraction | None, max_calls: int) -> Fraction:
    """How a v3 agent arm's episode cost scales from v3's 90-call bound to `max_calls`."""
    cap = Fraction(max_calls, V3_MAX_CALLS)
    return max(Fraction(1), cap) if calls_per_question is None else max(min(Fraction(1), cap), min(cap, calls_per_question * QUESTIONS / V3_MAX_CALLS))


def unit_costs(v3: dict, v3_calls: dict, pilot: dict, pilot_bound: dict, max_calls: int) -> dict:
    """{(model, condition, arm): (pico-USD per episode-arm unit, source)} for both models, both
    conditions and every v4 arm (module docstring)."""
    out = {}
    for model in (HAIKU, SONNET):
        for condition in CONDITIONS:
            for arm in ARMS:
                key, base = (model, condition, arm), (model, condition, BASE[arm])
                haiku_key, haiku_base = (HAIKU, condition, arm), (HAIKU, condition, BASE[arm])
                if key in pilot and pilot_bound.get(key) == max_calls:
                    out[key] = (pilot[key], "pilot")
                elif model != HAIKU and haiku_key in pilot and pilot_bound.get(haiku_key) == max_calls and base in v3 and haiku_base in v3:
                    out[key] = (pilot[haiku_key] * v3[base] / v3[haiku_base], f"pilot x v3 {model.removeprefix('claude-').split('-')[0]}/haiku ratio {float(v3[base] / v3[haiku_base]):.1f}")
                elif base in v3:
                    scale = bound_factor(v3_calls.get(base), max_calls) if BASE[arm] in AGENTS else Fraction(1)
                    out[key] = (v3[base] * FACTOR.get(arm, Fraction(1)) * scale, "v3" if arm == BASE[arm] else f"v3 {BASE[arm]} x {FACTOR[arm]}")
    return out


def design_cost(costs: dict, design: dict) -> Fraction:
    return sum((costs[(design["model"], c, a)][0] * design["episodes"] * len(design["rhos"]) for c in design["conditions"] for a in design["arms"]), Fraction(0))


def run_plan(costs: dict, design: dict) -> dict:
    """A grid's runs (one condition, one rho, half its episodes, all its arms): the expected cost of
    the larger half's run per condition and the suggested --budget."""
    half, step = -(-design["episodes"] // 2), PICO_PER_USD // 20
    out = {}
    for c in design["conditions"]:
        expected = sum((costs[(design["model"], c, a)][0] * half for a in design["arms"]), Fraction(0))
        out[c] = {"episodes": half, "expected": expected, "cap": -(-int(expected * CAP_FACTOR) // step) * step}
    return out


def plan_v4(v3_dirs: list[Path] | None = None, pilot_dirs: list[Path] | None = None, max_calls: int = DEFAULT_MAX_CALLS, custom: dict | None = None,
            remaining_pusd: int | None = None) -> dict:  # fmt: skip
    if v3_dirs:
        v3, v3_calls, _ = measured(v3_dirs)
        v3 = {k: v for k, v in v3.items() if k[2] in ("B3", "B4", "L")}
        source = f"measured from {len(v3_dirs)} v3 run directories"
    else:
        v3, v3_calls, source = dict(PRIOR_PUSD), dict(PRIOR_CALLS), "the v3 grid's means at W = 3,000 (built in; no v3 run given)"
    pilot, _, pilot_bound = measured(pilot_dirs) if pilot_dirs else ({}, {}, {})
    costs = unit_costs(v3, v3_calls, pilot, pilot_bound, max_calls)
    designs = {"pilot: questions v3 episodes 11-12, Haiku, rho 10/1000/10000, 6 arms": PILOT, **GRIDS, **({"custom grid": custom} if custom else {})}
    unpriced = sorted({(d["model"], c, a) for d in designs.values() for c in d["conditions"] for a in d["arms"]} - set(costs))
    if unpriced:
        raise PlanError(f"no cost source for {unpriced}")
    items = {name: design_cost(costs, d) for name, d in designs.items()}
    runs = {name: run_plan(costs, d) for name, d in designs.items() if d is not PILOT}
    return {"source": source, "pilot_runs": [str(p) for p in pilot_dirs or []], "max_calls": max_calls, "costs": costs, "items": items, "runs": runs,
            "ignored_pilot": sorted(k for k, b in pilot_bound.items() if b != max_calls), "remaining_pusd": remaining_pusd}  # fmt: skip


def render_plan_v4(p: dict) -> str:
    lines = [f"v4 plan at W = 3,000 and {p['max_calls']} calls per episode (every arm). v3 costs: {p['source']}"
             + (f"; pilot: {', '.join(p['pilot_runs'])}" if p["pilot_runs"] else "; no pilot run given (--measured)."),
             "Cost per episode-arm unit (26 questions):"]  # fmt: skip
    for (model, condition, arm), (v, src) in sorted(p["costs"].items()):
        lines.append(f"  {model:<18} {condition:<8} {arm:<9} {usd(v, 6):>11}  ({src})")
    if p["ignored_pilot"]:
        lines.append(f"Pilot units at another episode bound, not used: {p['ignored_pilot']}")
    lines.append(f"  {'item':<62} {'expected':>11} {'x 1.2':>11}")
    for name, v in p["items"].items():
        lines.append(f"  {name:<62} {usd(v, 6):>11} {usd(v * SAFETY, 6):>11}")
    lines.append("Grid runs (one condition, one rho, half the episodes, all the grid's arms): expected cost and suggested --budget:")
    for name, by_condition in p["runs"].items():
        lines += [f"  {name:<62} {c:<8} {r['episodes']} episodes {usd(r['expected'], 4):>10}  --budget {usd(r['cap'], 2)}" for c, r in by_condition.items()]
    if p["remaining_pusd"] is not None:
        lines.append(f"The --key's ledger has {usd(p['remaining_pusd'], 2)} usable left (scripts/v4 halt before a run that would take the combined spend past $180).")
    return "\n".join(lines)


__all__ = ["ARMS", "B5_FACTOR", "DEFAULT_MAX_CALLS", "GRIDS", "L4_FACTOR", "PILOT", "bound_factor", "measured", "plan_v4", "render_plan_v4", "run_plan", "unit_costs"]
