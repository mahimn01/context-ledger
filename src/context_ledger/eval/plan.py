"""The cost planner (E): the pilot's cost math with the parameters in force, and a verdict.

The window is W = 6,000 (PREREG section 2). Figures come from the latest LIVE smoke run at that
window if one exists (or one named explicitly), else from the brief's prior (Mahimn's agent's
defaults), which was stated at W = 16,000:

    Haiku 5.5 agent call (B4 or L): 8.5k cache read, 1.5k cache write (5m), 0.1k uncached input,
        200 output (thinking disabled); 6 calls per question, 8 questions per episode
    B4 summary call: 13.6k uncached input, 1.5k output; 4 per episode
    B3 call: 16k uncached input, 200 output; one per question
    Sonnet 5.5: the same shapes with 400 output (between_tools progress updates)

At another W the prior's window-derived input scales with it (`prior_params`): the RAG call
fills W, the compaction call reads 0.85 W, and the agent call's prompt is 10k/16k = 5/8 of W in
the same read/write/uncached proportions; outputs and call counts do not scale. A measured
smoke run replaces the prior with its mean call shapes and counts and, for each arm it ran, its
measured cost per episode (the arm's settled cost per question times 8, compaction calls
included). A smoke run at a different W is never used: costs do not transfer across windows.

Prices come from the verified table; all arithmetic is exact (pico-USD integers and Fractions).
The re-planned grid fits the $50 hard cap (raised from $10 by Mahimn on 2026-10-31) with sub-caps
per item that sum to $39.75, leaving a reserve of $10.25 (20.5%). It runs the full design: the Haiku
grid and the Sonnet subset (B4 and L at rho 10 and 1000, 12 episodes each). `run --grid` refuses if
the item's estimate x 1.2 exceeds its sub-cap or the SpendGuard's remaining budget. If measured
costs push an item over its sub-cap / 1.2, the planner shrinks that item's episode count.

`plan --measured RUN` also prints `episode_costs` (BUILD-V3): the measured cost of every
episode-arm unit of the run directory and of each of its rho-* parts, with its condition, and the
mean per (model, condition, rho, arm). Read-only.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from ..pricing import PICO_PER_USD, PricingTable, usd

HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
EPISODES, RHOS = 12, (10, 100, 1000)
WINDOW = 6000
PRIOR_WINDOW = 16_000
SONNET_RHOS = (10, 1000)
SAFETY = Fraction(6, 5)
UNCERTAINTY = 2
SUB_CAPS_USD = {"smoke": "0.75", "calibration": None, "haiku_grid": "6.00", "sonnet_subset": "33.00", "reserve": "10.25"}


class PlanError(Exception):
    """The planner was given figures it cannot use (a smoke run at another window)."""


@dataclass(frozen=True)
class CallShape:
    """Mean tokens of one call: cache read, cache write (5m), uncached input, output."""

    cache_read: int
    cache_write: int
    uncached: int
    output: int


@dataclass(frozen=True)
class Params:
    agent: CallShape
    summary: CallShape
    rag: CallShape
    calls_per_question: Fraction = Fraction(6)
    questions_per_episode: int = 8
    summaries_per_episode: Fraction = Fraction(4)
    source: str = "prior (BUILD-M3 section E)"
    per_episode: tuple[tuple[str, Fraction], ...] = ()  # measured (arm, pico-USD per episode)
    window: int = PRIOR_WINDOW


PRIOR = {
    HAIKU: Params(CallShape(8500, 1500, 100, 200), CallShape(0, 0, 13600, 1500), CallShape(0, 0, 16000, 200)),
    SONNET: Params(CallShape(8500, 1500, 100, 400), CallShape(0, 0, 13600, 1500), CallShape(0, 0, 16000, 400)),
}


def _scaled(shape: CallShape, factor: Fraction) -> CallShape:
    return CallShape(*(round(Fraction(n) * factor) for n in (shape.cache_read, shape.cache_write, shape.uncached)), shape.output)


def prior_params(window: int = WINDOW) -> dict[str, Params]:
    """The brief's prior with its window-derived input tokens scaled from W = 16,000 to `window`
    (identical to PRIOR at 16,000)."""
    factor = Fraction(window, PRIOR_WINDOW)
    source = PRIOR[HAIKU].source + ("" if window == PRIOR_WINDOW else f", inputs scaled to W = {window:,}")
    return {m: replace(p, agent=_scaled(p.agent, factor), summary=_scaled(p.summary, factor), rag=_scaled(p.rag, factor), source=source, window=window)
            for m, p in PRIOR.items()}  # fmt: skip


def call_pusd(pricing: PricingTable, model: str, shape: CallShape) -> int:
    """Exact pico-USD of one call of this shape (5-minute cache writes)."""
    usage = {"input_tokens": shape.uncached, "cache_read_input_tokens": shape.cache_read,
             "cache_creation_input_tokens": shape.cache_write, "output_tokens": shape.output}  # fmt: skip
    return pricing.cost_pusd(model, usage, "5m")


def episode_pusd(pricing: PricingTable, model: str, p: Params) -> dict[str, Fraction]:
    """Expected pico-USD per episode of each arm: measured where the params carry it, else the
    call-shape math."""
    calls = p.calls_per_question * p.questions_per_episode
    agent = calls * call_pusd(pricing, model, p.agent)
    return {
        "L": Fraction(agent),
        "B4": agent + p.summaries_per_episode * call_pusd(pricing, model, p.summary),
        "B3": Fraction(p.questions_per_episode * call_pusd(pricing, model, p.rag)),
        **dict(p.per_episode),
    }


def plan(pricing: PricingTable, params: dict[str, Params] | None = None, remaining_pusd: int | None = None, window: int = WINDOW) -> dict:
    """Every number of section E, exact, plus the re-planned grid and the verdict, at `window`
    (the prior scaled to it unless `params` are given)."""
    params = params or prior_params(window)
    haiku, sonnet = episode_pusd(pricing, HAIKU, params[HAIKU]), episode_pusd(pricing, SONNET, params[SONNET])
    per_q = {arm: v / params[HAIKU].questions_per_episode for arm, v in haiku.items()}
    grid = EPISODES * len(RHOS) * (haiku["L"] + haiku["B4"] + haiku["B3"])
    smoke = 3 * (per_q["L"] + per_q["B4"] + per_q["B3"]) + (per_q["L"] + per_q["B4"])
    sonnet_subset = len(SONNET_RHOS) * EPISODES * (sonnet["L"] + sonnet["B4"])  # B4 and L at rho 10 and 1000
    caps = {k: (None if v is None else int(Fraction(v) * PICO_PER_USD)) for k, v in SUB_CAPS_USD.items()}
    per_episode = {"haiku_grid": len(RHOS) * (haiku["L"] + haiku["B4"] + haiku["B3"]), "sonnet_subset": len(SONNET_RHOS) * (sonnet["L"] + sonnet["B4"])}
    episodes = {k: min(EPISODES, int(caps[k] / (SAFETY * v))) for k, v in per_episode.items()}
    items = {"smoke": smoke, "calibration": Fraction(0), "haiku_grid": episodes["haiku_grid"] * per_episode["haiku_grid"],
             "sonnet_subset": episodes["sonnet_subset"] * per_episode["sonnet_subset"]}  # fmt: skip
    total = sum(items.values())
    remaining = remaining_pusd if remaining_pusd is not None else 50 * PICO_PER_USD
    verdict_ok = total * SAFETY <= remaining and all(caps[k] is None or items[k] * SAFETY <= caps[k] for k in items) and min(episodes.values()) > 0
    shrunk = sorted(k for k, n in episodes.items() if n < EPISODES)
    return {
        "source": params[HAIKU].source,
        "window": params[HAIKU].window,
        "haiku_call": {"agent": call_pusd(pricing, HAIKU, params[HAIKU].agent), "summary": call_pusd(pricing, HAIKU, params[HAIKU].summary),
                       "rag": call_pusd(pricing, HAIKU, params[HAIKU].rag)},
        "sonnet_call": {"agent": call_pusd(pricing, SONNET, params[SONNET].agent), "summary": call_pusd(pricing, SONNET, params[SONNET].summary)},
        "haiku_episode": haiku,
        "sonnet_episode": sonnet,
        "haiku_grid": grid,
        "haiku_grid_2x": UNCERTAINTY * grid,
        "sonnet_design_subset": sonnet_subset,
        "episodes": episodes,
        "per_episode": per_episode,
        "items": items,
        "sub_caps": caps,
        "total": total,
        "remaining": remaining,
        "fits": verdict_ok,
        "shrunk": shrunk,
    }  # fmt: skip


def measured_params(run_dir: Path, base: dict[str, Params] | None = None) -> dict[str, Params]:
    """Replace the prior (scaled to the run's window) with a smoke run's measured mean call
    shapes, call counts and per-episode cost of each arm it ran."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    model = run["model"]
    base = base or prior_params(run.get("window", PRIOR_WINDOW))

    def mean_shape(rows: list[dict], fallback: CallShape) -> CallShape:
        if not rows:
            return fallback
        n = len(rows)
        mean = lambda key: round(Fraction(sum(r["usage"][key] for r in rows), n))  # noqa: E731
        return CallShape(mean("cache_read_input_tokens"), mean("cache_creation_input_tokens"), mean("input_tokens"), mean("output_tokens"))

    agent = [c for c in calls if c["arm"] in ("B4", "L") and c["kind"] == "answer"]
    questions = {(c["arm"], c["qid"]) for c in agent}
    summaries = [c for c in calls if c["kind"] == "summary"]
    b4_questions = {c["qid"] for c in calls if c["arm"] == "B4"}
    prior = base[model]
    measured = replace(
        prior,
        agent=mean_shape(agent, prior.agent),
        summary=mean_shape(summaries, prior.summary),
        rag=mean_shape([c for c in calls if c["arm"] == "B3"], prior.rag),
        calls_per_question=Fraction(len(agent), len(questions)) if questions else prior.calls_per_question,
        summaries_per_episode=(Fraction(len(summaries), len(b4_questions)) * prior.questions_per_episode) if b4_questions else prior.summaries_per_episode,
        source=f"measured from smoke run {run['run_id']}",
        per_episode=per_episode_costs(calls, prior.questions_per_episode),
        window=run.get("window", prior.window),
    )  # fmt: skip
    return {**base, model: measured}


def per_episode_costs(calls: list[dict], questions_per_episode: int) -> tuple[tuple[str, Fraction], ...]:
    """(arm, measured pico-USD per episode) for every arm with calls in `calls`: the arm's settled
    cost (cost_pusd of every call, compaction summaries included) per question asked, times
    the questions of an episode."""
    out = []
    for arm in sorted({c["arm"] for c in calls}):
        rows = [c for c in calls if c["arm"] == arm and "cost_pusd" in c]
        questions = {c["qid"] for c in rows}
        if questions:
            out.append((arm, Fraction(sum(c["cost_pusd"] for c in rows), len(questions)) * questions_per_episode))
    return tuple(out)


def episode_costs(run_dir: Path) -> list[dict]:
    """One row per (run part, arm, episode) of `run_dir` and its rho-* parts: model, condition,
    rho, the questions asked and the settled cost of every call of that episode-arm unit
    (compaction summaries included), in pico-USD."""
    rows = []
    for path in sorted({Path(run_dir) / "run.json", *Path(run_dir).glob("*/run.json")}):
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        read = lambda name: [json.loads(x) for x in (path.parent / name).read_text(encoding="utf-8").splitlines()] if (path.parent / name).exists() else []  # noqa: E731
        asked = Counter((r["arm"], r["episode"]) for r in read("results.jsonl"))
        cost: Counter = Counter()
        for c in read("calls.jsonl"):
            cost[(c["arm"], c["episode"])] += c["cost_pusd"]
        for arm, episode in sorted(set(asked) | set(cost)):
            rows.append({"model": run["model"], "condition": run.get("condition", "allowed"), "rho": run["rho"], "window": run.get("window"), "arm": arm,
                         "episode": episode, "questions": asked[(arm, episode)], "cost_pusd": cost[(arm, episode)]})  # fmt: skip
    return rows


def render_episode_costs(rows: list[dict]) -> str:
    """The measured per-episode-arm costs and, per (model, condition, rho, arm), their mean."""
    lines = ["Measured cost per episode-arm unit (settled cost of every call, compaction summaries included):",
             f"  {'model':<18} {'condition':<9} {'rho':>5} {'arm':<4} {'episode':>7} {'questions':>9} {'cost':>12} {'per question':>13}"]  # fmt: skip
    for r in rows:
        per_q = _d(Fraction(r["cost_pusd"], r["questions"])) if r["questions"] else "n/a"
        lines.append(f"  {r['model']:<18} {r['condition']:<9} {r['rho']:>5} {r['arm']:<4} {r['episode']:>7} {r['questions']:>9} {_d(r['cost_pusd']):>12} {per_q:>13}")
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["model"], r["condition"], r["rho"], r["arm"]), []).append(r["cost_pusd"])
    lines.append("Mean per episode:")
    lines += [f"  {m:<18} {c:<9} {rho:>5} {arm:<4} {len(v):>3} episodes {_d(Fraction(sum(v), len(v))):>12}" for (m, c, rho, arm), v in sorted(groups.items())]
    return "\n".join(lines)


def latest_live_smoke(runs_dir: Path, window: int | None = None) -> Path | None:
    """The most recent run directory whose run.json is a live smoke run (at `window`, if given)."""
    found = []
    for path in runs_dir.glob("*/run.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if window is not None and run.get("window") != window:
            continue
        if run.get("live") and run.get("smoke") and (path.parent / "calls.jsonl").exists():
            found.append((path.stat().st_mtime, path.parent))
    return max(found)[1] if found else None


def _d(value: Fraction | int) -> str:
    return usd(value, 6)


def render_plan(p: dict) -> str:
    """The section E math with the parameters in force, and the verdict."""
    h, s = p["haiku_episode"], p["sonnet_episode"]
    lines = [
        f"Window W = {p['window']:,} tokens. Parameters: {p['source']}; prices from the verified 2026-10-31 table (USD per million tokens).",
        "",
        "Haiku 5.5",
        f"  agent call (B4 or L): {usd(p['haiku_call']['agent'], 7)}   summary call: {_d(p['haiku_call']['summary'])}   RAG call: {_d(p['haiku_call']['rag'])}",
        f"  per episode: L {_d(h['L'])}, B4 {_d(h['B4'])} (with summaries), B3 {_d(h['B3'])}",
        f"  grid, 12 episodes x 3 rho x (L + B4 + B3): {_d(p['haiku_grid'])} expected; with the 2x model-uncertainty factor {_d(p['haiku_grid_2x'])}",
        "Sonnet 5.5",
        f"  agent call: {_d(p['sonnet_call']['agent'])}   per episode: L {_d(s['L'])}, B4 {_d(s['B4'])}",
        f"  the design's subset (B4 and L at rho 10 and 1000, 12 episodes each): {_d(p['sonnet_design_subset'])}",
        "",
        "Re-planned grid, $50 hard cap, sub-caps sum to $39.75 (reserve $10.25, 20.5%)",
        f"  {'item':<34} {'expected':>12} {'sub-cap (--budget)':>20}",
    ]
    names = {"smoke": "smoke (Haiku, rho 10, 3+1 questions)", "calibration": "calibration (count_tokens, free)",
             "haiku_grid": "Haiku 5.5 full grid", "sonnet_subset": "Sonnet 5.5 subset (B4+L, rho 10+1000)"}  # fmt: skip
    for key, label in names.items():
        cap = p["sub_caps"][key]
        eps = f"  ({p['episodes'][key]} episodes)" if key in p["episodes"] else ""
        lines.append(f"  {label:<34} {_d(p['items'][key]):>12} {('n/a' if cap is None else _d(cap)):>20}{eps}")
    lines.append(f"  {'unallocated reserve':<34} {'n/a':>12} {_d(p['sub_caps']['reserve']):>20}")
    lines.append(f"  {'total':<34} {_d(p['total']):>12} {_d(sum(v for v in p['sub_caps'].values() if v)):>20}")
    for key in p["shrunk"]:
        lines.append(f"  note: {names[key]} shrunk to {p['episodes'][key]} episodes so its estimate x 1.2 fits its sub-cap")
    lines += [
        "",
        f"SpendGuard remaining: {_d(p['remaining'])}; total estimate x 1.2 = {_d(p['total'] * SAFETY)}",
        "Verdict: FITS (the total estimate x 1.2 is within the remaining budget and every item within its sub-cap)"
        if p["fits"] else "Verdict: DOES NOT FIT: shrink the grid, or wait for Mahimn to raise the budget",
    ]
    return "\n".join(lines)
