"""The pilot report: runs/<run_id>/report.md and report.json (D, F.2).

Metrics per cell (arm x rho x model), as the design defines them:
    HR                  wrong / answerable
    false-answer rate   false answers / unanswerable
    abstention          abstentions / answerable
    selective accuracy  correct / (correct + wrong)
    leakage             answers equal to the unperturbed value / answerable default+constant
    fold-probe accuracy correct fold probes / fold probes
    fold loss           fold-probe accuracy - accuracy on the answerable questions of slots 1-4
    calls, tokens and cost per episode; cache-hit ratio cache_read / (input + cache_read + cache_write)
Every ratio carries a cluster-bootstrap 95% interval over episodes; paired arms at each rho get
McNemar's exact test on answerable correctness. The report also shows the SpendGuard summary,
the leakscan result, per-call predicted vs reported prompt and cache tokens, and each fold's
predicted vs actual delta and rewrite cost. report.json holds ints and strings only (ratios as
"num/den" and basis points).
"""

from __future__ import annotations

import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from ..ids import frac_str
from ..live.spend import read_spend, totals
from ..pricing import usd
from .stats import Estimate, bootstrap, difference, mcnemar_exact, paired_counts, ratio, resample_indices

RATIOS = {
    "HR": lambda r: (r["outcome"] == "wrong", r["answerable"]),
    "false_answer_rate": lambda r: (r["outcome"] == "false_answer", not r["answerable"]),
    "abstention": lambda r: (r["outcome"] == "abstain", r["answerable"]),
    "selective_accuracy": lambda r: (r["outcome"] == "correct", r["outcome"] in ("correct", "wrong")),
    "leakage": lambda r: (r["leaked"], r["answerable"] and r["has_upstream"]),
    "fold_probe_accuracy": lambda r: (r["outcome"] == "correct", r["answerable"] and r["fold_probe"]),
    "q1_4_accuracy": lambda r: (r["outcome"] == "correct", r["answerable"] and r["pos"] < 4),
}


def _bp(value: Fraction | None) -> int | None:
    return None if value is None else int(value * 10000)


def _show(e: Estimate) -> dict:
    return {
        "point": None if e.point is None else frac_str(e.point),
        "point_bp": _bp(e.point),
        "ci95": [None if e.low is None else frac_str(e.low), None if e.high is None else frac_str(e.high)],
        "ci95_bp": [_bp(e.low), _bp(e.high)],
        "valid_resamples": e.valid,
    }


def cell_metrics(rows: list[dict], calls: list[dict], resamples: int) -> dict:
    """Every metric of one cell, with bootstrap intervals over its episodes."""
    episodes = sorted({r["episode"] for r in rows})
    by_ep = {e: [r for r in rows if r["episode"] == e] for e in episodes}
    indices = resample_indices(len(episodes), resamples)
    out = {}
    for name, fn in RATIOS.items():
        clusters = [(sum(fn(r)[0] and fn(r)[1] for r in by_ep[e]), sum(bool(fn(r)[1]) for r in by_ep[e])) for e in episodes]
        out[name] = _show(bootstrap(clusters, ratio, indices))
    probe, early = RATIOS["fold_probe_accuracy"], RATIOS["q1_4_accuracy"]
    pairs = [
        (sum(probe(r)[0] and probe(r)[1] for r in by_ep[e]), sum(bool(probe(r)[1]) for r in by_ep[e]),
         sum(early(r)[0] and early(r)[1] for r in by_ep[e]), sum(bool(early(r)[1]) for r in by_ep[e]))
        for e in episodes
    ]  # fmt: skip
    out["fold_loss"] = _show(bootstrap(pairs, difference, indices))
    n_ep = max(1, len(episodes))
    tokens = sum(c["usage"][k] for c in calls for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"))
    seen = sum(c["usage"][k] for c in calls for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    read = sum(c["usage"]["cache_read_input_tokens"] for c in calls)
    out["counts"] = {
        "questions": len(rows),
        "episodes": len(episodes),
        **{o: sum(r["outcome"] == o for r in rows) for o in ("correct", "wrong", "abstain", "correct_abstain", "false_answer")},
        "hallucination_unsupported": sum(r["hallucination"] == "unsupported" for r in rows),
        "hallucination_misread": sum(r["hallucination"] == "misread" for r in rows),
        "hallucination_caught": sum(r["caught"] for r in rows),
        "bound_hit": sum(r["bound_hit"] for r in rows),
        "api_errors": sum(r["error"] for r in rows),
    }
    cost = sum(c["cost_pusd"] for c in calls)
    out["cost"] = {
        "calls": len(calls),
        "calls_per_episode": frac_str(Fraction(len(calls), n_ep)),
        "tokens_per_episode": frac_str(Fraction(tokens, n_ep)),
        "cost_pusd": cost,
        "cost_usd": usd(cost),
        "cost_per_episode_usd": usd(Fraction(cost, n_ep)),
        "cache_hit": frac_str(Fraction(read, seen) if seen else Fraction(0)),
        "cache_hit_bp": int(Fraction(read, seen) * 10000) if seen else 0,
    }
    out["predicted_vs_reported"] = prediction_totals(calls)
    return out


def reported_prompt(usage: dict) -> int:
    return usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"]


def prediction_totals(calls: list[dict]) -> dict:
    """Predicted vs API-reported tokens summed over a cell's calls. Prompt totals cover only the
    calls whose prompt was counted (both sides over the same calls); cache reads cover every call."""
    counted = [c for c in calls if isinstance(c.get("predicted_prompt"), int)]
    return {
        "calls": len(calls),
        "calls_counted": len(counted),
        "predicted_prompt": sum(c["predicted_prompt"] for c in counted),
        "reported_prompt": sum(reported_prompt(c["usage"]) for c in counted),
        "predicted_cache_read": sum(c["predicted_cache_read"] for c in calls),
        "reported_cache_read": sum(c["usage"]["cache_read_input_tokens"] for c in calls),
    }


def collect(run_dir: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """(run.json of every run under run_dir, scores, calls, folds), including grid subdirectories."""
    runs, scores, calls, folds = [], [], [], []
    for path in sorted({run_dir / "run.json", *run_dir.glob("*/run.json")}):
        if not path.exists():
            continue
        d = path.parent
        run = json.loads(path.read_text())
        runs.append(run)
        for name, sink in (("scores.jsonl", scores), ("calls.jsonl", calls), ("L/folds.jsonl", folds)):
            if (d / name).exists():
                sink += [{**json.loads(line), "rho": run["rho"]} for line in (d / name).read_text().splitlines()]
    return runs, scores, calls, folds


def build_report(run_dir: Path, resamples: int = 10_000) -> dict:
    """Compute the report of a scored run directory and write report.json and report.md."""
    runs, scores, calls, folds = collect(run_dir)
    if not scores:
        raise ValueError(f"no scores in {run_dir}; run `eval score` first")
    cells = defaultdict(list)
    for r in scores:
        cells[(r["arm"], r["rho"], r["model"])].append(r)
    by_cell_calls = defaultdict(list)
    for c in calls:
        by_cell_calls[(c["arm"], c["rho"], c["model"])].append(c)
    table = {f"{a}|{rho}|{m}": cell_metrics(rows, by_cell_calls[(a, rho, m)], resamples) for (a, rho, m), rows in sorted(cells.items())}
    series = defaultdict(dict)
    for (a, rho, m) in sorted(cells):
        series[f"{a}|{m}"][str(rho)] = table[f"{a}|{rho}|{m}"]["HR"]
    paired = []
    for (rho, m) in sorted({(rho, m) for _, rho, m in cells}):
        arms = sorted(a for a, r2, m2 in cells if (r2, m2) == (rho, m))
        right = {a: {r["qid"]: r["outcome"] == "correct" for r in cells[(a, rho, m)] if r["answerable"]} for a in arms}
        for i, a in enumerate(arms):
            for b in arms[i + 1 :]:
                only_a, only_b, n = paired_counts(right[a], right[b])
                paired.append({"rho": rho, "model": m, "a": a, "b": b, "only_a": only_a, "only_b": only_b, "paired": n,
                               "p_value": frac_str(mcnemar_exact(only_a, only_b))})  # fmt: skip
    spend = []
    for run in runs:
        records = read_spend(run["spend_ledger"]) if Path(run["spend_ledger"]).exists() else []
        t = totals(records)
        reserved = sum(r["worst_case_pusd"] for r in records if r["kind"] == "reserve" and r["run_id"] == run["run_id"])
        settled = sum(r["actual_pusd"] for r in records if r["kind"] == "settle" and r["run_id"] == run["run_id"])
        broken = sum(1 for r in records if r["kind"] == "settle" and r["actual_pusd"] > r["worst_case_pusd"])
        spend.append({"run_id": run["run_id"], "reserved_pusd": reserved, "settled_pusd": settled, "actual_above_worst": broken,
                      "spent_total_pusd": t.spent, "remaining_pusd": run["spend"]["global_cap_pusd"] - t.spent - t.outstanding,
                      "leakscan": run.get("leakscan")})  # fmt: skip
    per_call = [
        {"arm": c["arm"], "qid": c["qid"], "kind": c["kind"], "predicted_prompt": c["predicted_prompt"],
         "reported_prompt": reported_prompt(c["usage"]),
         "predicted_cache_read": c["predicted_cache_read"], "reported_cache_read": c["usage"]["cache_read_input_tokens"],
         "reported_cache_write": c["usage"]["cache_creation_input_tokens"]}
        for c in calls
    ]  # fmt: skip
    report = {"runs": [{k: run[k] for k in ("run_id", "live", "arms", "rho", "model", "window", "smoke", "incomplete", "stopped", "corpus_id")} for run in runs],
              "cells": table, "hr_by_rho": dict(series), "paired": paired, "spend": spend, "calls": per_call, "folds": folds}  # fmt: skip
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _ci(m: dict) -> str:
    lo, hi = m["ci95_bp"]
    point = "n/a" if m["point_bp"] is None else f"{m['point_bp'] / 100:.1f}%"
    return point if lo is None else f"{point} [{lo / 100:.1f}, {hi / 100:.1f}]"


def render_markdown(report: dict) -> str:
    lines = ["# Pilot report", ""]
    for run in report["runs"]:
        status = "INCOMPLETE: " + run["stopped"]["reason"] if run["incomplete"] else "complete"
        lines.append(f"- run `{run['run_id']}`: {'LIVE' if run['live'] else 'offline (fake server)'}, arms {','.join(run['arms'])}, "
                     f"rho {run['rho']}, {run['model']}, W={run['window']}{', smoke' if run['smoke'] else ''}; {status}")  # fmt: skip
    lines += ["", "## Metrics per cell (95% cluster-bootstrap intervals over episodes, in percent)", ""]
    names = ["HR", "false_answer_rate", "abstention", "selective_accuracy", "leakage", "fold_probe_accuracy", "fold_loss"]
    lines.append("| cell | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    for cell, m in report["cells"].items():
        lines.append(f"| {cell} | " + " | ".join(_ci(m[n]) for n in names) + " |")
    lines += ["", "## Counts", "", "| cell | correct | wrong | abstain | correct abstain | false answer | unsupported | misread | caught | bound hit |", "|---" * 10 + "|"]
    for cell, m in report["cells"].items():
        c = m["counts"]
        lines.append(f"| {cell} | {c['correct']} | {c['wrong']} | {c['abstain']} | {c['correct_abstain']} | {c['false_answer']} | "
                     f"{c['hallucination_unsupported']} | {c['hallucination_misread']} | {c['hallucination_caught']} | {c['bound_hit']} |")  # fmt: skip
    lines += ["", "## HR by rho", ""]
    for arm, by_rho in report["hr_by_rho"].items():
        lines.append(f"- {arm}: " + ", ".join(f"rho {rho}: {_ci(m)}" for rho, m in sorted(by_rho.items(), key=lambda kv: int(kv[0]))))
    lines += ["", "## Paired comparisons (McNemar exact, answerable correctness)", ""]
    for p in report["paired"]:
        lines.append(f"- rho {p['rho']} {p['model']}: {p['a']} vs {p['b']}: only {p['a']} {p['only_a']}, only {p['b']} {p['only_b']} of {p['paired']}; p = {p['p_value']}")
    lines += ["", "## Cost", "", "| cell | calls | calls/episode | tokens/episode | cost | cost/episode | cache hit |", "|---" * 7 + "|"]
    for cell, m in report["cells"].items():
        c = m["cost"]
        lines.append(f"| {cell} | {c['calls']} | {float(Fraction(c['calls_per_episode'])):.1f} | {float(Fraction(c['tokens_per_episode'])):.0f} | "
                     f"{c['cost_usd']} | {c['cost_per_episode_usd']} | {c['cache_hit_bp'] / 100:.1f}% |")  # fmt: skip
    lines += ["", "## SpendGuard", ""]
    for s in report["spend"]:
        lines.append(f"- {s['run_id']}: reserved {usd(s['reserved_pusd'])}, settled {usd(s['settled_pusd'])}, actual above worst: "
                     f"{s['actual_above_worst']} (must be 0), remaining {usd(s['remaining_pusd'])}; leakscan: {s['leakscan'] or 'not run'}")  # fmt: skip
    lines += ["", "## Predicted vs reported, totals per cell", "",
              "| cell | calls (counted) | prompt predicted | prompt reported | cache read predicted | cache read reported |", "|---" * 6 + "|"]
    for cell, m in report["cells"].items():
        t = m["predicted_vs_reported"]
        lines.append(f"| {cell} | {t['calls']} ({t['calls_counted']}) | {t['predicted_prompt']} | {t['reported_prompt']} | "
                     f"{t['predicted_cache_read']} | {t['reported_cache_read']} |")  # fmt: skip
    lines += ["", "## Predicted vs reported, per call", "", "| arm | qid | kind | prompt predicted | prompt reported | cache read predicted | cache read reported | cache write |", "|---" * 8 + "|"]
    for c in report["calls"]:
        lines.append(f"| {c['arm']} | {c['qid']} | {c['kind']} | {c['predicted_prompt']} | {c['reported_prompt']} | {c['predicted_cache_read']} | {c['reported_cache_read']} | {c['reported_cache_write']} |")
    lines += ["", "## Folds: predicted vs actual (L)", ""]
    if not report["folds"]:
        lines.append("No fold fired.")
    else:
        lines += ["| episode | fold | trigger | delta predicted | delta actual | rewrite predicted | rewrite actual |", "|---" * 7 + "|"]
        for f in report["folds"]:
            pred = "n/a" if f["predicted_rewrite_pusd"] is None else usd(f["predicted_rewrite_pusd"])
            lines.append(f"| {f['episode']} | {f['fold_id']} | {f['trigger']} | {f['predicted_delta']} | {f['actual_delta']} | {pred} | {usd(f['actual_rewrite_pusd'])} |")
    if not all(run["live"] for run in report["runs"]):
        lines += ["", "Offline runs use the fake server and a scripted model: their numbers test the harness, not any model."]
    lines.append("")
    return "\n".join(lines)
