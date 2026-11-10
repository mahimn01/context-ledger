"""Pilot diagnostics (BUILD-V2 item 7): `eval quick` and `eval diagnose-abstain`. Both read gold
(through the one arm-independent grader, score.py); the arms never do. For pilots only: the
pre-registered analysis is `eval analyze`.

quick             per arm x rho: answerable n, correct, wrong, abstain; unanswerable n and false
                  answers; the same by template; folds and compactions per episode; spend; and a
                  pressure line (BUILD-V2b): the p50, p95 and max of the actual prompt minus tool
                  definitions over W, and how many answer calls were above W, always recomputed
                  from calls.jsonl so runs written before those figures existed show them too.
                  BUILD-V3 adds a line per arm: the run's condition, the forced_violation and
                  forced_fallback counts (and how many fallback answers the grader scores
                  correct), and for L the final answer-gate reason of each question, the gate's
                  refusals by reason, and how many questions had no target (no_target; expect 0
                  on v2s and v3 questions). BUILD-V4: L4 gets the same gate line, and L4 and B5 a
                  line of their own: the pre-resolution outcome per question (exact, miss, note,
                  no_note), the repairs by reason, the unverified_guess count (forced: a blocked
                  candidate after repair, scored as an answer) and the calls per tool.
diagnose-abstain  L's abstentions on answerable questions by the harness or model reason x
                  whether an answer was blocked first (one the grader would score correct, only
                  others, none) x whether a fold happened in that turn; plus the verdict reason
                  of every claim in the run's L ledgers. This is the table in BUILD-V2's "Why".

Ledgers are opened read-only, and immutable when their write-ahead log is empty, so reading a
run's artifacts never changes them.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ..pricing import usd
from .runner import _read_jsonl, pressure
from .score import score, would_be_grade

LEDGER_ARMS = ("L", "L4", "L4-nogate")


def run_parts(run_dir: Path) -> list[tuple[dict, Path]]:
    """(run.json, directory) of a run and of each of its rho-* parts."""
    found = [p for p in (run_dir / "run.json", *sorted(run_dir.glob("*/run.json"))) if p.exists()]
    return [(json.loads(p.read_text(encoding="utf-8")), p.parent) for p in found]


def _ledger_rows(db: Path, sql: str) -> list[tuple]:
    wal = Path(f"{db}-wal")
    flags = "mode=ro" if wal.exists() and wal.stat().st_size else "mode=ro&immutable=1"
    con = sqlite3.connect(f"file:{db}?{flags}", uri=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def verdict_reasons(directory: Path) -> Counter:
    """The verdict reason of every claim in a run part's L ledgers (set-aside attempts excluded)."""
    out: Counter = Counter()
    for db in sorted((directory / "L").glob("episode-[0-9][0-9].sqlite")):
        for (payload,) in _ledger_rows(db, "SELECT payload FROM events WHERE type = 'claim'"):
            claim = json.loads(payload)["claim"]
            out[(claim.get("verdict") or {}).get("reason") or claim["status"]] += 1
    return out


def quick(run_dir: Path, questions_for: Callable[[dict], tuple[list[dict], dict[str, str]]]) -> list[dict]:
    """One row per (arm, rho) of the run (see the module docstring). `questions_for(run)` gives
    the run's questions and corpus texts."""
    rows = []
    for run, directory in run_parts(run_dir):
        questions, texts = questions_for(run)
        results = _read_jsonl(directory / "results.jsonl")
        calls = _read_jsonl(directory / "calls.jsonl")
        scored = score(results, questions, texts)
        tools = (run.get("tool_definitions") or {}).get("tokens") or {}
        pressed = pressure(calls, results, run["window"], tools)
        for arm in sorted({r["arm"] for r in scored}):
            mine = [s for s in scored if s["arm"] == arm]
            episodes = len({s["episode"] for s in mine}) or 1
            res = [r for r in results if r["arm"] == arm]
            answerable = [s for s in mine if s["answerable"]]
            templates = {}
            for t in sorted({s["template"] for s in mine}):
                of = [s for s in mine if s["template"] == t]
                templates[t] = dict(Counter(s["outcome"] for s in of))
            rows.append({
                "arm": arm, "rho": run["rho"], "model": run["model"],
                "answerable": len(answerable), **{k: sum(s["outcome"] == k for s in answerable) for k in ("correct", "wrong", "abstain")},
                "unanswerable": len(mine) - len(answerable), "false_answer": sum(s["outcome"] == "false_answer" for s in mine),
                "by_template": templates,
                "folds_per_episode": sum(r.get("folds", 0) for r in res) / episodes,
                "compactions_per_episode": sum(r.get("compactions", 0) for r in res) / episodes,
                "pressure": pressed.get(arm) or {},
                "spend_pusd": sum(c["cost_pusd"] for c in calls if c["arm"] == arm),
                "condition": run.get("condition", "allowed"),
                "forced_violation": sum(bool(r.get("forced_violation")) for r in res),
                "forced_fallback": sum(bool(r.get("forced_fallback")) for r in res),
                "forced_fallback_correct": sum(bool(r.get("forced_fallback")) and s["outcome"] == "correct" for r, s in zip(res, mine)),
                "gate": dict(Counter(r.get("gate_reason") or "not_run" for r in res)) if arm in LEDGER_ARMS else None,
                "gate_refusals": dict(Counter(g for r in res for g in (r.get("gate_reasons") or []) if g not in ("anchored", "no_target")))
                if arm in LEDGER_ARMS else None,
                "no_target": sum("gate_target" in r and r["gate_target"] is None for r in res) if arm in LEDGER_ARMS else None,
                **(v4_counts(res) if arm in ("B5", "L4", "L4-nogate") else {}),
            })  # fmt: skip
    return rows


def v4_counts(res: list[dict]) -> dict:
    """BUILD-V4: what L4's and B5's mechanisms did over these results rows."""
    return {
        "preresolved": dict(Counter((r.get("preresolved") or {}).get("status", "none") for r in res)),
        "repairs": dict(Counter(reason for r in res for reason in (r.get("repair_reasons") or []))),
        "repaired_questions": sum(bool(r.get("repairs")) for r in res),
        "unverified_guess": sum(bool(r.get("unverified_guess")) for r in res),
        "tools": dict(sum((Counter(r.get("tool_counts") or {}) for r in res), Counter())),
    }


def render_quick(rows: list[dict]) -> str:
    lines = ["arm  rho   answerable correct wrong abstain | unanswerable false | folds/ep compactions/ep p95 prompt/W | spend"]
    for r in rows:
        p = r["pressure"]
        shown = lambda key: p.get(key) or "n/a"  # noqa: E731
        lines.append(f"{r['arm']:<4} {r['rho']:<5} {r['answerable']:>10} {r['correct']:>7} {r['wrong']:>5} {r['abstain']:>7} | {r['unanswerable']:>12} "
                     f"{r['false_answer']:>5} | {r['folds_per_episode']:>8.2f} {r['compactions_per_episode']:>14.2f} "
                     f"{shown('prompt_over_w_p95'):>12} | {usd(r['spend_pusd'])}")  # fmt: skip
        lines.append(f"       pressure: prompt minus tools over W p50 {shown('prompt_over_w_p50')}, p95 {shown('prompt_over_w_p95')}, "
                     f"max {shown('prompt_over_w_max')}; {p.get('calls_over_w', 0)} of {p.get('answer_calls', 0)} answer calls above W")  # fmt: skip
        condition = (f"       condition {r.get('condition', 'allowed')}: forced_violation {r.get('forced_violation', 0)}, "
                     f"forced_fallback {r.get('forced_fallback', 0)} ({r.get('forced_fallback_correct', 0)} correct)")  # fmt: skip
        if r.get("gate") is not None:
            condition += (f"; gate: {', '.join(f'{k} {v}' for k, v in sorted(r['gate'].items()))}; refusals: "
                          f"{', '.join(f'{k} {v}' for k, v in sorted(r['gate_refusals'].items())) or 'none'}; no_target {r['no_target']}")  # fmt: skip
        lines.append(condition)
        if "preresolved" in r:
            lines.append(f"       v4: pre-resolution {', '.join(f'{k} {v}' for k, v in sorted(r['preresolved'].items()))}; repairs "
                         f"{sum(r['repairs'].values())} on {r['repaired_questions']} questions ({', '.join(f'{k} {v}' for k, v in sorted(r['repairs'].items())) or 'none'}); "
                         f"unverified_guess {r['unverified_guess']}; tools {', '.join(f'{k} {v}' for k, v in sorted(r['tools'].items())) or 'none'}")  # fmt: skip
        for t, outcomes in r["by_template"].items():
            lines.append(f"       {t}: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))
    return "\n".join(lines)


def diagnose_abstain(run_dir: Path, questions_for: Callable[[dict], list[dict]]) -> list[dict]:
    """Per rho (run part): L's abstentions on answerable questions, cross-tabulated by reason x
    blocked x folded, the number whose blocked answer the grader would score correct, and the
    verdict reasons of the part's L ledgers."""
    out = []
    for run, directory in run_parts(run_dir):
        by_qid = {q["qid"]: q for q in questions_for(run)}
        table: Counter = Counter()
        total = blocked_correct = 0
        for r in _read_jsonl(directory / "results.jsonl"):
            q = by_qid[r["qid"]]
            if r["arm"] != "L" or not q["answerable"] or not (r["abstained"] or r["answer"] is None):
                continue
            grades = [would_be_grade(q, a) for a in r.get("blocked_answers", [])]
            blocked = "blocked_correct" if "correct" in grades else "blocked_other" if grades else "none"
            folded = "folded" if r.get("folds", 0) else "no_fold"
            table[(r["reason"], blocked, folded)] += 1
            total += 1
            blocked_correct += blocked == "blocked_correct"
        out.append({"rho": run["rho"], "model": run["model"], "abstentions": total, "blocked_correct": blocked_correct,
                    "table": dict(sorted(table.items())), "verdict_reasons": dict(verdict_reasons(directory).most_common())})  # fmt: skip
    return out


def render_diagnose(parts: list[dict]) -> str:
    lines = []
    for p in parts:
        lines.append(f"{p['model']} rho {p['rho']}: L abstained on {p['abstentions']} answerable questions; {p['blocked_correct']} after a blocked "
                     "answer the grader would score correct")  # fmt: skip
        lines.append("  reason x blocked x folded:")
        lines += [f"    {reason:<20} {blocked:<16} {folded:<8} {n}" for (reason, blocked, folded), n in p["table"].items()]
        lines.append("  verdict reasons in the ledgers: " + ", ".join(f"{k} {v}" for k, v in p["verdict_reasons"].items()))
    return "\n".join(lines)


__all__ = ["diagnose_abstain", "quick", "render_diagnose", "render_quick", "run_parts", "verdict_reasons"]
