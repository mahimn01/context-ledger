"""BUILD-V2b item 5: `eval quick` prints a pressure line per arm: the p50, p95 and max of the actual
prompt minus the arm's tool definitions over W, and how many answer calls were above W."""

from __future__ import annotations

from pathlib import Path

import pytest

from context_ledger.eval import cli
from context_ledger.eval.diagnose import render_quick
from context_ledger.eval.runner import pressure

ROOT = Path(__file__).resolve().parents[1]


def call(arm: str, prompt: int, kind: str = "answer") -> dict:
    """A calls.jsonl row whose actual prompt (input + cache read + cache write) is `prompt`."""
    return {"arm": arm, "kind": kind, "usage": {"input_tokens": prompt - 300, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 100}}


def test_pressure_has_the_max_and_the_calls_above_w():
    calls = [call("L", t) for t in (600, 1100, 1300, 2100)] + [call("B4", 900), call("B4", 9000, kind="summary")]
    results = [{"arm": "L", "episode": 1, "folds": 3}, {"arm": "B4", "episode": 1, "compactions": 1}]
    p = pressure(calls, results, 1000, {"L": 100, "B4": 50})
    # L minus tools: 500, 1000, 1200, 2000 over W = 1000; exactly W is not above it
    assert {k: p["L"][k] for k in ("prompt_over_w_p50", "prompt_over_w_p95", "prompt_over_w_max", "calls_over_w", "answer_calls")} == {
        "prompt_over_w_p50": "1", "prompt_over_w_p95": "2", "prompt_over_w_max": "2", "calls_over_w": 2, "answer_calls": 4,
    }  # fmt: skip
    assert p["B4"]["prompt_over_w_max"] == "0.85" and p["B4"]["calls_over_w"] == 0 and p["B4"]["answer_calls"] == 1  # summary calls left out


def test_render_quick_prints_the_pressure_line():
    row = {"arm": "L", "rho": 10, "model": "claude-haiku-5-5", "answerable": 2, "correct": 1, "wrong": 0, "abstain": 1, "unanswerable": 0,
           "false_answer": 0, "by_template": {}, "folds_per_episode": 3.0, "compactions_per_episode": 0.0, "spend_pusd": 0,
           "pressure": {"prompt_over_w_p50": "1", "prompt_over_w_p95": "2", "prompt_over_w_max": "2", "calls_over_w": 2, "answer_calls": 4}}  # fmt: skip
    lines = render_quick([row]).splitlines()
    assert lines[1].startswith("L    10") and lines[1].split("|")[2].split()[-1] == "2"
    assert lines[2] == "       pressure: prompt minus tools over W p50 1, p95 2, max 2; 2 of 4 answer calls above W"
    empty = render_quick([{**row, "pressure": {}}]).splitlines()[2]
    assert empty == "       pressure: prompt minus tools over W p50 n/a, p95 n/a, max n/a; 0 of 0 answer calls above W"


PILOT = ROOT / "runs" / "pilot-v2a-haiku"


@pytest.mark.skipif(not (PILOT / "rho-10" / "run.json").exists() or not (ROOT / "runs" / "corpus" / "ca8e01d67009c").is_dir(), reason="v2a pilot artifacts absent")
def test_quick_shows_the_v2a_pilots_pressure_read_only(capsys):
    """BUILD-V2b's numbers: L's prompt minus its tool definitions was p50 1.706 W and p95 2.86 W
    at rho 10 (p95 3.278 W at rho 1000), against B3's 0.86 W and B4's 0.92 W."""
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in PILOT.rglob("*") if p.is_file()}
    assert cli.main(["quick", str(PILOT)]) == 0
    lines = capsys.readouterr().out.splitlines()
    pressure_of = {(lines[k - 1].split()[0], lines[k - 1].split()[1]): lines[k] for k in range(1, len(lines)) if lines[k].startswith("       pressure:")}
    assert pressure_of[("L", "10")] == "       pressure: prompt minus tools over W p50 1.706, p95 2.86, max 3.24; 52 of 69 answer calls above W"
    assert "p95 3.278" in pressure_of[("L", "1000")]
    assert "p95 0.8567" in pressure_of[("B3", "10")] and "p95 0.9173" in pressure_of[("B4", "10")]
    assert {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in PILOT.rglob("*") if p.is_file()} == before
