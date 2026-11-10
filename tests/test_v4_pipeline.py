"""BUILD-V4 items 7, 9 and 11 and BUILD-V4B decisions 1 and 7 end to end, offline: `eval run --arms
B3,B4,B5,L,L4,L4-nogate --max-calls-episode 260` under both conditions (the clp-v4 prompts, the v4
block and the bound in run.json, the v4 fields in v4 rows only), `eval quick` with the v4 line,
`eval analyze-v4`, and `eval plan --v4` priced from the runs as a measured pilot."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from context_ledger.eval import cli
from context_ledger.eval.prompts import CONDITION_PROMPTS_V4, prompt_sha256
from context_ledger.eval.analyze import HAIKU, SONNET
from context_ledger.eval.plan_v4 import GRIDS, plan_v4
from context_ledger.invariants import check
from context_ledger.store import Ledger, SqliteBlobStore
from pilot_fixture import v2_fixture_files, write_fixture_tarball

MIX = "anchors=5,probes=3,multihop=2,unanswerable=2,notes=4,supersessions=1,session=3,session_unanswerable=1"
ARMS = ("B3", "B4", "B5", "L", "L4", "L4-nogate")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    work = tmp_path_factory.mktemp("v4pipe")
    tar, sha = write_fixture_tarball(work, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    assert e("run", "--arms", "L4", "--questions", "v3", "--episodes", "1", "--budget", "1", "--run-id", "no-index") == 2  # no address index yet: refused
    assert e("index", "build") == 0
    assert e("questions", "build", "--version", "v3", "--episodes", "2", "--questions-per-episode", "12", "--mix", MIX, "--probe-gap", "4") == 0
    for condition in ("allowed", "forced"):
        assert e("run", "--arms", ",".join(ARMS), "--questions", "v3", "--rho", "10", "--episodes", "1-2", "--window", "3000", "--budget", "2",
                 "--run-id", f"v4-{condition}", "--condition", condition, "--max-calls-episode", "260") == 0  # fmt: skip
    return work, e


def rows(work, condition):
    return [json.loads(line) for line in (work / "runs" / f"v4-{condition}" / "results.jsonl").read_text().splitlines()]


def test_a_run_with_v4_arms_records_the_v4_prompts_and_block(runs):
    work, _ = runs
    for condition in ("allowed", "forced"):
        run = json.loads((work / "runs" / f"v4-{condition}" / "run.json").read_text())
        assert run["prompts"]["version"] == CONDITION_PROMPTS_V4[condition] and set(run["prompts"]["sha256"]) == set(ARMS) - {"L4-nogate"}  # it sends L4's
        assert run["prompts"]["sha256"]["L"] == prompt_sha256("L", "clp-v3-" + condition)  # L's prompt is v3's
        assert run["v4"]["l4_mechanism"] == "l-v4" and run["v4"]["index"]["version"] == "address-v4" and len(run["v4"]["index"]["root"]) == 64
        assert run["v4"]["l4_nogate_mechanism"] == "l-v4-nogate"
        assert run["l_mechanism"] == "l-v3" and run["max_calls_episode"] == 260 and run["leakscan"] == "pass" and not run["incomplete"]
        assert set(run["tool_definitions"]["tokens"]) == set(ARMS)
        mine = rows(work, condition)
        assert Counter(r["arm"] for r in mine) == {arm: 32 for arm in ARMS}
        for r in mine:
            assert ("preresolved" in r) == (r["arm"] in ("B5", "L4", "L4-nogate"))  # v3 arms' rows keep v3's fields only
        assert all(r["mechanism"] == {"B5": "b5-v4", "L4": "l-v4", "L4-nogate": "l-v4-nogate"}[r["arm"]] for r in mine if r["arm"] in ("B5", "L4", "L4-nogate"))


def test_l4_ledgers_check_clean_and_nothing_unverified_leaves(runs):
    work, _ = runs
    for condition in ("allowed", "forced"):
        for db in sorted((work / "runs" / f"v4-{condition}" / "L4").glob("episode-*.sqlite")):
            events, blobs = Ledger(db).events(), SqliteBlobStore(db)
            assert check(events, blobs) == []
            assert all(e.payload["abstained"] or e.payload["gate"]["passed"] for e in events if e.type == "answer_emitted")
        nogate = sorted((work / "runs" / f"v4-{condition}" / "L4-nogate").glob("episode-*.sqlite"))
        assert nogate and all(check(Ledger(db).events(), SqliteBlobStore(db)) == [] for db in nogate)
    forced = [r for r in rows(work, "forced") if r["arm"] == "L4"]
    guesses = [r for r in forced if r["unverified_guess"]]
    assert all(r["reason"] == "unverified_guess" and r["forced_fallback"] and r["answer"] is not None for r in guesses)
    assert all(r["reason"] in ("answered", "unverified_guess", "forced_noanswer", "budget_exceeded") for r in forced)


def test_quick_analyze_v4_and_plan_v4(runs, capsys, tmp_path):
    work, e = runs
    assert e("quick", "v4-allowed") == 0
    out = capsys.readouterr().out
    assert out.count("v4: pre-resolution") == 3  # B5, L4 and L4-nogate
    assert e("analyze-v4", "v4-allowed", "v4-forced", "--out", str(tmp_path / "an"), "--resamples", "50") == 0
    cells = (tmp_path / "an" / "cells.csv").read_text()
    assert ",L4,gated," in cells and ",B5,primary," in cells and ",L4-nogate,primary," in cells
    pilot = [work / "runs" / "v4-allowed", work / "runs" / "v4-forced"]
    p = plan_v4(None, pilot)  # the fake runs as a measured pilot at 260 calls per episode
    assert {p["costs"][(HAIKU, c, a)][1] for c in ("allowed", "forced") for a in ARMS} == {"pilot"} and not p["ignored_pilot"]
    assert p["costs"][(SONNET, "allowed", "L4-nogate")][1].startswith("pilot x v3 sonnet/haiku ratio")
    assert all(v > 0 for v in p["items"].values()) and set(GRIDS) <= set(p["items"])
    other = plan_v4(None, pilot, max_calls=90)  # another bound: the pilot is not used
    assert len(other["ignored_pilot"]) == 2 * len(ARMS) and other["costs"][(HAIKU, "allowed", "L4")][1] == "v3 L x 1"
    base = plan_v4(None)  # the built-in v3 means, scaled to 260 calls
    assert base["costs"][(HAIKU, "allowed", "B4")][0] > 25_200_000_000 and base["costs"][(HAIKU, "allowed", "B3")][0] == 7_000_000_000
    assert e("plan", "--v4", "--measured", "v4-allowed", "v4-forced", "--model", SONNET, "--episodes", "4", "--rhos", "10,10000", "--arms", "B3,L4") == 0
    out = capsys.readouterr().out
    assert "260 calls per episode" in out and "custom grid" in out and "--budget $" in out


def test_the_episode_call_bound_is_recorded_only_when_changed_and_binds_every_arm(runs):
    work, e = runs
    assert e("run", "--arms", "B4,L,B5,L4", "--questions", "v3", "--rho", "10", "--episodes", "1", "--window", "3000", "--budget", "2",
             "--run-id", "v4-cap", "--max-calls-episode", "5") == 0  # fmt: skip
    run = json.loads((work / "runs" / "v4-cap" / "run.json").read_text())
    assert run["max_calls_episode"] == 5
    for arm in ("B4", "L", "B5", "L4"):
        assert sum(r["calls"] for r in json.loads("[" + ",".join((work / "runs" / "v4-cap" / "results.jsonl").read_text().splitlines()) + "]") if r["arm"] == arm) <= 5
    assert json.loads((work / "runs" / "v4-allowed" / "run.json").read_text())["max_calls_episode"] == 260  # decision 1: recorded


def test_analyze_v4_on_a_stopped_run_and_its_fix1_rerun_matches_the_run_that_never_stopped(runs, tmp_path, capsys):
    """BUILD-V4FIX: the forced fixture run copied twice, once stopped in L4 episode 2 (rows cut short and
    flagged partial, as the grid's Sonnet runs stopped), once as its -fix1 rerun of L and L4 (episode 2
    of L4, and all of L, which then supersedes the stopped run's complete L units). analyze-v4 over
    them writes exactly what it writes over the complete run, and tests.json's meta lists the dropped
    and the superseded units."""
    import shutil

    work, e = runs
    source = work / "runs" / "v4-forced"
    stopped, rerun = tmp_path / "v4-grid-forced-c260", tmp_path / "v4-grid-forced-c260-fix1"
    shutil.copytree(source, stopped, ignore=shutil.ignore_patterns("*.sqlite", "B4", "B5"))
    run = json.loads((stopped / "run.json").read_text())
    results = [json.loads(x) for x in (stopped / "results.jsonl").read_text().splitlines()]
    redo = [r for r in results if (r["arm"], r["episode"]) == ("L4", 2)]
    later = [r for r in results if r["arm"] == "L4-nogate" and r["episode"] == 2]  # the arm after L4: never started in the stopped run
    kept = [r for r in results if (r["arm"], r["episode"]) != ("L4", 2) and r not in later] + [{**r, "partial": True} for r in redo[:3]]
    (stopped / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    (stopped / "run.json").write_text(json.dumps({**run, "run_id": stopped.name, "incomplete": True, "stopped": {"arm": "L4", "episode": 2, "reason": "api_errors"}}))
    rerun.mkdir()
    arms = ["L", "L4", "L4-nogate"]
    (rerun / "run.json").write_text(json.dumps({**run, "run_id": rerun.name, "arms": arms, "questions": {a: run["questions"][a] for a in arms}}))
    (rerun / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results if r["arm"] == "L" or r in redo or r in later))
    allowed = work / "runs" / "v4-allowed"
    assert e("analyze-v4", str(allowed), str(source), "--out", str(tmp_path / "complete"), "--resamples", "50") == 0
    assert e("analyze-v4", str(allowed), *map(str, sorted([rerun, stopped])), "--out", str(tmp_path / "merged"), "--resamples", "50") == 0
    capsys.readouterr()
    for name in ("cells.csv", "tests.csv", "figure_data.csv"):
        assert (tmp_path / "merged" / name).read_bytes() == (tmp_path / "complete" / name).read_bytes(), name
    complete, merged = (json.loads((tmp_path / d / "tests.json").read_text()) for d in ("complete", "merged"))
    assert merged["tests"] == complete["tests"] and complete["meta"]["dropped_units"] == complete["meta"]["superseded_units"] == []
    unit = lambda arm, ep: [run["model"], run["window"], "forced", arm, 10, ep]  # noqa: E731
    assert merged["meta"]["dropped_units"] == [{"unit": unit("L4", 2), "run": stopped.name, "rows": 3, "questions": len(redo), "reason": "run_stopped"}]
    assert merged["meta"]["superseded_units"] == [{"unit": unit("L", ep), "run": stopped.name, "replaced_by": rerun.name} for ep in (1, 2)]


def test_plan_v4_measures_only_units_that_ran_to_the_end(tmp_path, capsys):
    """BUILD-V4FIX: the grid's Sonnet L units the 400s cut short (the unit a run stopped in, partial
    rows, API-error rows) would underprice the rerun, so `plan --v4 --measured` leaves them out."""
    from context_ledger.eval.plan_v4 import measured

    run_dir = tmp_path / "v4-grid-sonnet-forced-r10-e1-4-c260"
    run_dir.mkdir()
    run = {"run_id": run_dir.name, "model": SONNET, "condition": "forced", "rho": 10, "max_calls_episode": 260, "stopped": {"arm": "L", "episode": 3, "reason": "api_errors"}}
    (run_dir / "run.json").write_text(json.dumps(run))
    units = {("B3", 1): {}, ("L", 1): {}, ("L", 2): {"error": {"status": 400}}, ("L", 3): {}, ("L4", 1): {"partial": True}}
    rows = [{"arm": a, "episode": e, "qid": f"q{a}{e}{k}", "calls": 2, **(flag if k == 0 else {})} for (a, e), flag in units.items() for k in range(2)]
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    calls = [{"arm": a, "episode": e, "cost_pusd": 10 ** 9 * (e + (a == "L"))} for (a, e) in units for _ in range(2)]
    (run_dir / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    cost, per_question, bound = measured([run_dir])
    assert cost == {(SONNET, "forced", "B3"): 2 * 10**9, (SONNET, "forced", "L"): 4 * 10**9}  # L episode 1 only; L4 had none complete
    assert per_question == {(SONNET, "forced", "B3"): 2, (SONNET, "forced", "L"): 2} and bound[(SONNET, "forced", "L4")] == 260
    assert cli.main(["--workdir", str(tmp_path), "plan", "--v4", "--measured", str(run_dir), "--model", SONNET, "--episodes", "2", "--rhos", "10",
                     "--arms", "L", "--conditions", "forced"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    custom = [line for line in out.splitlines() if line.lstrip().startswith("custom grid")]
    assert custom[0].split()[2] == "$0.008000" and [line.split()[2] for line in custom[1:]] == ["forced"]  # L at $0.004 x 2 episodes x 1 rho, forced only
