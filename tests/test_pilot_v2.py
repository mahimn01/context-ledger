"""BUILD-V2 item 7: runner, pilot and grid plumbing for v2 questions, `eval quick`,
`eval diagnose-abstain` and the two labelled analysis files; all offline on the v2 fixture."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from context_ledger.eval import cli
from context_ledger.eval.runner import episode_list, select_questions
from pilot_fixture import v2_fixture_files, write_fixture_tarball

ROOT = Path(__file__).resolve().parents[1]
MIX = "anchors=5,probes=3,multihop=2,unanswerable=2"


def e(work: Path, *argv: str) -> int:
    return cli.main(["--workdir", str(work), *argv])


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """A v2 corpus (C10 and C20) with v2 questions of 12 per episode, and an offline run of all
    arms over rho 10 and 20 and episodes 1-2."""
    work = tmp_path_factory.mktemp("pilot-v2")
    tar, sha = write_fixture_tarball(work, v2_fixture_files())
    assert e(work, "corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    assert e(work, "questions", "build", "--version", "v2", "--episodes", "3", "--questions-per-episode", "12", "--mix", MIX, "--probe-gap", "4") == 0
    assert e(work, "run", "--questions", "v2", "--rho", "10,20", "--episodes", "1-2", "--window", "1500", "--budget", "2", "--run-id", "v2run") == 0
    return work


def results(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_rho_lists_episode_ranges_and_per_episode_counts_from_the_file(work):
    run_dir = work / "runs" / "v2run"
    assert sorted(p.name for p in run_dir.glob("rho-*")) == ["rho-10", "rho-20"]
    for rho in (10, 20):
        run = json.loads((run_dir / f"rho-{rho}" / "run.json").read_text())
        rows = results(run_dir / f"rho-{rho}" / "results.jsonl")
        assert run["questions_version"] == "v2" and run["episodes"] == "1-2" and run["questions_per_episode"] is None
        assert {r["arm"] for r in rows} == {"B3", "B4", "L"} and {r["episode"] for r in rows} == {1, 2}
        for arm in ("B3", "B4", "L"):
            assert sum(r["arm"] == arm for r in rows) == 24  # 12 per episode, as the questions file holds
        assert all(r["qid"].startswith("v2e") and r["rho"] == rho for r in rows)


def test_results_rows_carry_the_question_tags_at_the_runs_rho(work):
    questions = {q["qid"]: q for q in (json.loads(x) for x in next((work / "runs" / "corpus").iterdir()).joinpath("questions-v2.jsonl").read_text().splitlines()) if "qid" in q}
    for rho in (10, 20):
        for r in results(work / "runs" / "v2run" / f"rho-{rho}" / "results.jsonl"):
            q = questions[r["qid"]]
            assert (r["template"], r["hops"], r["cross_file"], r["fold_probe"], r["probe_gap"]) == (q["template"], q["hops"], q["cross_file"], q["fold_probe"], q["probe_gap"])
            assert r["confusers"] == q["confusers"][str(rho)]
    rows = results(work / "runs" / "v2run" / "rho-10" / "results.jsonl")
    assert any(r["hops"] == 2 for r in rows) and any(r["fold_probe"] for r in rows)


def test_the_fake_model_answers_multihop_questions_through_two_hops(work):
    """With room to hold both hops (W = 20,000), the scripted model resolves a named default by
    reading the constant it names, cites each hop, and the one grader scores it correct."""
    assert e(work, "run", "--questions", "v2", "--rho", "10", "--episodes", "1", "--window", "20000", "--budget", "2", "--arms", "B3,L", "--run-id", "wide") == 0
    assert e(work, "score", "wide") == 0
    scores = results(work / "runs" / "wide" / "scores.jsonl")
    rows = {(r["arm"], r["qid"]): r for r in results(work / "runs" / "wide" / "results.jsonl")}
    two_hop = [s for s in scores if rows[(s["arm"], s["qid"])]["hops"] == 2 and s["outcome"] == "correct"]
    assert {s["arm"] for s in two_hop} == {"B3", "L"}
    assert all(len(rows[(s["arm"], s["qid"])]["citations"]) == 2 for s in two_hop)


def test_a_rerun_with_the_same_run_id_skips_every_completed_unit(work, capsys):
    before = {p: p.read_text() for p in (work / "runs" / "v2run").glob("rho-*/calls.jsonl")}
    capsys.readouterr()
    assert e(work, "run", "--questions", "v2", "--rho", "10,20", "--episodes", "1-2", "--window", "1500", "--budget", "2", "--run-id", "v2run") == 0
    out = capsys.readouterr().out
    assert "progress:" not in out and "complete" in out
    assert {p: p.read_text() for p in before} == before


def test_the_grid_is_idempotent_too(work, capsys):
    args = ("run", "--grid", "--questions", "v2", "--rho", "10", "--episodes", "1", "--window", "1500", "--budget", "5", "--arms", "B3,L", "--run-id", "g2")
    assert e(work, *args) == 0
    calls = (work / "runs" / "g2" / "rho-10" / "calls.jsonl").read_text()
    capsys.readouterr()
    assert e(work, *args) == 0
    assert "progress:" not in capsys.readouterr().out and (work / "runs" / "g2" / "rho-10" / "calls.jsonl").read_text() == calls


def test_quick_prints_the_pilot_table(work, capsys):
    capsys.readouterr()
    assert e(work, "quick", str(work / "runs" / "v2run")) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("arm  rho   answerable correct wrong abstain")
    assert sum(line.startswith(("B3 ", "B4 ", "L ")) for line in lines) == 6  # 3 arms x 2 rhos
    assert "default_ref:" in out or "constant_ref:" in out
    assert e(work, "quick", str(work / "nothing")) == 2


def test_diagnose_abstain_on_an_offline_run(work, capsys):
    capsys.readouterr()
    assert e(work, "diagnose-abstain", str(work / "runs" / "v2run")) == 0
    out = capsys.readouterr().out
    assert out.count("L abstained on") == 2 and "verdict reasons in the ledgers:" in out


M4 = [ROOT / "runs" / "grid-haiku-m4", ROOT / "runs" / "grid-sonnet-m4"]


@pytest.mark.skipif(not all(p.is_dir() for p in M4) or not (ROOT / "runs" / "corpus" / "c7580b09b2432" / "questions.jsonl").exists(), reason="M4 grid artifacts absent")
def test_diagnose_reproduces_the_m4_diagnosis_read_only():
    """BUILD-V2 "Why": the M4 L abstentions and ledger verdict counts, read without touching the
    artifacts (ledgers opened immutable)."""
    from context_ledger.eval.corpus import RUNS_CORPUS, load_manifest
    from context_ledger.eval.diagnose import diagnose_abstain
    from context_ledger.eval.runner import load_run_questions

    files = sorted(p for d in M4 for p in d.rglob("*") if p.is_file())
    stamps = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in files}
    get = lambda run: load_run_questions(load_manifest(run["corpus_id"], RUNS_CORPUS), RUNS_CORPUS, run.get("questions_version"))  # noqa: E731
    haiku, sonnet = diagnose_abstain(M4[0], get), diagnose_abstain(M4[1], get)
    assert [(p["rho"], p["abstentions"], p["blocked_correct"]) for p in haiku] == [(10, 25, 20), (100, 27, 25), (1000, 32, 30)]
    assert sum(p["abstentions"] for p in sonnet) == sum(p["blocked_correct"] for p in sonnet) == 29
    totals: dict = {}
    for p in haiku + sonnet:
        for k, v in p["verdict_reasons"].items():
            totals[k] = totals.get(k, 0) + v
    assert {k: totals[k] for k in ("quotes_verified", "identifier_not_anchored", "model_abstained", "value_not_covered", "unknown_ref", "no_valid_quote")} == {
        "quotes_verified": 258, "identifier_not_anchored": 140, "model_abstained": 126, "value_not_covered": 49, "unknown_ref": 24, "no_valid_quote": 16}  # fmt: skip
    assert {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in files} == stamps


def test_analyze_reads_gold_from_the_runs_questions_file_and_writes_the_labelled_files(work, tmp_path, capsys):
    out = tmp_path / "analysis"
    assert e(work, "analyze", str(work / "runs" / "v2run" / "rho-10"), "--out", str(out), "--resamples", "50") == 0
    by_template = (out / "by_template.csv").read_text().splitlines()
    pressure = (out / "pressure.csv").read_text().splitlines()
    assert by_template[0].startswith("# exploratory") and pressure[0].startswith("# manipulation check")
    rows = list(csv.DictReader(by_template[1:]))
    assert {r["template"] for r in rows} >= {"default_ref"} or {r["template"] for r in rows} >= {"constant_ref"}
    assert sum(int(r["n"]) for r in rows if r["arm"] == "L") == 24
    press = {r["arm"]: r for r in csv.DictReader(pressure[1:])}
    assert set(press) == {"B3", "B4", "L"} and all(r["prompt_over_w_p95"] for r in press.values())
    cells = list(csv.DictReader((out / "cells.csv").open()))
    assert {c["arm"] for c in cells} == {"B3", "B4", "L"} and all(c["n_questions"] == "24" for c in cells)
    assert (out / "tests.csv").exists()


def test_episode_specs():
    rows = [{"episode": e, "pos": p, "qid": f"e{e}q{p}"} for e in (1, 2, 3, 4) for p in range(3)]
    assert episode_list(rows, 2) == [1, 2] and episode_list(rows, "3") == [1, 2, 3] and episode_list(rows, "2-4") == [2, 3, 4]
    with pytest.raises(ValueError):
        episode_list(rows, "3-6")
    chosen = select_questions(rows, episodes="2-3", per_episode=None, smoke=False)["L"]
    assert [r["qid"] for r in chosen] == ["e2q0", "e2q1", "e2q2", "e3q0", "e3q1", "e3q2"]
    assert len(select_questions(rows, episodes=1, per_episode=2, smoke=False)["B3"]) == 2
