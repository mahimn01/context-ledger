"""F.3: the offline pilot end to end, on a fixture tarball built in-test.

Build a tiny CPython-like corpus (with a test-only sha256 override), perturb it, generate
questions, run all three arms against the FakeAnthropicServer with the scripted model, then
score, report and check the spend ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from context_ledger.eval.corpus import build_corpus, corpus_texts
from context_ledger.eval.leakscan import scan
from context_ledger.eval.questions import build_questions
from context_ledger.eval.report import build_report
from context_ledger.eval.runner import load_run, offline_client, run_pilot
from context_ledger.eval.score import score_run
from context_ledger.invariants import check
from context_ledger.live.spend import read_spend
from context_ledger.pricing import PricingTable
from context_ledger.store import Ledger, SqliteBlobStore
from context_ledger.tokens import BYTES4_V0
from pilot_fixture import write_fixture_tarball

TABLE = PricingTable.verified_default()


def pilot(tmp: Path) -> dict:
    tar, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    rows = build_questions(manifest, seed=9, episodes=2, runs_dir=tmp / "runs")
    run_dir = tmp / "runs" / "pilot"
    client, guard, server = offline_client(run_dir, run_id="pilot", pricing=TABLE, budget_usd="0.75")
    try:
        summary = run_pilot(manifest=manifest, rows=rows, arms=["B3", "B4", "L"], rho=10, model="claude-haiku-5-5", window=1500, episodes=2,
                            per_episode=8, smoke=False, run_dir=run_dir, run_id="pilot", pricing=TABLE, estimator=BYTES4_V0, client=client,
                            live=False, budget_usd="0.75")  # fmt: skip
    finally:
        guard.close()
    scores = score_run(run_dir, rows, corpus_texts(manifest))
    report = build_report(run_dir, resamples=1000)
    return {"run_dir": run_dir, "summary": summary, "scores": scores, "report": report, "server": server, "manifest": manifest}


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    return pilot(tmp_path_factory.mktemp("pilot-a")), pilot(tmp_path_factory.mktemp("pilot-b"))


def test_outcomes_are_deterministic(two_runs):
    a, b = two_runs
    keep = lambda rows: [{k: r[k] for k in ("arm", "qid", "outcome", "hallucination", "caught", "calls", "cost_pusd")} for r in rows]  # noqa: E731
    assert keep(a["scores"]) == keep(b["scores"]) and len(a["scores"]) == 48
    assert a["report"]["cells"] == b["report"]["cells"]
    heads = [[Ledger(db).head() for db in sorted((r["run_dir"] / "L").glob("*.sqlite"))] for r in (a, b)]
    assert heads[0] == heads[1] and len(heads[0]) == 2


def test_spend_ledger_reconciles_to_the_model_call_costs_exactly(two_runs):
    run = two_runs[0]
    _, results, calls = load_run(run["run_dir"])
    records = read_spend(run["run_dir"] / "fake-spend.jsonl")
    settled = sum(r["actual_pusd"] for r in records if r["kind"] == "settle")
    assert settled == sum(c["cost_pusd"] for c in calls) == sum(r["cost_pusd"] for r in results) > 0
    assert [r["kind"] for r in records].count("reserve") == len(calls) == [r["kind"] for r in records].count("settle")
    ledger_costs = 0
    for db in sorted((run["run_dir"] / "L").glob("*.sqlite")):
        ledger, blobs = Ledger(db), SqliteBlobStore(db)
        events = ledger.events()
        assert check(events, blobs) == []
        ledger_costs += sum(e.payload["cost_pusd"] for e in events if e.type == "model_call")
    assert ledger_costs == sum(c["cost_pusd"] for c in calls if c["arm"] == "L")
    assert all(r["actual_pusd"] <= r["worst_case_pusd"] for r in records if r["kind"] == "settle")


def test_predicted_fields_are_present(two_runs):
    _, _, calls = load_run(two_runs[0]["run_dir"])
    assert calls and all(isinstance(c["predicted_prompt"], int) and isinstance(c["predicted_cache_read"], int) for c in calls)
    for db in sorted((two_runs[0]["run_dir"] / "L").glob("*.sqlite")):
        calls_l = [e.payload for e in Ledger(db).events() if e.type == "model_call"]
        assert all(set(c["predicted"]) == {"prompt", "cache_read", "window"} for c in calls_l)  # BUILD-V2b: L's measured W
    per_call = two_runs[0]["report"]["calls"]
    assert all({"predicted_prompt", "reported_prompt", "predicted_cache_read", "reported_cache_read", "reported_cache_write"} <= set(c) for c in per_call)


def test_a_fold_fires_in_l_and_its_savings_are_reported(two_runs):
    run = two_runs[0]
    folds = run["report"]["folds"]
    assert folds and all(f["predicted_delta"] > 0 and f["actual_delta"] is not None for f in folds)
    assert any(f["predicted_rewrite_pusd"] is not None for f in folds)
    assert "## Folds: predicted vs actual (L)" in (run["run_dir"] / "report.md").read_text()
    assert run["report"]["cells"]["L|10|claude-haiku-5-5"]["counts"]["hallucination_caught"] >= 1  # refetch routing corrected a guess


def test_b4_compaction_fires(two_runs):
    run = two_runs[0]
    _, results, calls = load_run(run["run_dir"])
    assert any(r["compactions"] for r in results if r["arm"] == "B4")
    assert any(c["kind"] == "summary" and c["arm"] == "B4" for c in calls)


def test_leakscan_passes(two_runs):
    run = two_runs[0]
    report = scan(runtime=[run["run_dir"], Path(run["manifest"]["cache_dir"])], git=True)
    assert report.ok, report.render()


@pytest.mark.corpus
def test_real_c10_manifest_perturbation_and_question_filters(tmp_path):
    """Opt-in (`pytest -m corpus`): build from the real CPython 3.15.0 tarball."""
    from collections import Counter

    from context_ledger.eval import pyscan
    from context_ledger.eval.corpus import RELEASE, TARBALL, TARBALL_SHA256, read_sources

    if not TARBALL.exists():
        pytest.skip("the CPython tarball is not in .cache/cpython")
    manifest = build_corpus(seed=1, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache", progress=lambda m: None)
    assert manifest["release"] == RELEASE and manifest["sha256"] == TARBALL_SHA256 and manifest["perturbed"]
    assert all(int(manifest["totals"][r]) >= int(manifest["targets"][r]) for r in manifest["targets"])
    assert manifest["targets"] == {"10": 60_000, "100": 600_000, "1000": 6_000_000}  # W = 6,000 (M4)
    paths = [f["path"] for f in manifest["files"]]
    c10 = [f["path"] for f in manifest["files"] if f["rho"] == 10]
    assert paths[: len(manifest["gold"])] == manifest["gold"] and c10 == paths[: len(c10)]  # W = 6,000: C10 holds the first gold files
    original = dict(read_sources())
    texts = corpus_texts(manifest)
    for path, text in texts.items():
        if path.endswith(".py") and pyscan.parse(original[path]) is not None:
            assert pyscan.parse(text) is not None, path  # every file that parsed still parses
    rows = build_questions(manifest, seed=1, runs_dir=tmp_path / "runs")
    assert len({r["episode"] for r in rows}) == 12 and sum(r["answerable"] for r in rows) == 72
    assert sum(r["fold_probe"] for r in rows) == 36 and sum(not r["answerable"] for r in rows) == 24
    defined = Counter(n for p, t in texts.items() if p.endswith(".py") and pyscan.parse(t) for n in pyscan.defined_names(pyscan.parse(t)))
    in_c10 = {f["path"] for f in manifest["files"] if f["rho"] == 10}
    edits = [json.loads(line) for line in (tmp_path / "runs" / manifest["id"] / "perturbations.jsonl").read_text().splitlines()]
    for r in rows:
        if r["answerable"]:
            assert r["gold_path"] in in_c10 and defined[r["subject"]] == 1
            if r["template"] in ("default", "constant"):
                assert r["answer"] != r["upstream_answer"] and any(e["new"] == r["answer"] and e["path"] == r["gold_path"] for e in edits)
        else:
            assert all(r["subject"] not in t for t in texts.values())
