"""BUILD-V3 items 3 and 5: questions v3 (the v2s generator under its own label and file), `run
--questions v3`, and what `eval quick` and `eval plan --measured` show for a v3 run under both
conditions."""

from __future__ import annotations

import hashlib
import json

import pytest

from context_ledger.eval import cli
from context_ledger.eval.corpus import build_corpus
from context_ledger.eval.questions import QuestionError
from context_ledger.eval.questions_v2 import SESSION_FILENAME, V3_FILENAME, Mix, build_questions_v2, load_questions_v2
from context_ledger.eval.runner import questions_sha256
from pilot_fixture import v2_fixture_files, write_fixture_tarball

MIX = "anchors=5,probes=3,multihop=2,unanswerable=2,notes=4,supersessions=1,session=3,session_unanswerable=1"
ARMS = ("B3", "B4", "L")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v3")
    tar, sha = write_fixture_tarball(tmp, v2_fixture_files())
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=400, rhos=(10, 20), n_gold=8, runs_dir=tmp / "runs", cache_dir=tmp / "cache",
                            progress=lambda m: None, version="v2")  # fmt: skip
    return manifest, tmp / "runs"


def test_v3_is_v2s_relabelled_in_its_own_file(corpus):
    manifest, runs = corpus
    v2s = build_questions_v2(manifest, seed=2, episodes=3, questions=12, mix=Mix.parse(MIX), probe_gap=4, runs_dir=runs)
    before = hashlib.sha256((runs / manifest["id"] / SESSION_FILENAME).read_bytes()).hexdigest()
    v3 = build_questions_v2(manifest, seed=2, episodes=3, questions=12, mix=Mix.parse(MIX), probe_gap=4, runs_dir=runs, version="v3")
    assert hashlib.sha256((runs / manifest["id"] / SESSION_FILENAME).read_bytes()).hexdigest() == before  # v2s untouched
    assert all(r["version"] == "v3" and r["qid"] == f"v3e{r['episode']:02d}q{r['pos']:02d}" for r in v3)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("qid", "version", "probe_of")}  # noqa: E731
    assert [strip(r) for r in v3] == [strip(r) for r in v2s]  # the generator is unchanged
    assert [r["probe_of"] is None for r in v3] == [r["probe_of"] is None for r in v2s]
    meta = json.loads((runs / manifest["id"] / V3_FILENAME).read_text().split("\n", 1)[0])["metadata"]
    assert meta["version"] == "v3" and meta["seed"] == 2 and "session" in meta
    assert load_questions_v2(manifest, runs, "v3") == v3
    with pytest.raises(QuestionError):
        build_questions_v2(manifest, seed=2, episodes=3, questions=12, mix=Mix.parse(MIX.split(",notes")[0]), probe_gap=4, runs_dir=runs, version="v3")


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    work = tmp_path_factory.mktemp("v3-run")
    tar, sha = write_fixture_tarball(work, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    assert e("questions", "build", "--version", "v3", "--seed", "2", "--episodes", "2", "--questions-per-episode", "12", "--mix", MIX, "--probe-gap", "4") == 0
    for condition in ("allowed", "forced"):
        assert e("run", "--questions", "v3", "--rho", "10,20", "--episodes", "1-2", "--window", "3000", "--budget", "2", "--run-id", f"v3-{condition}",
                 "--condition", condition) == 0  # fmt: skip
    return work, e


def test_a_v3_run_records_its_questions_and_condition(work):
    path, _ = work
    manifest = json.loads(next((path / "runs" / "corpus").glob("*/manifest.json")).read_text())
    rows = load_questions_v2(manifest, path / "runs" / "corpus", "v3")
    for condition in ("allowed", "forced"):
        for rho in (10, 20):
            meta = json.loads((path / "runs" / f"v3-{condition}" / f"rho-{rho}" / "run.json").read_text())
            assert meta["questions_version"] == "v3" and meta["condition"] == condition
            assert meta["questions_sha256"] == questions_sha256({arm: [r for r in rows if r["episode"] in (1, 2)] for arm in ARMS})


def test_quick_shows_the_condition_forced_counts_and_gate_reasons(work, capsys):
    path, e = work
    capsys.readouterr()
    assert e("quick", str(path / "runs" / "v3-forced")) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip().startswith("condition forced:")]
    assert len(lines) == 6  # one per arm and rho
    gate_lines = [line for line in lines if "; gate: " in line]
    assert len(gate_lines) == 2 and all("no_target 0" in line and "refusals:" in line for line in gate_lines)
    assert any("forced_fallback 0" not in line for line in gate_lines)
    assert e("quick", str(path / "runs" / "v3-allowed")) == 0
    assert "condition allowed: forced_violation 0, forced_fallback 0" in capsys.readouterr().out


def test_plan_measured_prints_per_episode_arm_costs(work, capsys):
    path, e = work
    capsys.readouterr()
    assert e("plan", "--measured", str(path / "runs" / "v3-allowed"), "--window", "3000") == 0
    out = capsys.readouterr().out
    rows = [line.split() for line in out.splitlines() if line.startswith("  claude-haiku-5-5") and "episodes" not in line]
    assert len(rows) == 2 * 3 * 2  # rho x arm x episode
    assert {(r[2], r[3]) for r in rows} == {(rho, arm) for rho in ("10", "20") for arm in ARMS}
    assert "Mean per episode:" in out and "Verdict:" in out
