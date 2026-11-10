"""M4 item 8: the grid is idempotent and resumable unit by unit (arm, model, rho, episode)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from context_ledger.anthropic.fake import FakeAnthropicServer
from context_ledger.eval import cli
from context_ledger.eval.arms import set_aside_episode
from context_ledger.eval.corpus import build_corpus
from context_ledger.eval.fakemodel import ScriptedPilotModel
from context_ledger.eval.questions import build_questions
from context_ledger.eval.runner import RunConfigMismatch, load_run, run_pilot
from context_ledger.live.guarded import GuardedClient
from context_ledger.live.spend import OpenReservation, SpendGuard, read_spend
from context_ledger.pricing import PricingTable
from context_ledger.tokens import BYTES4_V0
from pilot_fixture import write_fixture_tarball

TABLE = PricingTable.verified_default()
ARMS = ["B3", "B4", "L"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("resume")
    tar, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    return manifest, build_questions(manifest, seed=9, episodes=2, runs_dir=tmp / "runs")


class Killed(BaseException):
    """The process dies mid-call (like SIGKILL): no handler runs, the reservation stays open."""


def server() -> FakeAnthropicServer:
    return FakeAnthropicServer(responder=ScriptedPilotModel(), pricing=TABLE, emulate=True, ttl_calls=0)  # no cache reads: costs depend on requests only


def client(run_dir, srv, kill_at: int | None = None) -> GuardedClient:
    raw = srv.as_fake_client()
    calls = {"n": 0}

    def create(**body):
        calls["n"] += 1
        if calls["n"] == kill_at:
            raise Killed()
        return raw.messages.create(**body)

    guard = SpendGuard(run_dir / "fake-spend.jsonl", lock_path=run_dir / "fake-spend.lock", run_id="r", pricing=TABLE, run_cap_usd="0.75")
    return GuardedClient(SimpleNamespace(messages=SimpleNamespace(create=create, count_tokens=raw.messages.count_tokens)), guard, sleep=lambda s: None)


def go(run_dir, corpus, c, progress=None, window=1500):
    manifest, rows = corpus
    return run_pilot(manifest=manifest, rows=rows, arms=ARMS, rho=10, model="claude-haiku-5-5", window=window, episodes=2, per_episode=8, smoke=False,
                     run_dir=run_dir, run_id="r", pricing=TABLE, estimator=BYTES4_V0, client=c, live=False, budget_usd="0.75", progress=progress)  # fmt: skip


def units_of(lines: list[str]) -> list[str]:
    return [line.split("claude-haiku-5-5 ", 1)[1].split(":", 1)[0] for line in lines]


def test_a_killed_run_resumes_without_rerunning_any_unit(tmp_path, corpus):
    ref_dir = tmp_path / "ref"
    ref = client(ref_dir, server())
    go(ref_dir, corpus, ref)
    ref.guard.close()
    _, ref_results, ref_calls = load_run(ref_dir)
    per_unit = {}
    for c in ref_calls:
        per_unit[(c["arm"], c["episode"])] = per_unit.get((c["arm"], c["episode"]), 0) + 1
    kill_at = per_unit[("B3", 1)] + per_unit[("B3", 2)] + per_unit[("B4", 1)] + 2  # inside (B4, 2)

    run_dir, srv = tmp_path / "run", server()  # the "API" outlives the killed process
    first: list[str] = []
    dying = client(run_dir, srv, kill_at)
    with pytest.raises(Killed):
        go(run_dir, corpus, dying, first.append)
    dying.guard.close()  # the process is gone; its lock with it
    assert units_of(first) == ["B3 episode 1", "B3 episode 2", "B4 episode 1"]
    assert json.loads((run_dir / "run.json").read_text())["status"] == "running"
    with pytest.raises(OpenReservation):
        client(run_dir, srv)
    assert cli.main(["spend", "resolve", "--assume-spent", "--spend-file", str(run_dir / "fake-spend.jsonl")]) == 0

    second: list[str] = []
    resumed = client(run_dir, srv)
    summary = go(run_dir, corpus, resumed, second.append)
    resumed.guard.close()
    assert units_of(second) == ["B4 episode 2", "L episode 1", "L episode 2"]  # no unit ran twice
    assert summary["units"] == {"done": 6, "total": 6} and not summary["incomplete"]

    _, results, calls = load_run(run_dir)
    key = lambda r: (r["arm"], r["qid"])  # noqa: E731
    assert sorted(results, key=key) == sorted(ref_results, key=key)  # identical rows, costs included
    assert sum(c["cost_pusd"] for c in calls) == sum(c["cost_pusd"] for c in ref_calls) and len(calls) == len(ref_calls)
    discarded = [json.loads(line) for line in (run_dir / "calls.discarded.jsonl").read_text().splitlines()]
    assert len(discarded) == 1 and (discarded[0]["arm"], discarded[0]["episode"]) == ("B4", 2)  # the cut-short attempt, kept
    assert (run_dir / "B4" / "episode-02.jsonl.discarded-1").exists()
    spend = read_spend(run_dir / "fake-spend.jsonl")
    assert any(r["kind"] == "assume_spent" for r in spend)  # the killed call's spend stays recorded
    assert sum(r["actual_pusd"] for r in spend if r["kind"] == "settle") > sum(c["cost_pusd"] for c in calls)


def test_set_aside_keeps_every_earlier_attempt(tmp_path):
    (tmp_path / "L").mkdir()
    for suffix in ("", "-wal", "-shm"):
        (tmp_path / "L" / f"episode-03.sqlite{suffix}").write_text("one")
    assert len(set_aside_episode(tmp_path, "L", 3)) == 3 and set_aside_episode(tmp_path, "L", 3) == []
    (tmp_path / "L" / "episode-03.sqlite").write_text("two")
    assert set_aside_episode(tmp_path, "L", 3) == [tmp_path / "L" / "episode-03.sqlite.discarded-2"]
    assert set_aside_episode(tmp_path, "B3", 3) == []


def test_a_finished_run_is_idempotent_and_another_config_is_refused(tmp_path, corpus):
    srv = server()
    c = client(tmp_path, srv)
    go(tmp_path, corpus, c)
    before, requests = (tmp_path / "results.jsonl").read_text(), len(srv.requests)
    lines: list[str] = []
    go(tmp_path, corpus, c, lines.append)
    assert lines == [] and len(srv.requests) == requests and (tmp_path / "results.jsonl").read_text() == before
    with pytest.raises(RunConfigMismatch, match="config hash"):
        go(tmp_path, corpus, c, window=1600)
    c.guard.close()


def test_a_budget_stop_keeps_the_partial_episode_flagged_and_a_resume_redoes_it(tmp_path, corpus):
    manifest, rows = corpus
    srv = server()
    guard = SpendGuard(tmp_path / "fake-spend.jsonl", lock_path=tmp_path / "fake-spend.lock", run_id="r", pricing=TABLE, run_cap_usd="0.003")
    c = GuardedClient(srv.as_fake_client(), guard, sleep=lambda s: None)
    summary = go(tmp_path, corpus, c)
    guard.close()
    assert summary["incomplete"] and summary["stopped"]["reason"] == "budget_refused"
    _, results, _ = load_run(tmp_path)
    partial = [r for r in results if r.get("partial")]
    assert partial and all((r["arm"], r["episode"]) == (summary["stopped"]["arm"], summary["stopped"]["episode"]) for r in partial)
    c = client(tmp_path, srv)  # a bigger budget for the same configuration
    assert not go(tmp_path, corpus, c)["incomplete"]
    c.guard.close()
    _, results, _ = load_run(tmp_path)
    assert not any(r.get("partial") for r in results) and len(results) == 3 * 16
    assert (tmp_path / "results.discarded.jsonl").read_text().count('"partial": true') == len(partial)
