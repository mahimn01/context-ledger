"""D: the three arms against the fake server and the scripted model, on the fixture corpus."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from context_ledger.eval.arms import B4Episode, ArmConfig, CallLog, LEpisode, run_b3, run_b4, run_l
from context_ledger.eval.corpus import CorpusIndex, build_corpus
from context_ledger.eval.fakemodel import ScriptedPilotModel, find_answer
from context_ledger.eval.prompts import COMPACTION_PROMPT, parse_pilot_answer, pilot_answer_text, prompt_sha256, system_prompt
from context_ledger.eval.questions import build_questions
from context_ledger.eval.runner import load_run, offline_client, run_pilot, select_questions
from context_ledger.invariants import check
from context_ledger.live.spend import read_spend
from context_ledger.pricing import PricingTable
from context_ledger.prompts import AnswerFormatError
from context_ledger.store import Ledger, SqliteBlobStore
from context_ledger.tokens import BYTES4_V0
from pilot_fixture import write_fixture_tarball

TABLE = PricingTable.verified_default()
HAIKU = "claude-haiku-5-5"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("arms")
    tar, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    return manifest, build_questions(manifest, seed=9, episodes=2, runs_dir=tmp / "runs")


def run(tmp_path, corpus, arms=("B3", "B4", "L"), window=1500, episodes=1, model=HAIKU, budget="0.75", **kw):
    manifest, rows = corpus
    client, guard, server = offline_client(tmp_path, run_id="t", pricing=TABLE, budget_usd=budget)
    try:
        summary = run_pilot(manifest=manifest, rows=rows, arms=list(arms), rho=10, model=model, window=window, episodes=episodes, per_episode=8,
                            smoke=False, run_dir=tmp_path, run_id="t", pricing=TABLE, estimator=BYTES4_V0, client=client, live=False, budget_usd=budget, **kw)  # fmt: skip
    finally:
        guard.close()
    return summary, server


def ledgers(run_dir):
    out = []
    for db in sorted((run_dir / "L").glob("*.sqlite")):
        ledger, blobs = Ledger(db), SqliteBlobStore(db)
        out.append((ledger.events(), blobs))
    return out


def test_all_arms_run_and_reconcile(tmp_path, corpus):
    summary, server = run(tmp_path, corpus)
    _, results, calls = load_run(tmp_path)
    assert summary["incomplete"] is False and {r["arm"] for r in results} == {"B3", "B4", "L"}
    assert len(results) == 3 * 8 and all(r["rho"] == 10 and r["model"] == HAIKU for r in results)
    b3 = [r for r in results if r["arm"] == "B3"]
    assert all(r["calls"] == 1 for r in b3)
    assert any(r["compactions"] for r in results if r["arm"] == "B4")  # compaction fires at W = 1500
    assert any(r["folds"] for r in results if r["arm"] == "L")  # and so does the backstop
    for events, blobs in ledgers(tmp_path):
        assert check(events, blobs) == [] and any(e.type == "fold" for e in events) and any(e.type == "tokenizer_set" for e in events)
    settled = sum(r["actual_pusd"] for r in read_spend(tmp_path / "fake-spend.jsonl") if r["kind"] == "settle")
    assert settled == sum(c["cost_pusd"] for c in calls) == summary["spend"]["spent_pusd"] > 0
    assert all({"predicted_prompt", "predicted_cache_read", "usage"} <= set(c) for c in calls)
    assert all(isinstance(c["predicted_prompt"], int) for c in calls)
    assert summary["prompts"]["sha256"] == {arm: prompt_sha256(arm) for arm in ("B3", "B4", "L")}
    assert all("temperature" not in r and r.get("thinking") == {"type": "disabled"} for r in server.requests)


def test_b3_fills_the_window_and_no_more(tmp_path, corpus):
    manifest, rows = corpus
    run(tmp_path, corpus, arms=("B3",), window=1200)
    _, results, _ = load_run(tmp_path)
    assert len(results) == 8
    text = (tmp_path / "calls.jsonl").read_text()
    assert text.count('"arm": "B3"') == 8


def test_b4_transcript_and_summary_calls(tmp_path, corpus):
    run(tmp_path, corpus, arms=("B4",))
    events = [json.loads(line) for line in (tmp_path / "B4" / "episode-01.jsonl").read_text().splitlines()]
    kinds = {e["type"] for e in events}
    assert {"question", "response", "tool", "answer", "compaction"} <= kinds
    _, _, calls = load_run(tmp_path)
    assert any(c["kind"] == "summary" for c in calls)  # the summary call's cost counts toward B4


def test_runs_are_deterministic(tmp_path, corpus):
    run(tmp_path / "a", corpus)
    run(tmp_path / "b", corpus)
    strip = lambda d: [{k: v for k, v in r.items()} for r in load_run(d)[1]]  # noqa: E731
    assert strip(tmp_path / "a") == strip(tmp_path / "b")
    heads = [[Ledger(db).head() for db in sorted((d / "L").glob("*.sqlite"))] for d in (tmp_path / "a", tmp_path / "b")]
    assert heads[0] == heads[1]


def test_question_and_episode_bounds(tmp_path, corpus):
    manifest, rows = corpus
    client, guard, _ = offline_client(tmp_path, run_id="b", pricing=TABLE, budget_usd="0.75")
    index = CorpusIndex(manifest["cache_dir"], 10)
    cfg = ArmConfig(model=HAIKU, window=1500, pricing=TABLE, estimator=BYTES4_V0, run_dir=tmp_path, max_calls_question=1, max_calls_episode=3)
    questions = select_questions(rows, episodes=1, per_episode=8, smoke=False)["L"]
    results = run_b4(questions, index, client, cfg, CallLog(tmp_path / "calls.jsonl"))
    assert any(r.bound_hit and r.reason == "bound_hit" for r in results)
    assert sum(r.calls for r in results) <= 3
    lres = run_l(questions, index, client, cfg, CallLog(tmp_path / "calls-l.jsonl"))
    assert sum(r.calls for r in lres) <= 3 and any(r.bound_hit for r in lres)
    guard.close()


def test_guard_refusal_stops_cleanly_and_keeps_partial_results(tmp_path, corpus):
    summary, _ = run(tmp_path, corpus, budget="0.002")
    _, results, calls = load_run(tmp_path)
    assert summary["incomplete"] is True and summary["stopped"]["reason"] == "budget_refused"
    assert results and calls and len(results) < 24
    assert summary["spend"]["spent_pusd"] <= 2 * 10**9


def test_arms_refuse_a_raw_client(tmp_path, corpus):
    manifest, rows = corpus

    class Sentinel:
        def __getattr__(self, name):
            raise AssertionError(f"raw client touched: {name}")

    index = CorpusIndex(manifest["cache_dir"], 10)
    cfg = ArmConfig(model=HAIKU, window=1500, pricing=TABLE, estimator=BYTES4_V0, run_dir=tmp_path)
    log = CallLog(tmp_path / "calls.jsonl")
    for arm in (run_b3, run_b4, run_l):
        with pytest.raises(TypeError, match="GuardedClient"):
            arm(rows[:1], index, Sentinel(), cfg, log)
    with pytest.raises(TypeError):
        B4Episode(index, Sentinel(), cfg, log, 1)
    with pytest.raises(TypeError):
        LEpisode(index, Sentinel(), cfg, log, 1)


def test_sonnet_thinking_blocks_pass_through_b4_and_l(tmp_path, corpus):
    summary, server = run(tmp_path, corpus, arms=("B4", "L"), model="claude-sonnet-5-5", budget="2")
    assert summary["incomplete"] is False and server.errors == []
    assert all(r["thinking"] == {"type": "between_tools"} for r in server.requests if r["system"][0]["text"] != COMPACTION_PROMPT)
    assert any(b["type"] == "thinking" for r in server.requests for m in r["messages"] if m["role"] == "assistant" for b in m["content"])
    for events, blobs in ledgers(tmp_path):
        assert check(events, blobs) == []


def test_pilot_answer_parser():
    parsed = parse_pilot_answer(pilot_answer_text(41, [{"text": "It is 41.", "citations": [{"ref": "@3", "quote": "LIMIT = 41"}]}]))
    assert parsed.answer == 41 and not parsed.abstain and parsed.claims[0].citations[0].ref == "@3"
    assert parse_pilot_answer(pilot_answer_text(None, [], True)).abstain
    for bad in ('<answer>{"claims": [], "abstain": false}</answer>', "no block", '<answer>{"answer": 1, "claims": [], "abstain": "no"}</answer>'):
        with pytest.raises(AnswerFormatError):
            parse_pilot_answer(bad)
    prompts = {arm: system_prompt(arm) for arm in ("B3", "B4", "L")}
    framing = lambda p: p.split("\n\n", 1)[0] + p.rsplit("If you are not sure", 1)[1]  # noqa: E731
    assert len({framing(p) for p in prompts.values()}) == 1  # identical except the tool paragraph


SOURCE = '''def drain_all(self, items, limit=12):
    """Drain."""
    if items is None:
        raise ValueError("no items")

    def inner():
        raise KeyError("nested raises do not count")

    if len(items) > limit:
        raise errors.OverflowError("too many")
    return items
'''


def test_scripted_model_reads_answers_from_verbatim_text():
    piece = {"kind": "span", "ref": "@4", "path": "Lib/spool.py", "body": "SPOOL_LIMIT = 127  # cap\n\n" + SOURCE}
    assert find_answer("constant", {"name": "SPOOL_LIMIT"}, piece) == ("127", "SPOOL_LIMIT = 127  # cap")
    assert find_answer("default", {"name": "drain_all", "param": "limit"}, piece) == ("12", "def drain_all(self, items, limit=12):")
    assert find_answer("raises", {"name": "drain_all"}, piece) == (["OverflowError", "ValueError"], ['raise ValueError("no items")', 'raise errors.OverflowError("too many")'])
    assert find_answer("locate", {"name": "drain_all"}, piece) == ("spool", "def drain_all(self, items, limit=12):")
    assert find_answer("locate", {"name": "missing_name"}, piece) is None
    model = ScriptedPilotModel()
    no_tools = model({"system": [{"type": "text", "text": "x"}], "messages": [{"role": "user", "content": [{"type": "text", "text": "Which module defines the function `nowhere_fn`?"}]}]})
    assert parse_pilot_answer(no_tools["content"][0]["text"]).abstain
    search = model({"system": [{"type": "text", "text": "x"}], "tools": [{"name": "corpus_search"}, {"name": "corpus_read"}],
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Which module defines the function `drain_all`?"}]}]})  # fmt: skip
    assert search["content"][0]["name"] == "corpus_search" and search["stop_reason"] == "tool_use"


def test_prompt_version_is_recorded_and_v1_still_runs(tmp_path, corpus):
    from context_ledger.eval.prompts import PILOT_PROMPT_V1, PILOT_PROMPT_V2

    for version in (PILOT_PROMPT_V2, PILOT_PROMPT_V1):
        out = tmp_path / version
        summary, server = run(out, corpus, prompt_version=version)
        assert summary["incomplete"] is False and summary["prompts"]["version"] == version
        assert summary["prompts"]["sha256"] == {arm: prompt_sha256(arm, version) for arm in ("B3", "B4", "L")}
        systems = {r["system"][0]["text"] for r in server.requests if r["system"][0]["text"] != COMPACTION_PROMPT}
        assert systems == {system_prompt(arm, version) for arm in ("B3", "B4", "L")}
        _, results, _ = load_run(out)
        assert any(r["reason"] == "answered" for r in results) and not any(r["reason"] == "unparseable_answer" for r in results)
        for events, _ in ledgers(out):
            builds = [e.payload["build"]["prompt"] for e in events if e.type == "model_call"]
            assert builds and all(b == {"version": version, "sha256": prompt_sha256("L", version)} for b in builds)
