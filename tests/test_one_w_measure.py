"""BUILD-V2b item 1: one W measure for L. The backstop judges the window on the request about to
be sent as count_tokens counts it minus L's tool definitions, so at every call its measure is
the predicted prompt minus the tool definitions; the estimator's own measure already equals the
calib-v3 measure of that request, and the pilot's gap was the estimator against count_tokens."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from context_ledger.anthropic.client import LedgerClient
from context_ledger.anthropic.fake import FakeAnthropicServer
from context_ledger.anthropic.request import build_request
from context_ledger.envelope import OFFLINE_ENVELOPE, prompt_tokens
from context_ledger.fold import BackstopPolicyV1
from context_ledger.invariants import check
from context_ledger.model import Event
from context_ledger.pricing import PricingTable
from context_ledger.prompts import SYSTEM_PROMPT, answer_text
from context_ledger.render import rendered_window_tokens
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore, SqliteBlobStore
from context_ledger.store.blobs import _verified
from context_ledger.tokens import envelope_of, estimator_for, load_calibration, window_estimator
from context_ledger.window import apply_event, replay
from m2util import MODULE, MODULE_PATH, index_of, mutate, props, reply, text, tool_use

HAIKU = "claude-haiku-5-5"
TABLE = PricingTable.verified_default()
ROOT = Path(__file__).resolve().parents[1]


def v3():
    return window_estimator(load_calibration(HAIKU), OFFLINE_ENVELOPE)


@pytest.fixture(scope="module")
def l_run(tmp_path_factory):
    """One offline L episode on the fixture corpus under calib-v3, W = 2500: several corpus_read
    tool turns, and backstop folds."""
    from context_ledger.eval.corpus import build_corpus
    from context_ledger.eval.questions import build_questions
    from context_ledger.eval.runner import offline_client, run_pilot
    from pilot_fixture import write_fixture_tarball

    tmp = tmp_path_factory.mktemp("one-w")
    tar, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    rows = build_questions(manifest, seed=9, episodes=2, runs_dir=tmp / "runs")
    run_dir = tmp / "run"
    client, guard, _ = offline_client(run_dir, run_id="w", pricing=TABLE, budget_usd="0.75")
    try:
        run_pilot(manifest=manifest, rows=rows, arms=["L"], rho=10, model=HAIKU, window=2500, episodes=1, per_episode=8, smoke=False,
                  run_dir=run_dir, run_id="w", pricing=TABLE, estimator=v3(), client=client, live=False, budget_usd="0.75")  # fmt: skip
    finally:
        guard.close()
    db = run_dir / "L" / "episode-01.sqlite"
    return run_dir, Ledger(db).events(), SqliteBlobStore(db)


def test_at_every_l_call_the_backstop_window_is_the_counted_prompt_minus_tool_definitions(l_run):
    run_dir, events, blobs = l_run
    run = json.loads((run_dir / "run.json").read_text())
    tools = run["tool_definitions"]["tokens"]["L"]
    calls = [e.payload for e in events if e.type == "model_call"]
    reads = sum(b.get("name") == "corpus_read" for e in events if e.type == "segment_appended" and e.payload["segment"].get("content_type") == "blocks"
                for b in json.loads(blobs.get(e.payload["segment"]["text_hash"])))  # fmt: skip
    assert reads >= 3 and len(calls) >= 6
    assert all(c["predicted"]["window"] == c["predicted"]["prompt"] - tools for c in calls)
    logged = [c for c in (json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines()) if c["arm"] == "L"]
    assert [c["window"] for c in logged] == [c["predicted"]["window"] for c in calls]
    folds = [e.payload["fold"] for e in events if e.type == "fold" and e.payload["fold"]["trigger"] == "backstop"]
    assert folds and all(isinstance(f["policy"]["measured"], int) for f in folds)
    assert check(events, blobs) == []  # P9 re-runs every fold on its recorded measured window
    # The estimator's own measure is not the counted one: held to it, L would not be held to W.
    state, estimates = replay([]), []
    for event in events:
        if event.type == "call":
            estimates.append(rendered_window_tokens(state, blobs))
        apply_event(state, event)
    assert any(est != c["predicted"]["window"] for est, c in zip(estimates, calls))


def _request_at(state, blobs, model_call: dict) -> dict:
    build = model_call["build"]
    return build_request(state, blobs, model=model_call["model"], max_tokens=build["max_tokens"], tools=json.loads(blobs.get(build["tools_hash"])),
                         ttl=build["ttl"], pricing=PricingTable.from_dict(state.pricing), frozen=build["frozen"], prev_last=build["prev_last"],
                         thinking=build.get("thinking"))[0]  # fmt: skip


def test_the_estimators_measure_is_the_calib_v3_measure_of_the_request_sent(l_run):
    """No piece of the request is missing from L's measure: system prompt, tool-loop envelopes,
    the current question, alias headers and fold summaries are all counted, exactly."""
    _, events, blobs = l_run
    state, checked = replay([]), 0
    for n, event in enumerate(events):
        if event.type == "call":
            model_call = next(e.payload for e in events[n:] if e.type == "model_call")
            request = _request_at(state, blobs, model_call)
            assert rendered_window_tokens(state, blobs) == prompt_tokens(request, estimator_for(state), envelope_of(estimator_for(state)))
            checked += 1
        apply_event(state, event)
    assert checked >= 6


class _ReadOnlyBlobs:
    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con

    def get(self, h: str) -> bytes:
        row = self.con.execute("SELECT data FROM blobs WHERE hash = ?", (h,)).fetchone()
        return _verified(h, None if row is None else bytes(row[0]))

    def has(self, h: str) -> bool:
        return self.con.execute("SELECT 1 FROM blobs WHERE hash = ?", (h,)).fetchone() is not None


PILOT_LEDGER = ROOT / "runs" / "pilot-v2a-haiku" / "rho-10" / "L" / "episode-16.sqlite"


@pytest.mark.skipif(not PILOT_LEDGER.exists(), reason="the v2a pilot artifacts are not in this checkout")
def test_the_pilot_gap_is_the_estimator_against_count_tokens_read_only():
    """runs/pilot-v2a-haiku rho 10, episode 16, the call after the fold at seq 259 (BUILD-V2b's
    table): the backstop's measure, 3359, is exactly the calib-v3 measure of the request sent;
    count_tokens counted 6546, minus L's 1428 tool definitions 5118."""
    digest = hashlib.sha256(PILOT_LEDGER.read_bytes()).hexdigest()
    siblings = sorted(p.name for p in PILOT_LEDGER.parent.iterdir())
    con = sqlite3.connect(f"file:{PILOT_LEDGER}?mode=ro&immutable=1", uri=True)
    try:
        blobs = _ReadOnlyBlobs(con)
        rows = con.execute("SELECT seq, type, payload, prev_hash, hash FROM events WHERE seq <= 263 ORDER BY seq").fetchall()
        events = [Event(seq, t, json.loads(p), a, b) for seq, t, p, a, b in rows]
        state = replay(events[:259])  # every event before the call at seq 260
        model_call = events[262].payload
        measure = rendered_window_tokens(state, blobs)
        request = _request_at(state, blobs, model_call)
        estimator = estimator_for(state)
        assert measure == prompt_tokens(request, estimator, envelope_of(estimator)) == 3359
        assert model_call["predicted"]["prompt"] - 1428 == 5118
    finally:
        con.close()
    assert hashlib.sha256(PILOT_LEDGER.read_bytes()).hexdigest() == digest and sorted(p.name for p in PILOT_LEDGER.parent.iterdir()) == siblings


# The measured window inside the backstop


def session(budget: int) -> Session:
    estimator = v3()
    policy = BackstopPolicyV1(budget_tokens=budget, model=HAIKU, token_counter=estimator.name)
    s = Session(Ledger(":memory:"), MemoryBlobStore(), budget, policy, pricing=TABLE, tokenizer=estimator)
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    return s


def filled(budget: int) -> Session:
    """Five tool rounds of two spans each, then a call, then a follow-up question."""
    s = session(budget)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = s.register_python_spans(src, MODULE_PATH)
    s.append_raw("user", "What is the default timeout?")
    for n in range(5):
        tid = f"toolu_{n:03d}"
        s.append_blocks([{"type": "text", "text": "reading"}, {"type": "tool_use", "id": tid, "name": "corpus_read", "input": {"n": n}}])
        for span in spans[2 * n : 2 * n + 2]:
            s.append_span(span.span_id, tool_use_id=tid)
        s.append_raw("tool", "read 2 spans", tool_use_id=tid)
    s.call()
    s.append_raw("assistant", "ANSWER: 37.0")
    s.append_raw("user", "Next: what is MAX_RETRIES?")
    return s


def test_measured_equal_to_the_estimate_decides_as_before_and_records_it():
    s = filled(1120)  # mandatory by the estimator after the follow-up
    L = s.payoff_model().L
    plain, measured = s.payoff_model().analysis(), s.payoff_model(L).analysis()
    assert plain.decision is not None and plain.window == plain.used == L and measured.window == L
    assert measured.decision == replace(plain.decision, measured=L) and measured.runs == plain.runs
    assert "measured" not in plain.decision.record(s.policy, s.state.pricing_hash)
    assert measured.decision.record(s.policy, s.state.pricing_hash)["measured"] == L


def test_the_zone_follows_the_measured_window_and_p9_replays_it():
    budget = 4000
    s = filled(budget)
    L = s.payoff_model().L
    assert L * 10000 <= 7000 * budget and s.payoff_model().analysis().decision is None  # normal by the estimator: no fold
    counted = budget  # count_tokens says the window is at 100% of W: mandatory
    analysis = s.payoff_model(counted).analysis()
    assert analysis.zone == "mandatory" and analysis.window == counted
    pool = [r for r in analysis.runs if counted * (L - r.delta) * 10000 <= 8500 * budget * L]  # reaches hard, scaled per estimated token
    assert pool and (analysis.decision.i, analysis.decision.j) in {(r.i, r.j) for r in pool}
    # a counter reading the window at budget / L counted tokens per estimated token
    record = s.prepare_call(lambda: rendered_window_tokens(s.state, s.blobs) * budget // L)
    assert record is not None and record.policy["measured"] == counted
    assert s.last_window == rendered_window_tokens(s.state, s.blobs) * budget // L <= 8500 * budget // 10000  # the call goes out at or below hard
    assert check(s.events, s.blobs) == []
    k = index_of(s.events, "fold", lambda p: p["fold"]["trigger"] == "backstop")
    assert "P9" in props(mutate(s.events, k, lambda p: p["fold"]["policy"].pop("measured")), s.blobs)
    assert "P9" in props(mutate(s.events, k, lambda p: p["fold"]["policy"].update(measured=L)), s.blobs)


def test_a_count_that_fails_falls_back_to_the_estimator_with_a_warning(caplog):
    s = session(100_000)
    s.append_raw("system", SYSTEM_PROMPT, pinned=True)
    server = FakeAnthropicServer([reply(text(answer_text([], abstain=True)))], pricing=TABLE)

    def broken(**body):
        raise RuntimeError("count_tokens is down")

    fake = server.as_fake_client()
    client = SimpleNamespace(messages=SimpleNamespace(create=fake.messages.create, count_tokens=broken))
    wrapper = LedgerClient(s, client, model=HAIKU, max_tokens=256, tools=[], predict=True, tool_tokens=100)
    with caplog.at_level(logging.WARNING, logger="context_ledger.client"):
        turn = wrapper.run_turn("What is the default timeout?", turn_id="q1")
    assert turn.abstained and "count_tokens failed" in caplog.text and "count_tokens is down" not in caplog.text
    model_call = next(e.payload for e in s.events if e.type == "model_call")
    assert model_call["predicted"]["window"] == s.last_window and s.last_backstop.measured is None


def test_a_working_count_is_exactly_the_window_less_tool_tokens():
    s = session(100_000)
    s.append_raw("system", SYSTEM_PROMPT, pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    alias = s.assign_alias(spans["DEFAULT_TIMEOUT"].span_id)

    def lookup(session, name, input, *, turn_id, tool_use_id=None):
        from context_ledger.tools import ToolOutcome

        session.refetch(session.state.aliases.resolve(input["ref"]), tool_use_id=tool_use_id)
        return ToolOutcome("ok")

    tool = {"name": "lookup", "description": "Bring a span back.", "input_schema": {"type": "object", "properties": {"ref": {"type": "string"}}}}
    script = [reply(tool_use("toolu_01", "lookup", ref=alias)),
              reply(text(answer_text([{"text": "The timeout is 37.", "citations": [{"ref": alias, "quote": "DEFAULT_TIMEOUT = 37.0"}]}])))]  # fmt: skip
    server = FakeAnthropicServer(script, pricing=TABLE)
    wrapper = LedgerClient(s, server.as_fake_client(), model=HAIKU, max_tokens=256, tools=[tool], dispatch=lookup, predict=True, tool_tokens=7)
    wrapper.run_turn("What is the default timeout?", turn_id="q1")
    calls = [e.payload for e in s.events if e.type == "model_call"]
    counted = [server.count_tokens({k: v for k, v in r.items() if k != "max_tokens"})["input_tokens"] for r in server.requests]
    assert len(calls) == 2 and [c["predicted"]["window"] for c in calls] == [n - 7 for n in counted]
    assert check(s.events, s.blobs) == []
