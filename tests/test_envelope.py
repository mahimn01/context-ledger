"""BUILD-V2 item 4: W counts the whole prompt (envelopes included) minus the arm's tool
definitions, measured the same way in every arm (calib-v3), with the envelope fitted offline
against a fake counter, and a live run refused without a fitted envelope file."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from context_ledger.anthropic.request import build_request
from context_ledger.envelope import (
    OFFLINE_ENVELOPE,
    Envelope,
    EnvelopeError,
    EnvelopeMissing,
    envelope_calibration,
    load_envelope,
    prompt_tokens,
    text_class,
    text_pieces,
)
from context_ledger.eval import cli
from context_ledger.eval.arms import ArmConfig, B4Episode
from context_ledger.eval.calibrate import CalibrationError, FakeEnvelopeCounter, calibrate_envelope, write_envelope
from context_ledger.fold import BackstopPolicyV1, extractive_summary_v1, fold_delta, fold_id_for, plan_fold_aliases, render_fold
from context_ledger.invariants import check
from context_ledger.pricing import PricingTable
from context_ledger.render import rendered_window_tokens, tool_blocks
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.tokens import BYTES4_V0, from_payload, load_calibration, window_estimator
from m2util import MODULE, MODULE_PATH

HAIKU = "claude-haiku-5-5"
TABLE = PricingTable.verified_default()
ENV = Envelope(base=5, message=4, text=3, tool_use=17, tool_result=2, thinking=6)


def v3(env: Envelope = ENV):
    return window_estimator(load_calibration(HAIKU), env)


def test_prompt_tokens_adds_every_envelope():
    request = {
        "system": [{"type": "text", "text": "abcd"}],
        "tools": [{"name": "ignored", "description": "x" * 400, "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "12345678"}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "abcd", "signature": "s"}, {"type": "tool_use", "id": "t1", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "abcd"}, {"type": "text", "text": "abcd"}]}]},
        ],
    }
    # base 5; system text 3+1; message 4 + text 3+2; message 4 + thinking 6+1 + tool_use 17+ceil(len("f {}")/4)=1;
    # message 4 + tool_result 2 + two inner texts (3+1) each
    assert prompt_tokens(request, BYTES4_V0, ENV) == 5 + 4 + (4 + 5) + (4 + 7 + 18) + (4 + 2 + 8)


def test_text_blocks_are_classed_by_their_header():
    assert text_class("[@3 Lib/x.py#f]\ndef f(): pass", "other") == "python"
    assert text_class("[Doc/a.rst:1-5]\ntext", "json") == "prose"
    assert text_class("[fold @A]\n- (@B) line [@3]", "other") == "other"
    assert text_class("read Lib/x.py:1-5 as @3 (f)", "json") == "json"
    context = "<context>\n[Lib/x.py:1-3]\na = 1\n\n[Doc/y.rst:1-2]\nwords\n</context>"
    assert [cls for _, cls in text_pieces(context, "other")] == ["other", "python", "prose"]
    assert "".join(piece for piece, _ in text_pieces(context, "other")) == context


def test_the_estimator_round_trips_through_tokenizer_set():
    estimator = v3()
    assert estimator.name == "calib-v3" and from_payload(estimator.payload()) == estimator
    bad = {**estimator.payload(), "envelope": {**ENV.to_dict(), "text": -1}}
    with pytest.raises(ValueError):
        from_payload(bad)
    assert from_payload(window_estimator(load_calibration(HAIKU)).payload()).name == "calib-v2"  # M4 ledgers unchanged


# The ledger arm: per-segment costs sum to the measure of the request the builder renders.


def session(budget: int = 100_000, env: Envelope = ENV) -> Session:
    estimator = v3(env)
    policy = BackstopPolicyV1(budget_tokens=budget, model=HAIKU, token_counter=estimator.name)
    s = Session(Ledger(":memory:"), MemoryBlobStore(), budget, policy, pricing=TABLE, tokenizer=estimator)
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    return s


def tool_round(s: Session, spans: list, n: int, *, raw: bool = False) -> None:
    tid = f"toolu_{n:03d}"
    s.append_blocks([{"type": "text", "text": "reading"}, {"type": "tool_use", "id": tid, "name": "corpus_read", "input": {"path": MODULE_PATH, "n": n}}])
    for span in spans:
        s.append_span(span.span_id, tool_use_id=tid)
    s.append_raw("tool", f"read {len(spans)} spans", tool_use_id=tid)
    if raw:
        s.append_raw("user", "a follow-up note")


def world(budget: int = 100_000, rounds: int = 4) -> Session:
    s = session(budget)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = s.register_python_spans(src, MODULE_PATH)
    s.append_raw("user", "What is the default timeout?")
    for n in range(rounds):
        tool_round(s, spans[n % len(spans) : n % len(spans) + 2], n, raw=n % 2 == 1)
    return s


def request_of(s: Session) -> dict:
    return build_request(s.state, s.blobs, model=HAIKU, max_tokens=512, tools=[], ttl="5m", pricing=TABLE)[0]


def test_the_ledger_window_measure_is_the_request_measure():
    s = world()
    s.call()
    assert rendered_window_tokens(s.state, s.blobs) == prompt_tokens(request_of(s), s.counter, ENV)
    s.append_raw("assistant", "ANSWER: 37.0")
    s.append_raw("user", "Next: what is MAX_RETRIES?")
    assert rendered_window_tokens(s.state, s.blobs) == prompt_tokens(request_of(s), s.counter, ENV)
    # without an envelope the measure is M4's: text only
    assert rendered_window_tokens(s.state, s.blobs, window_estimator(load_calibration(HAIKU))) < prompt_tokens(request_of(s), s.counter, ENV)


def _runs(s: Session):
    blocks = tool_blocks(s.state, s.blobs)
    n = len(s.state.segments)
    for i in range(n):
        for j in range(i + 1, n + 1):
            segs = s.state.segments[i:j]
            if any(seg.role == "system" for seg in segs) or blocks[i][0] != i or blocks[j - 1][1] != j:
                continue
            yield i, j


def test_the_backstop_delta_equals_the_simulated_fold_delta():
    s = world(rounds=5)
    s.call()
    model = s.payoff_model()
    checked = 0
    for i, j in _runs(s):
        victims = [seg.segment_id for seg in s.state.segments[i:j]]
        sentences = extractive_summary_v1(s.state, victims, s.blobs)
        fold_id = fold_id_for("backstop", i, victims)
        text = render_fold(fold_id, sentences, plan_fold_aliases(s.state.aliases, fold_id, sentences), s.state.aliases)
        assert model.delta(i, j) == fold_delta(s.state, s.state.segments[i:j], text, s.blobs, aliased=True, k=i), (i, j)
        checked += 1
    assert checked > 20


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 3), st.booleans()), min_size=2, max_size=7), st.integers(0, 10_000))
def test_folds_shrink_the_whole_measure_and_check_clean(rounds, pick):
    s = session()
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = s.register_python_spans(src, MODULE_PATH)
    s.append_raw("user", "What is the default timeout?")
    for n, (k, raw) in enumerate(rounds):
        tool_round(s, spans[n % len(spans) : n % len(spans) + k], n, raw=raw)
    s.call()
    runs = [(i, j) for i, j in _runs(s) if s.payoff_model().infeasible(i, j) is None]
    if runs:
        i, j = runs[pick % len(runs)]
        victims = [seg.segment_id for seg in s.state.segments[i:j]]
        before = rendered_window_tokens(s.state, s.blobs)
        try:
            record = s.fold(victims, extractive_summary_v1(s.state, victims, s.blobs), k=i)
        except Exception as exc:  # a summary that would grow the window is refused, never recorded
            assert getattr(exc, "reason", "") == "fold_would_grow_window"
        else:
            assert rendered_window_tokens(s.state, s.blobs) == before - record.freed_tokens
            s.call()
            assert rendered_window_tokens(s.state, s.blobs) == prompt_tokens(request_of(s), s.counter, ENV)
    assert check(s.events, s.blobs) == []


def test_backstop_folds_under_the_envelope_replay_clean():
    s = world(budget=2600, rounds=1)
    src = next(iter(s.state.sources.values()))
    spans = s.register_python_spans(src, MODULE_PATH)
    folds = 0
    for n in range(1, 12):
        s.call()
        folds += sum(e.type == "fold" for e in s.events[-3:])
        tool_round(s, spans[n % len(spans) : n % len(spans) + 2], n, raw=True)
        s.append_raw("assistant", f"noted {n}")
        s.append_raw("user", f"question {n}")
    assert any(e.type == "fold" and e.payload["fold"]["trigger"] == "backstop" for e in s.events)
    assert check(s.events, s.blobs) == []


# B4 and B3 measure with the same function.


def test_b4_measures_its_conversation_with_the_same_envelope(tmp_path):
    cfg = ArmConfig(model=HAIKU, window=6000, pricing=TABLE, estimator=v3(), run_dir=tmp_path)
    ep = B4Episode.__new__(B4Episode)  # only the window measure is exercised
    ep.cfg, ep.system = cfg, cfg.system("B4")
    ep.messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is `m.X`?"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "corpus_read", "input": {"path": "Lib/m.py", "start_line": 1, "end_line": 9}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "[Lib/m.py:1-9]\nX = 3"}]}]},
    ]
    request = {"system": [{"type": "text", "text": ep.system}], "messages": ep.messages}
    assert ep.window_tokens() == prompt_tokens(request, cfg.estimator, ENV)


# The envelope fit (offline, fake counter)


def content():
    return window_estimator(load_calibration(HAIKU))


def test_the_fit_recovers_the_envelope_and_measures_tool_definitions():
    result = calibrate_envelope(FakeEnvelopeCounter(ENV, content()), content(), model=HAIKU, thinking={"type": "disabled"}, created="2026-11-01")
    assert Envelope.from_dict(result["envelope"]) == Envelope(5, 4, 3, 19, 0, 6)  # tool_use + tool_result: only their sum is identifiable
    assert result["max_abs_residual"] == 0 and result["unmeasured"] == []
    assert result["tool_definitions"]["B3"]["tokens"] == 0 and 0 < result["tool_definitions"]["B4"]["tokens"] < result["tool_definitions"]["L"]["tokens"]
    assert {r["kind"] for r in result["requests"]} == {"text", "message", "tool", "thinking"} and len(result["requests"]) == 36
    text = json.dumps(result)
    assert '"x"' not in text and "corpus_read" not in text  # hashes and counts only, no request text


def test_a_kind_the_counter_rejects_is_recorded_unmeasured():
    class NoThinking(FakeEnvelopeCounter):
        def count_request(self, body):
            if any(b.get("type") == "thinking" for m in body["messages"] for b in m["content"]):
                raise RuntimeError("thinking blocks need a valid signature")
            return super().count_request(body)

    result = calibrate_envelope(NoThinking(ENV, content()), content(), model=HAIKU)
    assert result["unmeasured"] == ["thinking"] and result["envelope"]["thinking"] == 0 and result["envelope"]["text"] == 3

    class Broken(FakeEnvelopeCounter):
        def count_request(self, body):
            raise RuntimeError("down")

    with pytest.raises(CalibrationError):
        calibrate_envelope(Broken(ENV, content()), content(), model=HAIKU)


def test_envelope_files_are_separate_never_overwritten_and_fakes_refused(tmp_path):
    result = calibrate_envelope(FakeEnvelopeCounter(ENV, content()), content(), model=HAIKU, created="2026-11-01")
    path = write_envelope(result, tmp_path)
    assert path.name == f"{HAIKU}-envelope-20261101.json"
    with pytest.raises(CalibrationError):
        write_envelope(result, tmp_path)
    with pytest.raises(EnvelopeMissing):
        load_envelope(HAIKU, tmp_path)  # fitted with the fake counter
    fitted = load_envelope(HAIKU, tmp_path, allow_fake=True)
    assert fitted.envelope == Envelope(5, 4, 3, 19, 0, 6) and fitted.calibration_sha256 == content().calibration_sha256
    with pytest.raises(EnvelopeMissing):
        load_envelope(HAIKU, tmp_path / "none")
    with pytest.raises(EnvelopeError):
        envelope_calibration({"version": "other"})
    assert load_calibration(HAIKU).name == "calib-v1"  # the calib-v1 glob never picks up an envelope file


def test_run_estimator_needs_an_envelope_live_and_matches_its_calibration(tmp_path):
    calibration = load_calibration(HAIKU)
    with pytest.raises(EnvelopeMissing):
        cli.run_estimator(HAIKU, True, calibration, tmp_path)
    estimator, defs, warnings = cli.run_estimator(HAIKU, False, calibration, tmp_path)
    assert estimator.name == "calib-v3" and estimator.envelope == OFFLINE_ENVELOPE and defs is None and "default envelope" in warnings[0]
    result = calibrate_envelope(FakeEnvelopeCounter(ENV, content()), content(), model=HAIKU, created="2026-11-01")
    data = {**result, "counter": "count_tokens"}  # as if fitted live
    (tmp_path / f"{HAIKU}-envelope-20261101.json").write_text(json.dumps(data))
    estimator, defs, warnings = cli.run_estimator(HAIKU, True, calibration, tmp_path)
    assert estimator.envelope == Envelope(5, 4, 3, 19, 0, 6) and defs == {arm: d["tokens"] for arm, d in result["tool_definitions"].items()} and not warnings
    (tmp_path / f"{HAIKU}-envelope-20261102.json").write_text(json.dumps({**data, "calibration_sha256": "0" * 64}))
    with pytest.raises(EnvelopeMissing):
        cli.run_estimator(HAIKU, True, calibration, tmp_path)


def test_a_live_run_without_an_envelope_refuses_before_the_gate_or_any_key(tmp_path, monkeypatch, capsys):
    from context_ledger.live.keys import EnvironmentKeyProvider

    def missing(model, directory=None, **kw):
        raise EnvelopeMissing(f"no envelope calibration for {model}")

    monkeypatch.setattr(cli, "load_envelope", missing)
    monkeypatch.setattr(cli, "open_live", lambda *a, **k: pytest.fail("reached the gate"))
    monkeypatch.setattr(EnvironmentKeyProvider, "get", lambda self: pytest.fail("key read"))
    assert cli.main(["--workdir", str(tmp_path), "run", "--live", "--budget", "0.75", "--smoke"]) == 2
    assert "no envelope calibration for claude-haiku-5-5" in capsys.readouterr().err


def test_offline_envelope_calibration_never_writes_calibration(tmp_path, capsys):
    from context_ledger.tokens import CALIBRATION_DIR

    # calibration/ holds Mahimn's live envelope files (2026-11-01): it must be byte-for-byte unchanged
    snapshot = lambda: {p.name: p.read_bytes() for p in CALIBRATION_DIR.glob("*.json")}  # noqa: E731
    before = snapshot()
    assert cli.main(["--workdir", str(tmp_path), "calibrate", "--envelope", "--counter", "fake", "--model", HAIKU]) == 0
    written = list((tmp_path / "runs" / "calibration-fake").glob(f"{HAIKU}-envelope-*.json"))
    assert len(written) == 1 and "fake counter" in capsys.readouterr().out
    assert snapshot() == before


# A run under calib-v3: compaction and folds still fire, and run.json has the pressure block.


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from context_ledger.eval.corpus import build_corpus
    from context_ledger.eval.questions import build_questions
    from pilot_fixture import write_fixture_tarball

    tmp = tmp_path_factory.mktemp("envelope")
    tar, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    return manifest, build_questions(manifest, seed=9, episodes=2, runs_dir=tmp / "runs")


def test_a_calib_v3_run_folds_compacts_and_writes_the_pressure_block(tmp_path, corpus):
    from context_ledger.eval.runner import offline_client, run_pilot

    manifest, rows = corpus
    client, guard, server = offline_client(tmp_path, run_id="v3", pricing=TABLE, budget_usd="0.75")
    try:
        summary = run_pilot(manifest=manifest, rows=rows, arms=["B3", "B4", "L"], rho=10, model=HAIKU, window=2500, episodes=1, per_episode=8,
                            smoke=False, run_dir=tmp_path, run_id="v3", pricing=TABLE, estimator=v3(OFFLINE_ENVELOPE), client=client, live=False,
                            budget_usd="0.75")  # fmt: skip
    finally:
        guard.close()
    b3 = [r for r in server.requests if "No tools:" in r["system"][0]["text"]]  # not B4's summary calls
    assert b3 and all(prompt_tokens(r, v3(OFFLINE_ENVELOPE), OFFLINE_ENVELOPE) <= 2500 - 1024 for r in b3)  # B3 fills W - max_tokens
    assert any("<context>" in json.dumps(r) for r in b3)
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["estimator"]["estimator"] == "calib-v3" and run["tool_definitions"]["source"] == "measured with count_tokens"
    assert run["tool_definitions"]["tokens"]["B3"] == 0 and run["tool_definitions"]["tokens"]["L"] > run["tool_definitions"]["tokens"]["B4"] > 0
    pressure = run["pressure"]
    assert set(pressure) == {"B3", "B4", "L"} and all(p["answer_calls"] > 0 and p["prompt_over_w_p95"] is not None for p in pressure.values())
    assert float(pressure["B4"]["compactions_per_episode"]) >= 1 and float(pressure["L"]["folds_per_episode"]) >= 1
    assert summary["pressure"] == pressure
    from context_ledger.store import SqliteBlobStore

    db = tmp_path / "L" / "episode-01.sqlite"
    events = Ledger(db).events()
    assert check(events, SqliteBlobStore(db)) == []
