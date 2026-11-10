"""BUILD-V4FIX: thinking blocks go back only while what preceded them is unchanged, and a response cut
at max_tokens while holding tool uses leaves no tool use unanswered.

The v4 grid's Sonnet runs stopped in L (46 error rows) and L4 (2) on the API's 400 "messages.N.content.M:
Invalid `signature` in `thinking` block. The block is bound to a different conversation. ... Content
before this block was rewritten since it was created, first at `messages.X.content.Y`." Read-only replays
of the 17 grid ledgers with a refused call (and the Sonnet pilot's one) show one cause in all: L's
backstop folded earlier turns into a fold line between the call that returned a thinking block and
the next call, so the block went back after rewritten content. The same l-v3 code had sent 37 such blocks in the v3 Sonnet grid (2026-11-01) and the
API took every one; from 2026-11-02 it refused all 50 and took the 22 that followed unchanged content.

Two Haiku runs (forced, rho 10000, episodes 1-5 and 6-10) stopped in L for a different reason: a
response cut at max_tokens held 12 ctx_refetch calls, the turn ended with them unanswered, and every
later request of the episode failed InvalidWindow ("message 3: a tool_use is not answered by the next
message"). Those are the only two such responses in every v3 and v4 ledger.

These tests never read runs/: the row below is copied from
runs/v4-grid-sonnet-allowed-r1000-e1-4-c260/results.jsonl (qid v4e01q09, the API's message without its
request id), and its request's shape from that run's L ledger (call 35 of episode 1), texts replaced.
"""

from __future__ import annotations

import json
import re

import pytest

from context_ledger.anthropic import client as client_module
from context_ledger.anthropic.client import LedgerClient, _LedgerClientCore
from context_ledger.anthropic.client_v4 import LedgerClientV4
from context_ledger.anthropic.fake import FakeAnthropicServer, FakeAPIStatusError
from context_ledger.anthropic.request import InvalidWindow, build_request, request_problems, strip_cache_control
from context_ledger.eval import cli
from context_ledger.eval.arms import HARNESS_PATCH, HARNESS_PATCH_V4FIX
from context_ledger.eval.runner import load_run
from context_ledger.invariants import check
from context_ledger.pricing import PricingTable
from context_ledger.store import Ledger, SqliteBlobStore
from context_ledger.window import replay
from context_ledger.prompts import answer_text
from m2util import reply, text, tool_use
from pilot_fixture import v2_fixture_files, write_fixture_tarball
from test_arms import corpus, ledgers, run  # noqa: F401  (corpus is a fixture)
from test_tools import loaded

SONNET, HAIKU = "claude-sonnet-5-5", "claude-haiku-5-5"
BETWEEN = {"type": "between_tools"}
TABLE = PricingTable.verified_default()
REAL_400 = ("messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation. Remove the block, "
            'or set `thinking.block_binding.prefix_mismatch_behavior` to "drop_block". That setting requires the '
            "`thinking-binding-controls-2026-08-01` value in the `anthropic-beta` header. Content before this block was rewritten since it "
            "was created, first at `messages.0.content.4`.")  # fmt: skip
BOUND = re.compile(r"messages\.\d+\.content\.\d+: Invalid `signature` in `thinking` block\. The block is bound to a different conversation\. "
                   r"Remove the block, .* Content before this block was rewritten since it was created, first at `messages\.\d+\.content\.\d+`\.")  # fmt: skip
TOOL = {"name": "corpus_read", "description": "read", "input_schema": {"type": "object", "properties": {}}}


def _use(n: int) -> dict:
    return tool_use(f"toolu_{n}", "corpus_read")


def _result(n: int) -> dict:
    return {"type": "tool_result", "tool_use_id": f"toolu_{n}", "content": [{"type": "text", "text": f"lines of read {n}"}]}


def _body(messages: list[dict]) -> dict:
    return {"model": SONNET, "max_tokens": 1024, "thinking": BETWEEN, "tools": [TOOL], "system": [{"type": "text", "text": "L"}], "messages": messages}


def _live_shape():
    """The stopped call's two requests, shaped as in the ledger: the call that returned the thinking
    block (five messages: four user text blocks, then two tool rounds) and the next one, after the
    backstop folded both rounds into a fold line appended to message 0."""
    head = [text("[fold @9P] earlier context"), text("[fold @9W] earlier context"), text("[@96 note:9#0-59] Update: the shard tag"),
            text("[fold @9T] earlier context")]  # fmt: skip
    created = _body([
        {"role": "user", "content": head},
        {"role": "assistant", "content": [_use(1), _use(2)]},
        {"role": "user", "content": [_result(1), _result(2)]},
        {"role": "assistant", "content": [text("I'll read the two files."), _use(3), _use(4)]},
        {"role": "user", "content": [_result(3), _result(4)]},
    ])  # fmt: skip
    return created, head


def test_the_fake_answers_the_live_rebound_block_with_the_apis_own_400():
    created, head = _live_shape()
    server = FakeAnthropicServer([reply(_use(5)), reply(text("ok")), reply(text("ok"))], pricing=TABLE, emulate=True)
    client = server.as_fake_client()
    thinking, use = client.messages.create(**created)["content"]
    assert thinking["type"] == "thinking"
    after = lambda first, blocks: _body([{"role": "user", "content": first}, {"role": "assistant", "content": blocks},  # noqa: E731
                                         {"role": "user", "content": [_result(5)]}])  # fmt: skip
    folded = [*head, text("[fold @A7] - (@A8) assistant message: I'll read the two files.")]
    with pytest.raises(FakeAPIStatusError) as refused:
        client.messages.create(**after(folded, [thinking, use]))
    assert refused.value.body["error"] == {"type": "invalid_request_error", "message": REAL_400}  # the row's message, word for word
    with pytest.raises(FakeAPIStatusError, match="bound to a different conversation"):
        client.messages.count_tokens(**{k: v for k, v in after(folded, [thinking, use]).items() if k != "max_tokens"})  # as count_tokens was
    client.messages.create(**after(folded, [use]))  # removed, as the 400 says: accepted
    unchanged = {**created, "messages": [*created["messages"], {"role": "assistant", "content": [thinking, use]}, {"role": "user", "content": [_result(5)]}]}
    client.messages.create(**unchanged)  # nothing before it rewritten: accepted, cache_control or not
    assert server.errors == [400]  # (the fake client does not log a count_tokens error)


def unpatched(monkeypatch) -> None:
    """The client as it was before BUILD-V4FIX: no thinking binding is recorded (so the builder never
    leaves a thinking block out) and a cut response's tool uses stay unanswered."""
    monkeypatch.setattr(client_module, "thinking_bindings", lambda request, content: {})
    monkeypatch.setattr(_LedgerClientCore, "_answer_cut_tool_uses", lambda self, step: None)


def test_unpatched_l_on_sonnet_hits_the_live_400(tmp_path, corpus, monkeypatch):  # noqa: F811
    """The fake would have caught it: the offline Sonnet pilot through L (its backstop folds at W =
    1500) gets the API's 400, as the grid did."""
    unpatched(monkeypatch)
    summary, server = run(tmp_path, corpus, arms=("L",), model=SONNET, budget="2")
    _, results, _ = load_run(tmp_path)
    errors = [r["error"]["message"] for r in results if r.get("error")]
    assert errors and all(BOUND.search(m) for m in errors) and 400 in server.errors


def test_patched_l_on_sonnet_leaves_out_exactly_the_rebound_blocks(tmp_path, corpus):  # noqa: F811
    """With the patch the same pilot has no error, every ledger checks clean (P15 rebuilds every
    request, bindings and all), and each request is the unpatched builder's request for the same
    window with exactly the thinking blocks left out whose prefix changed: byte for byte the same
    when none did."""
    summary, server = run(tmp_path, corpus, arms=("L",), model=SONNET, budget="2")
    assert summary["incomplete"] is False and server.errors == []
    dropped = kept = same = 0
    for events, blobs in ledgers(tmp_path):
        assert check(events, blobs) == []
        assert any("thinking_bound" in e.payload for e in events if e.type == "model_call")
        for k, event in enumerate(events):
            if event.type != "model_call":
                continue
            at = replay(events[: next(j for j in range(k, -1, -1) if events[j].type == "call") + 1])
            build = event.payload["build"]
            tools = json.loads(blobs.get(build["tools_hash"]).decode("utf-8"))
            make = lambda state: build_request(state, blobs, model=SONNET, max_tokens=build["max_tokens"], tools=tools, ttl=build["ttl"],  # noqa: E731
                                               pricing=PricingTable.from_dict(state.pricing), frozen=build["frozen"],
                                               prev_last=build["prev_last"], thinking=build["thinking"])[0]  # fmt: skip
            patched = make(at)
            legacy = at.copy()
            legacy.thinking_bound = None
            old = make(legacy)
            if patched == old:
                same += 1
                kept += sum(b["type"] == "thinking" for m in patched["messages"] for b in m["content"])
                continue
            plain = lambda r: [[strip_cache_control(b) for b in m["content"]] for m in r["messages"]]  # noqa: E731
            gone = [[b for b in m if b["type"] == "thinking" and b not in p] for m, p in zip(plain(old), plain(patched))]
            assert plain(patched) == [[b for b in m if b not in g] for m, g in zip(plain(old), gone)] and any(gone)
            assert {k: v for k, v in patched.items() if k != "messages"} == {k: v for k, v in old.items() if k != "messages"}
            dropped += sum(map(len, gone))
    assert dropped and kept and same  # the fixture exercises both: blocks left out, and blocks kept unchanged
    for r in server.requests:  # and the fake (the API's rule) took every one
        assert not request_problems(r)


def _v4_work(work):
    """The v4 pipeline fixture (corpus at rho 10, its address index, questions v3) in `work`, and
    the CLI bound to it."""
    work.mkdir(parents=True)
    tar, sha = write_fixture_tarball(work, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    assert e("index", "build") == 0
    assert e("questions", "build", "--version", "v3", "--episodes", "2", "--questions-per-episode", "12", "--probe-gap", "4",
             "--mix", "anchors=5,probes=3,multihop=2,unanswerable=2,notes=4,supersessions=1,session=3,session_unanswerable=1") == 0  # fmt: skip
    return e


def _cli_digests(work) -> dict[str, str]:
    """Haiku (thinking disabled) through L, L4 and L4-nogate on the v4 pipeline fixture, both
    conditions: every ledger's event count and head hash (each model_call holds its request_hash, so
    equal heads mean byte-identical requests), calls.jsonl, and the results rows without
    harness_patch."""
    e = _v4_work(work)
    out = {}
    for condition in ("allowed", "forced"):
        assert e("run", "--arms", "L,L4,L4-nogate", "--questions", "v3", "--rho", "10", "--episodes", "1-2", "--window", "3000", "--budget", "2",
                 "--run-id", condition, "--condition", condition, "--max-calls-episode", "260") == 0  # fmt: skip
        root = work / "runs" / condition
        out[f"{condition}/calls"] = (root / "calls.jsonl").read_text()
        out[f"{condition}/results"] = [{k: v for k, v in json.loads(line).items() if k != "harness_patch"} for line in (root / "results.jsonl").read_text().splitlines()]
        for db in sorted(root.glob("L*/episode-*.sqlite")):
            ledger = Ledger(db)
            try:
                out[f"{condition}/{db.parent.name}/{db.name}"] = f"{len(ledger.events())}:{ledger.head()}"
            finally:
                ledger.close()
    return out


def test_haiku_requests_are_byte_identical_with_and_without_the_patch(tmp_path, monkeypatch):
    patched = _cli_digests(tmp_path / "patched")
    with monkeypatch.context() as m:
        unpatched(m)
        before = _cli_digests(tmp_path / "unpatched")
    assert patched == before and sum("/L" in k for k in patched) >= 6  # L, L4 and L4-nogate, two episodes, both conditions
    run = json.loads((tmp_path / "patched" / "runs" / "forced" / "run.json").read_text())
    assert run["harness_patch"] == f"{HARNESS_PATCH}+{HARNESS_PATCH_V4FIX}"  # a run with a v4 arm records the patch
    rows = [json.loads(line) for line in (tmp_path / "patched" / "runs" / "forced" / "results.jsonl").read_text().splitlines()]
    assert {r["harness_patch"] for r in rows} == {run["harness_patch"]}


@pytest.mark.parametrize("patched", [False, True])
def test_l4_on_sonnet_stops_on_the_400_unpatched_and_runs_clean_patched(tmp_path, monkeypatch, patched):
    """L4 and L4-nogate share L's request path: unpatched, the forced Sonnet run at W = 3000 stops on
    the API's 400 as the grid's L4 did; patched it completes and every ledger checks clean."""
    if not patched:
        unpatched(monkeypatch)
    e = _v4_work(tmp_path / "w")
    e("run", "--arms", "L4,L4-nogate", "--model", SONNET, "--questions", "v3", "--rho", "10", "--episodes", "1-2", "--window", "3000",
      "--budget", "5", "--run-id", "s", "--condition", "forced", "--max-calls-episode", "260")  # fmt: skip
    root = tmp_path / "w" / "runs" / "s"
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    errors = [r["error"]["message"] for r in rows if r.get("error")]
    if not patched:
        assert len(errors) == 3 and all(BOUND.search(m) for m in errors) and json.loads((root / "run.json").read_text())["stopped"]["reason"] == "api_errors"
        return
    assert errors == [] and len(rows) == 2 * 2 * 16
    dbs = sorted(root.glob("L4*/episode-*.sqlite"))
    assert len(dbs) == 4 and all(check(Ledger(db).events(), SqliteBlobStore(db)) == [] for db in dbs)


# The Haiku stop: a response cut at max_tokens while it holds tool uses


def _cut_then_next(client_cls):
    s = loaded()
    refetch = [tool_use(f"toolu_c{n}", "ctx_search", query=f"batch {n}") for n in range(3)]
    cut = reply(text("I'll refetch them in batches."), *refetch, stop_reason="max_tokens")
    server = FakeAnthropicServer([cut, reply(text(answer_text([], abstain=True)))], pricing=s.pricing)
    client = client_cls(s, server.as_fake_client(), model=HAIKU, max_tokens=512, thinking={"type": "disabled"})
    turn = (lambda q, t: client.run_turn(q, turn_id=t, max_rounds=3)) if client_cls is LedgerClient else (
        lambda q, t: client.run_turn_v4(q, turn_id=t, max_rounds=3, preamble=lambda: None, repair=lambda v: "repair"))  # fmt: skip
    first = turn("What is the default timeout?", "t1")
    assert first.reason == "max_tokens"
    return s, server, turn


@pytest.mark.parametrize("client_cls", [LedgerClient, LedgerClientV4])
def test_a_cut_responses_tool_uses_get_error_results_and_the_next_turn_runs(client_cls):
    s, server, turn = _cut_then_next(client_cls)
    results = [seg for seg in s.state.segments if seg.tool_use_id and seg.tool_use_id.startswith("toolu_c")]
    assert [r.tool_use_id for r in results] == ["toolu_c0", "toolu_c1", "toolu_c2"] and all(r.is_error for r in results)
    assert "stopped (max_tokens) before this tool call could run" in s.blobs.get(results[0].text_hash).decode("utf-8")
    second = turn("What is the retry limit?", "t2")
    assert second.reason != "max_tokens" and len(server.requests) == 2 and not request_problems(server.requests[1])
    assert check(s.events, s.blobs) == []


def test_unpatched_the_next_turn_after_a_cut_response_is_not_a_request(monkeypatch):
    unpatched(monkeypatch)
    _, server, turn = _cut_then_next(LedgerClient)
    with pytest.raises(InvalidWindow, match="a tool_use is not answered by the next message"):
        turn("What is the retry limit?", "t2")  # as every later question of the two stopped Haiku episodes
    assert len(server.requests) == 1
