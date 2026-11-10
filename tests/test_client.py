"""D.4 to D.6: LedgerClient and AsyncLedgerClient through the real SDK, all offline."""

from __future__ import annotations

import asyncio
import json

import pytest

from context_ledger.anthropic.client import AsyncLedgerClient, LedgerClient
from context_ledger.anthropic.fake import FakeAnthropicServer
from context_ledger.anthropic.request import FORBIDDEN_PARAMS
from context_ledger.cost import cost_report
from context_ledger.invariants import check
from context_ledger.prompts import PROMPT_SHA256, SYSTEM_PROMPT, answer_text
from context_ledger.store import rechain
from context_ledger.tools import ToolOutcome
from m2util import MODULE, MODULE_PATH, index_of, mutate, pricing, props, reply, sdk_client, text, tool_use, v1_session

LOOKUP = {
    "name": "lookup",
    "description": "Bring a span back verbatim by its alias.",
    "input_schema": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"], "additionalProperties": False},
}


def lookup(session, name, input, *, turn_id, tool_use_id=None):
    """A minimal tool dispatcher for these tests (the ctx tools have their own tests)."""
    if name != "lookup":
        return ToolOutcome(f"error: unknown_tool {name}", True)
    status = session.refetch(session.state.aliases.resolve(input["ref"]), tool_use_id=tool_use_id).status
    return ToolOutcome(f"{input['ref']} {status}")


def setup(budget=100_000, **policy):
    s = v1_session(budget, **policy)
    s.append_raw("system", SYSTEM_PROMPT, pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    for name in ["DEFAULT_TIMEOUT", "MAX_RETRIES"]:
        s.assign_alias(spans[name].span_id)
    return s, spans


def two_tool_rounds(s, spans):
    timeout, retries = (s.state.aliases.alias_of(spans[n].span_id) for n in ["DEFAULT_TIMEOUT", "MAX_RETRIES"])
    return [
        reply(text("Looking it up."), tool_use("toolu_01", "lookup", ref=timeout)),
        reply(tool_use("toolu_02", "lookup", ref=retries)),
        reply(text(answer_text([
            {"text": "The default timeout is 37 seconds.", "citations": [{"ref": timeout, "quote": "DEFAULT_TIMEOUT = 37.0"}]},
            {"text": "It retries 4 times.", "citations": [{"ref": retries, "quote": "MAX_RETRIES = 4"}]},
        ]))),
    ]  # fmt: skip


def run(asynchronous: bool, script_for=two_tool_rounds, **policy):
    s, spans = setup(**policy)
    server = FakeAnthropicServer(script_for(s, spans), pricing=s.pricing)
    client = sdk_client(server, asynchronous)
    if asynchronous:
        wrapper = AsyncLedgerClient(s, client, model="claude-haiku-5-5", max_tokens=512, tools=[LOOKUP], dispatch=lookup)
        result = asyncio.run(wrapper.run_turn("What are the timeout and the retry count?", turn_id="t1"))
    else:
        wrapper = LedgerClient(s, client, model="claude-haiku-5-5", max_tokens=512, tools=[LOOKUP], dispatch=lookup)
        result = wrapper.run_turn("What are the timeout and the retry count?", turn_id="t1")
    return s, server, result


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_full_turn_with_two_tool_rounds(asynchronous):
    s, server, result = run(asynchronous)
    assert result.reason == "answered" and not result.abstained and len(result.steps) == 3
    assert [st.handling for st in result.steps] == ["tool_use", "tool_use", "end_turn"]
    assert [c.status for c in result.claims] == ["grounded", "grounded"] and len(result.emitted) == 2
    assert check(s.events, s.blobs) == []
    types = [e.type for e in s.events]
    assert types.count("call") == types.count("model_call") == 3 and types[-1] == "answer_emitted"
    # Each tool_result follows its tool_use, and the requests carried them that way.
    second = server.requests[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result" and second[-1]["content"][0]["tool_use_id"] == "toolu_01"
    assert not any(FORBIDDEN_PARAMS & set(r) for r in server.requests)
    assert server.authorized == 3


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_usage_and_cost_are_recorded_exactly(asynchronous):
    s, server, result = run(asynchronous)
    table = pricing()
    calls = [e.payload for e in s.events if e.type == "model_call"]
    for payload, step in zip(calls, result.steps):
        assert payload["usage"] == step.usage and payload["cost_pusd"] == step.cost_pusd
        assert payload["cost_pusd"] == table.cost_pusd("claude-haiku-5-5", payload["usage"])
        assert payload["build"]["prompt"]["sha256"] == PROMPT_SHA256
    assert result.cost_pusd == sum(p["cost_pusd"] for p in calls) > 0
    assert result.usage["input_tokens"] == sum(p["usage"]["input_tokens"] for p in calls)
    report = cost_report(s.events)
    assert report.totals["cost_pusd"] == result.cost_pusd and [r.call_id for r in report.rows] == [st.call_id for st in result.steps]
    assert json.loads(json.dumps(report.to_dict())) == report.to_dict()


def test_the_key_never_reaches_the_ledger():
    s, server, result = run(False)
    assert server.authorized == 3
    payloads = json.dumps([e.payload for e in s.events])
    stored = b"".join(s.blobs.data.values())
    for secret in ("test-key-not-real", "Bearer", "authorization"):
        assert secret not in payloads and secret.encode() not in stored


def test_sync_and_async_ledgers_are_byte_identical():
    sync, _, _ = run(False)
    asyn, _, _ = run(True)
    assert sync.ledger.head() == asyn.ledger.head()
    assert [e.hash for e in sync.events] == [e.hash for e in asyn.events]


def test_budget_exceeded_forces_an_abstention():
    s, spans = setup(budget=200)
    for name in ["retrying", "backoff_delay", "InventoryClient"]:
        s.append_span(spans[name].span_id)  # nothing called yet: all of it is recent
    server = FakeAnthropicServer([], pricing=s.pricing)
    result = LedgerClient(s, server.as_fake_client(), model="claude-haiku-5-5", max_tokens=64, tools=[]).run_turn("Anything?", turn_id="t1")
    assert result.abstained and result.reason == "budget_exceeded" and result.steps == ()
    assert s.events[-1].type == "answer_emitted" and s.events[-1].payload["abstained"] is True
    assert not server.requests and check(s.events, s.blobs) == []


def retries_answer(s, spans) -> str:
    alias = s.state.aliases.alias_of(spans["MAX_RETRIES"].span_id)
    return answer_text([{"text": "MAX_RETRIES is 4.", "citations": [{"ref": alias, "quote": "MAX_RETRIES = 4"}]}])


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", "answered"),
        ("stop_sequence", "answered"),
        ("max_tokens", "max_tokens"),
        ("refusal", "refusal"),
        ("model_context_window_exceeded", "max_tokens"),
        ("a_future_reason", "answered"),
    ],
)
def test_stop_reasons(stop_reason, expected):
    def script(s, spans):
        s.append_span(spans["MAX_RETRIES"].span_id)
        return [reply(text(retries_answer(s, spans)), stop_reason=stop_reason)]

    s, server, result = run(False, script)
    assert result.reason == expected and result.abstained == (expected != "answered")
    assert s.events[index_of(s.events, "model_call")].payload["stop_reason"] == stop_reason
    assert check(s.events, s.blobs) == []


def test_tool_use_and_pause_turn_continue_the_loop():
    def script(s, spans):
        s.append_span(spans["MAX_RETRIES"].span_id)
        alias = s.state.aliases.alias_of(spans["MAX_RETRIES"].span_id)
        return [reply(text("thinking..."), stop_reason="pause_turn"), reply(tool_use("toolu_1", "lookup", ref=alias)), reply(text(retries_answer(s, spans)))]

    s, server, result = run(False, script)
    assert [st.handling for st in result.steps] == ["pause_turn", "tool_use", "end_turn"] and result.reason == "answered"
    assert check(s.events, s.blobs) == []


def test_unparseable_answer_asks_once_then_abstains():
    def script(s, spans):
        return [reply(text("The answer is 4.")), reply(text("<answer>{not json}</answer>"))]

    s, server, result = run(False, script)
    assert result.reason == "unparseable_answer" and result.abstained
    flagged = [c for c in result.claims if c.status == "ungrounded"]
    assert [c.verdict["reason"] for c in flagged] == ["unparseable_answer", "unparseable_answer"]
    assert "could not be parsed" in server.requests[1]["messages"][-1]["content"][-1]["text"]
    assert check(s.events, s.blobs) == []


def test_rounds_exhausted_abstains_after_answering_every_tool_use():
    def script(s, spans):
        return [reply(tool_use(f"toolu_{n}", "lookup", ref="@1")) for n in range(3)]

    s2, spans = setup()
    server2 = FakeAnthropicServer(script(s2, spans), pricing=s2.pricing)
    result = LedgerClient(s2, server2.as_fake_client(), model="claude-haiku-5-5", max_tokens=64, tools=[LOOKUP], dispatch=lookup).run_turn(
        "Loop forever?", turn_id="t9", max_rounds=3
    )
    assert result.reason == "rounds_exhausted" and len(result.steps) == 3
    assert s2.state.segments[-1].tool_use_id == "toolu_2"  # the last tool use was still answered
    assert check(s2.events, s2.blobs) == []


# P15 and P16 tamper tests on a ledger written by the real client


@pytest.fixture(scope="module")
def client_ledger():
    s, server, result = run(False)
    assert check(s.events, s.blobs) == []
    return s.events, s.blobs


def test_p15_tampered_request_hash(client_ledger):
    events, blobs = client_ledger
    assert props(mutate(events, index_of(events, "model_call"), lambda p: p.update(request_hash="f" * 64)), blobs) == {"P15"}


def test_p15_orphan_tool_result(client_ledger):
    events, blobs = client_ledger
    i = index_of(events, "segment_appended", lambda p: p["segment"].get("tool_use_id") == "toolu_02")
    assert props(mutate(events, i, lambda p: p["segment"].update(tool_use_id="toolu_99")), blobs) == {"P15"}


def test_p15_fifth_breakpoint(client_ledger):
    events, blobs = client_ledger
    extra = {"bp": "BP5", "position": 0, "block": 1, "prefix_tokens": 1}
    assert props(mutate(events, index_of(events, "model_call", last=True), lambda p: p["breakpoints"].append(extra)), blobs) == {"P15"}


def test_p16_cost_off_by_one_pico_usd(client_ledger):
    events, blobs = client_ledger
    assert props(mutate(events, index_of(events, "model_call"), lambda p: p.update(cost_pusd=p["cost_pusd"] + 1)), blobs) == {"P16"}


def test_p16_negative_usage(client_ledger):
    events, blobs = client_ledger
    i = index_of(events, "model_call")
    assert props(mutate(events, i, lambda p: p["usage"].update(output_tokens=-1)), blobs) == {"P16"}


def test_p16_missing_pricing_set(client_ledger):
    """Without a table in force no call can be priced (P16), and no request can be rebuilt
    either, because the model's min_cache_tokens decides the breakpoints (P15)."""
    events, blobs = client_ledger
    i = index_of(events, "pricing_set")
    assert props(rechain(events[:i] + events[i + 1 :]), blobs) == {"P16", "P15"}


def test_p16_model_call_not_after_its_call(client_ledger):
    events, blobs = client_ledger
    i = index_of(events, "model_call")
    j = next(k for k in range(i, len(events)) if events[k].type == "call")  # the next call
    moved = events[:i] + events[i + 1 : j + 1] + [events[i]] + events[j + 1 :]
    assert "P16" in props(rechain(moved), blobs)
