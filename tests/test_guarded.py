"""B.2 to B.3: the GuardedClient, thinking configuration and the fake server's count_tokens and
model emulation. Everything offline: the fake server, through the fake client or the real SDK
over httpx2.MockTransport."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from context_ledger.anthropic.client import LedgerClient
from context_ledger.anthropic.fake import COUNT_OVERHEAD, TOOL_SYSTEM_PROMPT_TOKENS, FakeAPIStatusError, FakeAnthropicServer, FakeTimeout
from context_ledger.anthropic.request import PILOT_THINKING, InvalidWindow, block_tokens, linearize
from context_ledger.fold import BackstopPolicyV1
from context_ledger.invariants import check
from context_ledger.live.entry import open_live
from context_ledger.live.gate import LiveRefused
from context_ledger.live.guarded import GuardedClient, ttl_of
from context_ledger.live.keys import FakeKeyProvider
from context_ledger.live.redact import REDACTED, forget_secrets, redact
from context_ledger.live.spend import BudgetRefused, OpenReservation, SpendGuard, prompt_upper_fallback, read_spend
from context_ledger.pricing import VERIFIED_PRICING_PATH, PricingTable
from context_ledger.prompts import SYSTEM_PROMPT, answer_text
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from m2util import MODULE, MODULE_PATH, reply, sdk_client, text, tool_use
from test_client import LOOKUP, lookup

TABLE = PricingTable.load(VERIFIED_PRICING_PATH, live=True)
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"


def body(model=HAIKU, content="hello there", **extra) -> dict:
    return {"model": model, "max_tokens": 64, "system": [], "tools": [], "messages": [{"role": "user", "content": [{"type": "text", "text": content}]}], **extra}


def guard(tmp_path, **kw) -> SpendGuard:
    return SpendGuard(tmp_path / "spend.jsonl", run_id="t", pricing=TABLE, **kw)


def kinds(g: SpendGuard) -> list[str]:
    return [r["kind"] for r in read_spend(g.path)]


def test_every_create_is_reserved_before_the_network_call(tmp_path):
    g = guard(tmp_path)
    seen = []

    def on_request(path, request_body):
        if path == "/v1/messages":
            last = read_spend(g.path)[-1]  # read back from disk: durable before the call
            seen.append(last["kind"])
            assert last["kind"] == "reserve" and last["outcome"] == "reserved"

    server = FakeAnthropicServer([reply(text("a")), reply(text("b"))], pricing=TABLE, emulate=True, on_request=on_request)
    client = GuardedClient(server.as_fake_client(), g)
    for _ in range(2):
        client.messages.create(**body())
    assert seen == ["reserve", "reserve"] and kinds(g) == ["reserve", "settle", "reserve", "settle"]
    assert client.last_prompt_count == sum(block_tokens(b) for _, b in linearize(body())) + COUNT_OVERHEAD
    assert server.counted[0] == {k: v for k, v in body().items() if k != "max_tokens"}  # identical body, no max_tokens


def test_settled_cost_is_the_api_usage(tmp_path):
    g = guard(tmp_path)
    server = FakeAnthropicServer([reply(text("ok"))], pricing=TABLE, emulate=True)
    response = GuardedClient(server.as_fake_client(), g).messages.create(**body())
    usage = response["usage"]
    settle = read_spend(g.path)[-1]
    expected = TABLE.cost_pusd(HAIKU, {**{k: usage[k] for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}})
    assert settle["actual_pusd"] == expected and settle["actual_pusd"] <= settle["worst_case_pusd"]
    assert "output_tokens_details" in usage and usage["output_tokens_details"]["thinking_tokens"] == 0


@pytest.mark.parametrize("through", ["fake client", "sdk"])
def test_void_on_4xx(tmp_path, through):
    g = guard(tmp_path)
    server = FakeAnthropicServer([lambda b: {"__error__": 400, "error": {"type": "invalid_request_error", "message": "bad"}}], pricing=TABLE)
    raw = server.as_fake_client() if through == "fake client" else sdk_client(server)
    client = GuardedClient(raw, g, sleep=lambda s: pytest.fail("a 400 is not retried"))
    with pytest.raises(Exception) as info:
        client.messages.create(**body())
    assert getattr(info.value, "status_code", None) == 400
    assert kinds(g) == ["reserve", "void"] and read_spend(g.path)[-1]["outcome"] == "http_400" and g.spent == 0
    assert client.errors[0]["status"] == 400 and client.errors[0]["error_type"] == "invalid_request_error"


@pytest.mark.parametrize("through", ["fake client", "sdk"])
def test_assume_spent_on_timeout(tmp_path, through):
    def timeout(b):
        raise FakeTimeout()

    g = guard(tmp_path)
    server = FakeAnthropicServer([timeout], pricing=TABLE)
    raw = server.as_fake_client() if through == "fake client" else sdk_client(server)
    with pytest.raises(Exception):
        GuardedClient(raw, g, max_retries=0).messages.create(**body())
    records = read_spend(g.path)
    assert [r["kind"] for r in records] == ["reserve", "assume_spent"] and g.spent == records[0]["worst_case_pusd"]


def test_retries_are_new_reservations_with_backoff(tmp_path):
    g = guard(tmp_path)
    overloaded = lambda b: {"__error__": 529, "error": {"type": "overloaded_error", "message": "busy"}}  # noqa: E731
    server = FakeAnthropicServer([overloaded, overloaded, reply(text("ok"))], pricing=TABLE)
    slept = []
    GuardedClient(server.as_fake_client(), g, sleep=slept.append).messages.create(**body())
    assert kinds(g) == ["reserve", "void", "reserve", "void", "reserve", "settle"] and slept == [2.0, 8.0]
    server = FakeAnthropicServer([overloaded] * 3, pricing=TABLE)
    with pytest.raises(FakeAPIStatusError):
        GuardedClient(server.as_fake_client(), guard(tmp_path / "b"), sleep=lambda s: None).messages.create(**body())
    assert len(server.requests) == 3  # one call plus at most two retries


def test_counting_failure_falls_back_to_the_conservative_bound(tmp_path):
    g = guard(tmp_path)
    server = FakeAnthropicServer([reply(text("ok"))], pricing=TABLE)
    fake = server.as_fake_client()

    def broken(**kw):
        raise FakeAPIStatusError(500, {"type": "error", "error": {"type": "api_error", "message": "count failed"}})

    raw = SimpleNamespace(messages=SimpleNamespace(create=fake.messages.create, count_tokens=broken))
    client = GuardedClient(raw, g)
    client.messages.create(**body())
    assert read_spend(g.path)[0]["prompt_tokens_upper"] == prompt_upper_fallback(body()) and client.last_prompt_count is None


def test_a_refused_reservation_makes_no_call(tmp_path):
    g = guard(tmp_path, run_cap_usd="0.000001")
    server = FakeAnthropicServer([reply(text("ok"))], pricing=TABLE)
    with pytest.raises(BudgetRefused):
        GuardedClient(server.as_fake_client(), g).messages.create(**body(content="x" * 40_000))
    assert server.requests == [] and kinds(g) == []


def test_raw_client_is_only_touched_through_the_guard(tmp_path):
    g = guard(tmp_path)

    class Sentinel:
        """Fails on anything but messages.create and count_tokens, and on a create without an
        open reservation."""

        def __getattr__(self, name):
            if name != "messages":
                raise AssertionError(f"raw client touched: {name}")
            return SimpleNamespace(create=self.create, count_tokens=lambda **kw: {"input_tokens": 10})

        def create(self, **kw):
            assert g.outstanding > 0, "create without a reservation"
            return {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 10, "output_tokens": 1}, "stop_reason": "end_turn"}

    client = GuardedClient(Sentinel(), g)
    client.messages.create(**body())
    assert not hasattr(client, "beta") and sorted(n for n in vars(type(client.messages)) if not n.startswith("_")) == ["count_tokens", "create"]


def test_ttl_of():
    assert ttl_of(body()) == "5m"
    assert ttl_of({"messages": [{"content": [{"cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]}) == "1h"


def test_fake_count_tokens_endpoint_through_the_sdk():
    server = FakeAnthropicServer([], pricing=TABLE, emulate=True)
    tools = [LOOKUP]
    counted = sdk_client(server).messages.count_tokens(model=HAIKU, messages=body()["messages"], tools=tools)
    expected = sum(block_tokens(b) for _, b in linearize({"tools": tools, "messages": body()["messages"]}))
    assert counted.input_tokens == expected + TOOL_SYSTEM_PROMPT_TOKENS + COUNT_OVERHEAD


def test_fake_emulates_each_models_thinking_rules():
    server = FakeAnthropicServer([reply(text("ok"))] * 3, pricing=TABLE, emulate=True)
    client = server.as_fake_client()
    client.messages.create(**body(HAIKU, thinking={"type": "disabled"}))
    with pytest.raises(FakeAPIStatusError) as info:
        client.messages.create(**body(SONNET, thinking={"type": "disabled"}))
    assert info.value.status_code == 400 and "not supported" in str(info.value)
    client.messages.create(**body(SONNET, thinking={"type": "between_tools"}))
    with pytest.raises(FakeAPIStatusError):
        client.messages.create(**body(HAIKU, thinking={"type": "between_tools"}))
    assert PILOT_THINKING == {HAIKU: {"type": "disabled"}, SONNET: {"type": "between_tools"}}


def setup(model: str):
    s = Session(Ledger(":memory:"), MemoryBlobStore(), 100_000, BackstopPolicyV1(budget_tokens=100_000, model=model), pricing=TABLE)
    s.append_raw("system", SYSTEM_PROMPT, pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    alias = s.assign_alias(spans["DEFAULT_TIMEOUT"].span_id)
    return s, alias


@pytest.mark.parametrize("model", [HAIKU, SONNET])
def test_thinking_passthrough_and_predictions_in_a_tool_loop(tmp_path, model):
    s, alias = setup(model)
    script = [
        reply(text("Looking."), tool_use("toolu_01", "lookup", ref=alias)),
        reply(tool_use("toolu_02", "lookup", ref=alias)),
        reply(text(answer_text([{"text": "The timeout is 37.", "citations": [{"ref": alias, "quote": "DEFAULT_TIMEOUT = 37.0"}]}]))),
    ]
    server = FakeAnthropicServer(script, pricing=TABLE, emulate=True)
    g = guard(tmp_path)
    client = GuardedClient(sdk_client(server), g)
    wrapper = LedgerClient(s, client, model=model, max_tokens=512, tools=[LOOKUP], dispatch=lookup, thinking=PILOT_THINKING[model], predict=True)
    background = "Background notes on the inventory client. " * 60  # enough for a cacheable prefix
    result = wrapper.run_turn(background + "What is the timeout?", turn_id="t1")
    assert result.reason == "answered" and check(s.events, s.blobs) == []  # P15 rebuilds with the thinking config
    assert all(r["thinking"] == PILOT_THINKING[model] for r in server.requests)
    calls = [e.payload for e in s.events if e.type == "model_call"]
    assert all(c["build"]["thinking"] == PILOT_THINKING[model] for c in calls)
    thinking_blocks = [b for c in server.requests for m in c["messages"] if m["role"] == "assistant" for b in m["content"] if b["type"] == "thinking"]
    assert bool(thinking_blocks) == (model == SONNET)  # Sonnet's progress updates came back unchanged
    for c, counted in zip(calls, server.counted):
        expected = sum(block_tokens(b) for _, b in linearize(counted)) + TOOL_SYSTEM_PROMPT_TOKENS + COUNT_OVERHEAD
        assert c["predicted"]["prompt"] == expected
    assert calls[0]["predicted"]["cache_read"] == 0 and calls[1]["predicted"]["cache_read"] > 0
    assert [c["predicted"]["cache_read"] for c in calls] == [c["usage"]["cache_read_input_tokens"] for c in calls]
    assert sum(r["actual_pusd"] for r in read_spend(g.path) if r["kind"] == "settle") == sum(c["cost_pusd"] for c in calls)


def test_m2_clients_record_no_predictions():
    s, alias = setup(HAIKU)
    server = FakeAnthropicServer([reply(text(answer_text([], abstain=True)))], pricing=TABLE)
    LedgerClient(s, server.as_fake_client(), model=HAIKU, max_tokens=64, tools=[LOOKUP], dispatch=lookup).run_turn("q", turn_id="t")
    call = next(e.payload for e in s.events if e.type == "model_call")
    assert "predicted" not in call and "thinking" not in call["build"] and "thinking" not in server.requests[0]


def test_builder_never_sends_context_management_or_tiers():
    s, _ = setup(HAIKU)
    s.append_raw("user", "hi")
    from context_ledger.anthropic.request import NEVER_SENT, build_request, request_problems

    request, _ = build_request(s.state, s.blobs, model=HAIKU, max_tokens=64, tools=[], ttl="5m", pricing=TABLE)
    assert not NEVER_SENT & set(request)
    for key in ("context_management", "inference_geo", "service_tier", "temperature"):
        assert request_problems({**request, key: 1})


class Args(SimpleNamespace):
    live = True


def test_open_live_checks_everything_before_the_key(tmp_path):
    keys = FakeKeyProvider("plain-secret-value-0123456789", profile="aperture")  # BUILD-KEY: its ledger is spend/spend.jsonl
    built = []
    factory = lambda secret: built.append(secret) or FakeAnthropicServer([], pricing=TABLE).as_fake_client()  # noqa: E731
    common = dict(keys=keys, pricing=TABLE, run_id="r", run_cap_usd="0.75", spend_dir=tmp_path / "spend", factory=factory)
    with pytest.raises(LiveRefused, match="--live"):
        open_live(SimpleNamespace(live=False), {}, **common)
    with pytest.raises(LiveRefused, match="CLAUDECODE"):
        open_live(Args(), {"CLAUDECODE": "1"}, **common)
    unverified = PricingTable.from_dict({**TABLE.to_dict(), "verified": False})
    with pytest.raises(LiveRefused, match="not verified"):
        open_live(Args(), {}, **{**common, "pricing": unverified})
    assert keys.calls == 0 and built == []
    with SpendGuard(tmp_path / "spend" / "spend.jsonl", run_id="crashed", pricing=TABLE) as g:
        g.reserve(model=HAIKU, request_sha256="0" * 64, prompt_tokens_upper=10, max_tokens=1)
    with pytest.raises(OpenReservation):
        open_live(Args(), {}, **common)
    assert keys.calls == 0 and built == []
    with SpendGuard(tmp_path / "spend" / "spend.jsonl", run_id="fix", pricing=TABLE, resolve=True) as g:
        g.resolve_assume_spent()
    try:
        ctx = open_live(Args(), {}, **common)
        assert keys.calls == 1 and built == [ctx.secret] and isinstance(ctx.client, GuardedClient)
        assert redact("a plain-secret-value-0123456789 b") == f"a {REDACTED} b"  # registered for redaction
        ctx.close()
    finally:
        forget_secrets()


def test_fake_takes_sonnets_thinking_block_back_unchanged_or_dropped_never_altered():
    """BUILD-V4FIX: the API's 400 for a rebound block says to remove it, so a dropped block is
    accepted (M3's fake refused one); an altered one is refused as an invalid signature."""
    server = FakeAnthropicServer([reply(tool_use("toolu_9", "lookup", ref="@0")), reply(text("ok")), reply(text("ok"))], pricing=TABLE, emulate=True)
    client = server.as_fake_client()
    first = client.messages.create(**body(SONNET, thinking={"type": "between_tools"}, tools=[LOOKUP]))
    thinking, use = first["content"]
    assert thinking["type"] == "thinking" and thinking["signature"] and first["usage"]["output_tokens_details"]["thinking_tokens"] > 0
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_9", "content": [{"type": "text", "text": "x"}]}]}
    base = body(SONNET, thinking={"type": "between_tools"}, tools=[LOOKUP])
    altered = {**base, "messages": [*base["messages"], {"role": "assistant", "content": [{**thinking, "thinking": "edited"}, use]}, result]}
    with pytest.raises(FakeAPIStatusError, match=r"messages\.1\.content\.0: Invalid `signature` in `thinking` block\.'"):
        client.messages.create(**altered)
    dropped = {**base, "messages": [*base["messages"], {"role": "assistant", "content": [use]}, result]}
    client.messages.create(**dropped)
    kept = {**base, "messages": [*base["messages"], {"role": "assistant", "content": [thinking, use]}, result]}
    client.messages.create(**kept)
