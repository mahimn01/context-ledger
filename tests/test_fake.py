"""D.5: the FakeAnthropicServer's prompt-cache simulation and request assertions."""

from __future__ import annotations

import pytest

from context_ledger.anthropic.fake import FakeAnthropicServer
from context_ledger.anthropic.request import block_tokens
from context_ledger.pricing import PricingTable
from m2util import pricing, reply, text

EPHEMERAL = {"type": "ephemeral"}


def table(min_tokens: int = 1) -> PricingTable:
    d = pricing().to_dict()
    for model in d["models"].values():
        model["min_cache_tokens"] = min_tokens
    return PricingTable.from_dict(d)


def request(*texts: str, marks=(), ttl=None, model="claude-haiku-5-5") -> dict:
    """One user message whose k-th text block carries a cache marker if k is in `marks`."""
    marker = {"type": "ephemeral", "ttl": ttl} if ttl else EPHEMERAL
    blocks = [{"type": "text", "text": t, **({"cache_control": marker} if k in marks else {})} for k, t in enumerate(texts)]
    return {"model": model, "max_tokens": 64, "system": [], "tools": [], "messages": [{"role": "user", "content": blocks}]}


def sizes(body: dict) -> list[int]:
    return [block_tokens(b) for b in body["messages"][0]["content"]]


def test_write_then_read_then_partial_read():
    server = FakeAnthropicServer([reply(text("ok"))] * 3, pricing=table())
    a = request("alpha " * 40, "beta " * 40, "gamma " * 40, marks=(1,))
    first = server.handle(a)["usage"]
    t = sizes(a)
    assert first["cache_read_input_tokens"] == 0 and first["cache_creation_input_tokens"] == t[0] + t[1]
    assert first["input_tokens"] == t[2] and first["cache_creation"] == {"ephemeral_5m_input_tokens": t[0] + t[1], "ephemeral_1h_input_tokens": 0}
    b = request("alpha " * 40, "beta " * 40, "gamma " * 40, "delta " * 40, marks=(1, 3))
    second = server.handle(b)["usage"]
    t = sizes(b)
    assert second["cache_read_input_tokens"] == t[0] + t[1]
    assert second["cache_creation_input_tokens"] == t[2] + t[3] and second["input_tokens"] == 0
    c = request("alpha " * 40, "CHANGED " * 30, "gamma " * 40, marks=(2,))
    third = server.handle(c)["usage"]
    assert third["cache_read_input_tokens"] == 0  # a changed block breaks every later prefix


def test_moving_a_marker_does_not_change_a_prefix():
    server = FakeAnthropicServer([reply(text("ok"))] * 2, pricing=table())
    server.handle(request("one " * 30, "two " * 30, marks=(1,)))
    usage = server.handle(request("one " * 30, "two " * 30, "three " * 30, marks=(1, 2)))["usage"]
    assert usage["cache_read_input_tokens"] == sum(sizes(request("one " * 30, "two " * 30)))


def test_min_cache_tokens_and_ttl_in_calls():
    small = FakeAnthropicServer([reply(text("ok"))] * 2, pricing=table(10_000))
    body = request("tiny", marks=(0,))
    assert small.handle(body)["usage"]["cache_creation_input_tokens"] == 0
    assert small.handle(body)["usage"]["cache_read_input_tokens"] == 0
    short = FakeAnthropicServer([reply(text("ok"))] * 3, pricing=table(), ttl_calls=1)
    body = request("words " * 50, marks=(0,))
    short.handle(body)
    assert short.handle(body)["usage"]["cache_read_input_tokens"] > 0  # live for the next call
    short.handle(request("other " * 50, marks=(0,)))
    fresh = FakeAnthropicServer([reply(text("ok"))] * 3, pricing=table(), ttl_calls=1)
    fresh.handle(body)
    fresh.handle(request("other " * 50, marks=(0,)))
    assert fresh.handle(body)["usage"]["cache_read_input_tokens"] == 0  # expired after one idle call


def test_one_hour_writes_are_reported_separately():
    server = FakeAnthropicServer([reply(text("ok"))], pricing=table())
    usage = server.handle(request("x " * 100, marks=(0,), ttl="1h"))["usage"]
    assert usage["cache_creation"] == {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": usage["cache_creation_input_tokens"]}


@pytest.mark.parametrize(
    "tamper",
    [
        lambda b: b.update(temperature=0),
        lambda b: b.update(top_p=1),
        lambda b: b.update(top_k=5),
        lambda b: b.update(tools=[{"name": "ctx.fold", "description": "x", "input_schema": {"type": "object"}}]),
        lambda b: [blk.update(cache_control=EPHEMERAL) for blk in b["messages"][0]["content"]],
    ],
    ids=["temperature", "top_p", "top_k", "dotted tool name", "five breakpoints"],
)
def test_server_rejects_invalid_requests(tamper):
    server = FakeAnthropicServer([reply(text("ok"))], pricing=table())
    body = request(*["block"] * 5)
    tamper(body)
    with pytest.raises(AssertionError):
        server.handle(body)
    assert server.requests == []


def test_script_order_ids_and_exhaustion():
    server = FakeAnthropicServer([reply(text("a")), reply(text("b"), stop_reason="max_tokens")], pricing=table())
    one, two = server.handle(request("q")), server.handle(request("q"))
    assert (one["id"], one["content"][0]["text"], one["stop_reason"]) == ("msg_fake_0000", "a", "end_turn")
    assert (two["id"], two["stop_reason"], two["model"]) == ("msg_fake_0001", "max_tokens", "claude-haiku-5-5")
    with pytest.raises(AssertionError):
        server.handle(request("q"))
