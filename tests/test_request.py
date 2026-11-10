"""D.2 and D.3: tool-use segments and the cache-aware request builder; P15."""

from __future__ import annotations

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from context_ledger.anthropic.request import (
    FORBIDDEN_PARAMS,
    InvalidWindow,
    build_request,
    linearize,
    request_problems,
    strip_cache_control,
)
from context_ledger.fold import BackstopPolicyV1, FoldError, SummarySentence
from context_ledger.ids import canonical_json, sha256_hex
from context_ledger.invariants import check
from context_ledger.model import Citation
from context_ledger.pricing import PricingTable
from context_ledger.render import rendered_window_tokens
from context_ledger.session import BudgetExceeded, SessionError
from m2util import MODULE, MODULE_PATH, index_of, mutate, pricing, props, v1_session

TOOLS = [
    {"name": "lookup", "description": "Look a symbol up.", "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"], "additionalProperties": False}},
    {"name": "audit", "description": "Audit the window.", "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


def low_min_cache_pricing(min_tokens: int = 1) -> PricingTable:
    d = pricing().to_dict()
    for model in d["models"].values():
        model["min_cache_tokens"] = min_tokens
    return PricingTable.from_dict(d)


def build(s, tools=TOOLS, **kw):
    return build_request(
        s.state, s.blobs, model=s.policy.model, max_tokens=256, tools=tools, ttl=s.policy.ttl, pricing=s.pricing,
        frozen=kw.get("frozen", s.first_candidate()), prev_last=kw.get("prev_last", s.state.model_last),
    )  # fmt: skip


def model_step(s, blocks, tools=TOOLS):
    """What LedgerClient.step does around the network call, with a scripted response."""
    s.prepare_call()
    params = {
        "max_tokens": 256,
        "ttl": s.policy.ttl,
        "tool_names": sorted(t["name"] for t in tools),
        "tools_hash": s.blobs.put(canonical_json(sorted(tools, key=lambda t: t["name"]))),
        "frozen": s.first_candidate(),
        "prev_last": s.state.model_last,
    }
    request, meta = build(s, tools, frozen=params["frozen"], prev_last=params["prev_last"])
    call = s.record_call(request)
    s.append_blocks(blocks)
    usage = dict.fromkeys(["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens", "cache_creation_5m", "cache_creation_1h"], 0)
    s.record_model_call(
        call.call_id, model=s.policy.model, request=request, build=params, breakpoints=meta.breakpoints, usage=usage,
        usage_raw={}, stop_reason="end_turn", response_id="msg_test", response_hash=sha256_hex(canonical_json(blocks)), cost_pusd=0,
    )  # fmt: skip
    return request, meta


@pytest.fixture
def s():
    s = v1_session(10**6)
    s.set_pricing(low_min_cache_pricing())
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    s.spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    return s


def test_merging_and_tool_result_placement(s):
    s.append_raw("user", "What is the default timeout?")
    s.append_span(s.spans["DEFAULT_TIMEOUT"].span_id, role="user")
    model_step(s, [{"type": "text", "text": "Let me look."}, {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "timeout"}}])
    s.refetch(s.spans["DEFAULT_TIMEOUT"].span_id, tool_use_id="toolu_1")
    s.append_raw("tool", "lookup: 1 hit", tool_use_id="toolu_1", is_error=True)
    request, meta = build(s)
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert [b["type"] for b in request["messages"][0]["content"]] == ["text", "text"]  # merged
    (result,) = request["messages"][2]["content"]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1" and result["is_error"] is True
    assert [c["text"].split("\n")[-1] for c in result["content"]] == ["DEFAULT_TIMEOUT = 37.0", "lookup: 1 hit"]
    assert request["system"][0]["text"] == "Answer only from cited spans."
    assert [t["name"] for t in request["tools"]] == ["audit", "lookup"]
    assert not request_problems(request) and not FORBIDDEN_PARAMS & set(request)
    assert meta.paths[-1] == meta.paths[-2] == (("messages", 2, 0),)
    assert check(s.events, s.blobs) == []


def test_invalid_windows(s):
    s.append_blocks([{"type": "text", "text": "I speak first."}])
    with pytest.raises(InvalidWindow, match="first message"):
        build(s)
    t = v1_session()
    t.append_raw("user", "hi")
    t.append_raw("tool", "orphan", tool_use_id="toolu_9")
    with pytest.raises(InvalidWindow, match="answers no tool_use"):
        build(t)
    u = v1_session()
    u.append_raw("user", "hi")
    u.append_blocks([{"type": "tool_use", "id": "toolu_2", "name": "audit", "input": {}}])
    u.append_raw("user", "interrupting")
    u.append_raw("tool", "late", tool_use_id="toolu_2")
    with pytest.raises(InvalidWindow, match="must come first"):
        build(u)
    pending = v1_session()
    pending.append_raw("user", "hi")
    pending.append_blocks([{"type": "tool_use", "id": "toolu_3", "name": "audit", "input": {}}])
    request, _ = build(pending)
    assert request["messages"][-1]["role"] == "assistant"  # a pending tool_use is allowed only at the very end


def test_folds_and_evicts_keep_tool_pairs_whole(s):
    s.append_raw("user", "q")
    model_step(s, [{"type": "tool_use", "id": "toolu_1", "name": "audit", "input": {}}])
    s.append_raw("tool", "x" * 400, tool_use_id="toolu_1")
    s.call()
    use = s.state.segments[-2]
    with pytest.raises(FoldError) as info:
        s.fold([use.segment_id], [SummarySentence("used a tool", ())])
    assert info.value.reason == "tool_pair"
    with pytest.raises(SessionError):
        s.evict([s.state.segments[-1].segment_id], "x")
    model = s.payoff_model()
    blocks = {model.blocks[p] for p in range(len(s.state.segments))}
    assert (len(s.state.segments) - 2, len(s.state.segments)) in blocks
    s.fold([use.segment_id, s.state.segments[-1].segment_id], [SummarySentence("used a tool", ())])  # the whole pair is fine


def test_breakpoint_priority_and_cap(s):
    for name in ["retrying", "backoff_delay", "InventoryClient.__init__", "DEFAULT_TIMEOUT"]:
        s.append_span(s.spans[name].span_id, role="user")
        s.call()
    retrying = s.spans["retrying"].span_id
    s.claim("retrying wraps fn", [Citation(retrying, "def retrying(fn):")])  # cited: not foldable, so BP2 moves past it
    s.append_raw("user", "question one")
    model_step(s, [{"type": "text", "text": "answer one"}])
    s.append_raw("user", "question two")
    request, meta = build(s)
    names = [b["bp"] for b in meta.breakpoints]
    assert names == ["BP1", "BP2", "BP3", "BP4"]
    marked = [block for _, block in linearize(request) if "cache_control" in block]
    assert len(marked) == 4 and all(b["cache_control"] == {"type": "ephemeral"} for b in marked)
    blocks = [b["block"] for b in meta.breakpoints]
    assert blocks == sorted(blocks) and [b["prefix_tokens"] for b in meta.breakpoints] == sorted(b["prefix_tokens"] for b in meta.breakpoints)
    assert meta.breakpoints[0]["position"] == 0  # the system segment
    assert meta.breakpoints[2]["position"] == s.state.segments.index(next(g for g in s.state.segments if g.segment_id == s.state.model_last))


def test_min_cache_tokens_skips_short_prefixes(s):
    s.set_pricing(pricing())  # Haiku: 4096-token minimum, far above this window
    s.append_raw("user", "short")
    request, meta = build(s)
    assert meta.breakpoints == () and not any("cache_control" in b for _, b in linearize(request))
    s.set_pricing(low_min_cache_pricing(40))
    request, meta = build(s)
    assert all(b["prefix_tokens"] >= 40 for b in meta.breakpoints) and meta.breakpoints


def test_ttl_1h_marker():
    t = v1_session(10**6, ttl="1h")
    t.set_pricing(low_min_cache_pricing())
    t.append_raw("system", "s", pinned=True)
    t.append_raw("user", "hello")
    request, _ = build(t)
    assert request["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_tools_order_is_byte_stable(s):
    s.append_raw("user", "hi")
    a, _ = build(s, TOOLS)
    b, _ = build(s, list(reversed(TOOLS)))
    assert canonical_json(a) == canonical_json(b)


def test_p15_clean_ledger_and_tampers(s):
    s.append_raw("user", "What is the default timeout?")
    model_step(s, [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "timeout"}}])
    s.refetch(s.spans["DEFAULT_TIMEOUT"].span_id, tool_use_id="toolu_1")
    model_step(s, [{"type": "text", "text": "It is 37 seconds."}])
    events = s.events
    assert check(events, s.blobs) == []
    i = index_of(events, "model_call")
    assert props(mutate(events, i, lambda p: p.update(request_hash="0" * 64)), s.blobs) == {"P15"}
    j = index_of(events, "segment_appended", lambda p: p["segment"].get("tool_use_id") == "toolu_1")
    assert props(mutate(events, j, lambda p: p["segment"].update(tool_use_id="toolu_orphan")), s.blobs) == {"P15"}
    k = index_of(events, "model_call", last=True)
    fifth = {"bp": "BP5", "position": 1, "block": 3, "prefix_tokens": 9}
    assert props(mutate(events, k, lambda p: p["breakpoints"].append(fifth)), s.blobs) == {"P15"}


@settings(max_examples=40, deadline=None)
@given(sizes=st.lists(st.integers(30, 300), min_size=4, max_size=10), percent=st.integers(96, 200), raw_every=st.integers(2, 4))
def test_prefix_bytes_before_a_backstop_fold_are_unchanged(sizes, percent, raw_every):
    """For a backstop-v1 fold at [i, j), the request blocks of tools, system and every position
    before i are byte-identical before and after the fold (cache markers aside)."""
    s = v1_session(10**7)
    s.set_pricing(low_min_cache_pricing())
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    units = [f"unit {k}\n" + "w" * n + "\n" for k, n in enumerate(sizes)]
    src = s.register_source("doc.txt", "".join(units))
    start = 0
    for k, unit in enumerate(units):
        if k % raw_every == 0:
            s.append_raw("user", f"question {k}")
        s.append_span(s.register_text_span(src, start, start + len(unit) - 1).span_id, role="user")
        start += len(unit)
        s.call()
    before, before_meta = build(s)
    s.policy = BackstopPolicyV1(budget_tokens=rendered_window_tokens(s.state, s.blobs) * 100 // percent)
    s.budget_tokens = s.policy.budget_tokens
    try:  # BUILD-V2b: several folds per call; the window may stay mandatory after them
        s.prepare_call()
    except BudgetExceeded:
        pass
    event("folded" if s.last_folds else "no fold")
    if not s.last_folds:
        return
    k = min(f.k for f in s.last_folds)
    after, after_meta = build(s)

    def prefix(request, meta, i):
        head = [canonical_json(strip_cache_control(b)) for b in request["tools"] + request["system"]]
        return head + [canonical_json(strip_cache_control(_at(request, path))) for own in meta.paths[:i] for path in own]

    assert prefix(before, before_meta, k) == prefix(after, after_meta, k)


def _at(request, path):
    return request["messages"][path[1]]["content"][path[2]] if path[0] == "messages" else request[path[0]][path[1]]
