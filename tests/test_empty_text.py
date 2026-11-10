"""BUILD-V3FIX: no request carries an empty text block, with or without cache_control.

Three Haiku runs of the PREREG-v3 grid stopped in L on repeated 400s: first "messages.N.content.0.text:
cache_control cannot be set for empty text blocks", then "messages: text content blocks must be
non-empty". The empty block was not the model's: L appended an empty user message itself. emit_answer
refused an answer because an earlier claim of the turn still needed a re-fetch (the answer's claims had
superseded other open claims), and the refetch message listed only the answer's own claims, so it was
"". Now:

- the client writes a line for that earlier claim instead (_refetch_message);
- the request builder drops every empty text block of a message, keeping "(empty)" for a message left
  with nothing, and a breakpoint meant for a dropped segment goes on the nearest earlier block;
- request_problems flags an empty text block, and the fake server answers one with the API's own 400.

The stopped windows are frozen in v3fix_stopped_windows.py; these tests never read runs/.
"""

from __future__ import annotations

import anthropic
import pytest

from context_ledger.anthropic import request as builder
from context_ledger.anthropic.client import LedgerClient, _LedgerClientCore
from context_ledger.anthropic.fake import FakeAnthropicServer, FakeAPIStatusError
from context_ledger.anthropic.request import EMPTY_TEXT_PLACEHOLDER, build_request, empty_text_problems, linearize, request_problems
from context_ledger.eval.prompts import prompt_sha256, system_prompt
from context_ledger.eval.tools import CORPUS_TOOLS
from context_ledger.invariants import check
from context_ledger.pricing import VERIFIED_PRICING_PATH, PricingTable
from context_ledger.prompts import answer_text
from context_ledger.render import empty_text_block
from context_ledger.tools import CTX_TOOLS
from m2util import pricing, reply, sdk_client, text, tool_use, v1_session
from test_request import build, low_min_cache_pricing
from test_tools import alias, loaded, run
from v3fix_stopped_windows import CASES

HAIKU = "claude-haiku-5-5"
CACHE = {"type": "ephemeral"}
GRID_PRICING = PricingTable.load(VERIFIED_PRICING_PATH, live=True)  # the table the v3 grid ran with: Haiku caches from 512 tokens
assert GRID_PRICING.pricing_hash() == "a6d3867dcfe43a9767e93f5fd2bdfdb6470bf9426dea9334d3c5f7feb5dbbbfc"  # the stopped runs' run.json


def unpatched(monkeypatch) -> None:
    """The request builder and client as they were before BUILD-V3FIX: empty text blocks are sent
    (and not flagged by the builder's own check), and the refetch message lists only the answer's
    claims. The fake server keeps checking as the API does."""
    monkeypatch.setattr(builder, "_drop_empty_text", lambda messages, paths: paths)
    real = builder.request_problems
    monkeypatch.setattr(builder, "request_problems", lambda r: [p for p in real(r) if p not in empty_text_problems(r)])

    def old_refetch_message(self, turn_id, claims, exc):
        return "\n".join(
            f"Verification: claim {n} relies on {' '.join(self.session.assign_alias(o) for o in c.verdict['obligations'])} which you no "
            f"longer hold verbatim ({c.verdict['reason']}). Call ctx_refetch on them and answer again."
            for n, c in enumerate(claims, start=1)
            if c.claim_id in exc.claim_ids
        )

    monkeypatch.setattr(_LedgerClientCore, "_refetch_message", old_refetch_message)


def empty_blocks(request: dict) -> list[dict]:
    return [b for _, b in linearize(request) if empty_text_block(b)]


# The root cause: an earlier claim's refetch obligation


def folded_turn():
    """A session whose backoff_delay span is held only behind a fold sentence, and the answers of
    the failing turn: one with an ungrounded claim and a claim on the sentence (needs_refetch), then
    one whose single grounded claim supersedes the ungrounded one, leaving the needs_refetch one open."""
    s = loaded()
    s.set_pricing(GRID_PRICING)
    run(s, "ctx_fold", targets=[alias(s, "retrying"), alias(s, "backoff_delay")],
        summary=[{"text": "retry decorator; backoff_delay caps the wait", "cites": [alias(s, "backoff_delay")]}])  # fmt: skip
    fold = next(iter(s.state.folds.values()))
    sentence = s.state.aliases.alias_of(f"fold:{fold.fold_id}#0")
    timeout = {"text": "The default timeout is 37 seconds.", "citations": [{"ref": alias(s, "DEFAULT_TIMEOUT"), "quote": "DEFAULT_TIMEOUT = 37.0"}]}
    first = answer_text([
        {"text": "The default timeout is 99 seconds.", "citations": [{"ref": alias(s, "DEFAULT_TIMEOUT"), "quote": "DEFAULT_TIMEOUT = 99.0"}]},
        {"text": "The backoff wait is capped at 600 seconds.", "citations": [{"ref": sentence, "quote": "backoff_delay caps the wait"}]},
    ])  # fmt: skip
    return s, sentence, first, answer_text([timeout])


def test_an_earlier_claims_refetch_obligation_gets_a_message_not_an_empty_one():
    s, sentence, first, second = folded_turn()
    backoff = alias(s, "backoff_delay")
    final = answer_text([{"text": "The backoff wait is capped at 300 seconds.", "citations": [{"ref": backoff, "quote": "return min(2 ** attempt, 300)"}]},
                         {"text": "The default timeout is 37 seconds.", "citations": [{"ref": alias(s, "DEFAULT_TIMEOUT"), "quote": "DEFAULT_TIMEOUT = 37.0"}]}])  # fmt: skip
    seen = []

    def after_earlier_claim_message(request):
        seen.append(request["messages"][-1]["content"][-1]["text"])
        return {"content": [tool_use("toolu_r", "ctx_refetch", refs=[sentence])], "stop_reason": "tool_use"}

    server = FakeAnthropicServer([reply(text(first)), reply(text(second)), after_earlier_claim_message, reply(text(final))], pricing=s.pricing)
    result = LedgerClient(s, sdk_client(server), model=HAIKU, max_tokens=512).run_turn("What are the timeout and the backoff cap?", turn_id="t1")
    ungrounded, needs_refetch, grounded = result.claims[:3]
    assert (ungrounded.status, needs_refetch.status, grounded.status, grounded.supersedes) == ("ungrounded", "needs_refetch", "grounded", ungrounded.claim_id)
    (message,) = seen  # the message after the second answer: before the fix, ""
    assert message == (f'Verification: an earlier claim of this turn still fails (summary_only): "{needs_refetch.text}". It relies on {backoff}; call '
                       "ctx_refetch on them if you no longer hold them verbatim. Only a claim that passes clears it: answer again with one passing "
                       "claim in its place; or abstain.")  # fmt: skip
    assert result.reason == "answered" and result.refetch_routes == 2 and server.errors == []
    assert all(not empty_blocks(r) and not request_problems(r) for r in server.requests)
    assert check(s.events, s.blobs) == []


def test_the_fake_would_have_caught_the_empty_refetch_message(monkeypatch):
    """The same turn on the unpatched harness: the fake answers the request carrying the empty
    message with the API's 400, through the real SDK, as the live runs got it."""
    unpatched(monkeypatch)
    s, _, first, second = folded_turn()
    server = FakeAnthropicServer([reply(text(first)), reply(text(second))], pricing=s.pricing)
    with pytest.raises(anthropic.BadRequestError, match="cache_control cannot be set for empty text blocks") as caught:
        LedgerClient(s, sdk_client(server), model=HAIKU, max_tokens=512).run_turn("What are the timeout and the backoff cap?", turn_id="t1")
    assert caught.value.status_code == 400 and server.errors == [400] and len(server.requests) == 2  # the refused request used no scripted reply


def test_forced_earlier_claim_message_never_offers_to_abstain():
    s, _, first, second = folded_turn()
    seen = []
    server = FakeAnthropicServer([reply(text(first)), reply(text(second)), lambda r: seen.append(r) or reply(text(answer_text([], abstain=True)))(r)],
                                 responder=reply(text(answer_text([], abstain=True))), pricing=s.pricing)  # fmt: skip
    LedgerClient(s, sdk_client(server), model=HAIKU, max_tokens=512, forced=True).run_turn("What are the timeout and the backoff cap?", turn_id="t1", max_rounds=3)
    message = seen[0]["messages"][-1]["content"][-1]["text"]
    assert message.startswith("Verification: an earlier claim of this turn still fails (summary_only)") and message.endswith("claim in its place.")


# (a) The request builder never sends an empty text block


def test_an_empty_text_beside_a_tool_use_is_not_replayed():
    """A model turn of text "" plus a tool_use: the ledger keeps it verbatim, the next request
    replays only the tool_use, and the ledger still rebuilds every request (P15)."""
    s = loaded()
    answer = answer_text([{"text": "The default timeout is 37 seconds.", "citations": [{"ref": alias(s, "DEFAULT_TIMEOUT"), "quote": "DEFAULT_TIMEOUT = 37.0"}]}])
    replayed = []

    def answer_after_tool(request):
        replayed.append(request["messages"][-2]["content"])
        return reply(text(answer))(request)

    server = FakeAnthropicServer([reply(text(""), tool_use("toolu_s", "ctx_status")), answer_after_tool], pricing=s.pricing)
    result = LedgerClient(s, sdk_client(server), model=HAIKU, max_tokens=512).run_turn("What is the default timeout?", turn_id="t1")
    assert result.reason == "answered" and server.errors == []
    assert [b["type"] for b in replayed[0]] == ["tool_use"]
    assert all(not empty_blocks(r) for r in server.requests)
    kept = next(seg for seg in s.state.segments if seg.content_type == "blocks")
    assert s.blobs.get(kept.text_hash).decode().startswith('[{"text":"","type":"text"}')  # the model's turn, verbatim
    assert check(s.events, s.blobs) == []


def test_a_message_left_empty_is_sent_as_the_placeholder():
    """A model turn that is only text "": the next request carries it as "(empty)", as B4 does."""
    s = loaded()
    answer = answer_text([{"text": "The default timeout is 37 seconds.", "citations": [{"ref": alias(s, "DEFAULT_TIMEOUT"), "quote": "DEFAULT_TIMEOUT = 37.0"}]}])
    server = FakeAnthropicServer([reply(text("")), reply(text(answer))], pricing=s.pricing)
    result = LedgerClient(s, sdk_client(server), model=HAIKU, max_tokens=512).run_turn("What is the default timeout?", turn_id="t1")
    assert result.reason == "answered" and server.errors == []
    assistant, reformat = server.requests[1]["messages"][-2:]
    assert assistant == {"role": "assistant", "content": [{"type": "text", "text": EMPTY_TEXT_PLACEHOLDER}]}
    assert reformat["content"][0]["text"].startswith("Verification: your answer could not be parsed")
    assert check(s.events, s.blobs) == []


@pytest.mark.parametrize("blank", ["", " \n\t"])
def test_a_breakpoint_meant_for_a_dropped_segment_goes_on_the_nearest_earlier_block(blank):
    s = v1_session(10**6)
    s.set_pricing(low_min_cache_pricing())
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    s.append_raw("user", "What is the default timeout?")
    dropped = s.append_raw("user", blank)
    request, meta = build(s, prev_last=dropped.segment_id)
    assert request["messages"] == [{"role": "user", "content": [{"type": "text", "text": "What is the default timeout?", "cache_control": CACHE}]}]
    assert meta.paths[2] == () and meta.paths[1] == (("messages", 0, 0),)  # the blank segment renders into nothing
    assert [(b["bp"], b["position"]) for b in meta.breakpoints] == [("BP1", 0), ("BP3", 1)]  # BP3 moved to the question; BP4 is the same block
    assert request_problems(request) == []


def test_a_window_without_empty_text_renders_exactly_as_before(monkeypatch):
    s = loaded()
    s.append_raw("user", "What is the default timeout?")
    s.append_blocks([text("Let me check."), tool_use("toolu_1", "ctx_status")])
    s.append_raw("tool", "status: fine", tool_use_id="toolu_1")
    patched = build(s, tools=CTX_TOOLS)
    unpatched(monkeypatch)
    assert build(s, tools=CTX_TOOLS) == patched


# (b) request_problems and the fake server refuse an empty text block as the API does


def conversation(last: dict) -> dict:
    return {"model": HAIKU, "max_tokens": 64, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Question?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Answer."}]},
        {"role": "user", "content": [last]},
    ]}  # fmt: skip


@pytest.mark.parametrize(("last", "message"), [
    ({"type": "text", "text": "", "cache_control": CACHE}, "messages.2.content.0.text: cache_control cannot be set for empty text blocks"),
    ({"type": "text", "text": ""}, "messages: text content blocks must be non-empty"),
    ({"type": "text", "text": "  \n"}, "messages: text content blocks must contain non-whitespace text"),
])  # fmt: skip
def test_the_fake_refuses_an_empty_text_block_with_the_apis_400(last, message):
    body = conversation(last)
    assert empty_text_problems(body) == [message] and message in request_problems(body)
    server = FakeAnthropicServer([reply(text("never used"))], pricing=pricing())
    client = server.as_fake_client()
    for call in (lambda: client.messages.create(**body), lambda: client.messages.count_tokens(**{k: v for k, v in body.items() if k != "max_tokens"})):
        with pytest.raises(FakeAPIStatusError) as caught:
            call()
        assert (caught.value.status_code, caught.value.type, caught.value.body["error"]["message"]) == (400, "invalid_request_error", message)
    with pytest.raises(anthropic.BadRequestError, match=message.replace(".", r"\.")):
        sdk_client(server).messages.create(**body)
    assert server.requests == [] and server.counted == []  # refused before any scripted reply is used
    fixed = conversation({"type": "text", "text": "Next question?"})
    assert client.messages.create(**fixed)["content"] == [text("never used")]


# (c) The stopped runs' failing requests, rebuilt from their frozen windows


def stopped_window(case: dict):
    """The frozen window as a session (each segment appended as the text the request carried for
    it), with L's tools and the call's BP2 and BP3 segment ids."""
    s = v1_session(10**6)
    s.set_pricing(GRID_PRICING)
    s.append_raw("system", system_prompt("L", case["prompt_version"]), pinned=True)
    assert s.state.segments[0].text_hash == prompt_sha256("L", case["prompt_version"])
    for seg in case["segments"]:
        if "blocks" in seg:
            s.append_blocks(seg["blocks"])
        else:
            s.append_raw(seg["role"], seg["text"], tool_use_id=seg.get("tool_use_id"), is_error=seg.get("is_error", False))
    ids = [seg.segment_id for seg in s.state.segments]
    frozen = None if case["frozen"] is None else ids[case["frozen"]]
    return s, CTX_TOOLS + CORPUS_TOOLS, frozen, ids[case["prev_last"]]


def build_stopped(case: dict) -> tuple[dict, object, int]:
    s, tools, frozen, prev_last = stopped_window(case)
    request, meta = build_request(s.state, s.blobs, model=HAIKU, max_tokens=case["max_tokens"], tools=tools, ttl=case["ttl"], pricing=s.pricing,
                                  frozen=frozen, prev_last=prev_last, thinking=case["thinking"])  # fmt: skip
    return request, meta, case["empty"]


@pytest.mark.parametrize("run_id", sorted(CASES))
def test_the_stopped_runs_request_gets_the_apis_error_unpatched_and_is_valid_patched(run_id, monkeypatch):
    case = CASES[run_id]
    patched, meta, empty = build_stopped(case)
    with monkeypatch.context() as m:
        unpatched(m)
        old, old_meta, _ = build_stopped(case)
    assert empty_text_problems(old)[0] == case["api_error"]  # the API's first 400 for that call, word for word
    server = FakeAnthropicServer(responder=reply(text("ok")), pricing=GRID_PRICING, emulate=True)
    client = server.as_fake_client()
    for body in (old, {k: v for k, v in old.items() if k != "max_tokens"}):
        with pytest.raises(FakeAPIStatusError, match=case["api_error"].replace(".", r"\.")):
            (client.messages.create if "max_tokens" in body else client.messages.count_tokens)(**body)
    assert request_problems(patched) == [] and not empty_blocks(patched)
    client.messages.count_tokens(**{k: v for k, v in patched.items() if k != "max_tokens"})
    assert client.messages.create(**patched)["content"] == [text("ok")]
    # the only difference is the empty block: dropped beside another block, else sent as "(empty)"
    (old_path,) = old_meta.paths[empty]
    strip = lambda r: [[builder.strip_cache_control(b) for b in m["content"]] for m in r["messages"]]  # noqa: E731
    expected = strip(old)
    if meta.paths[empty]:
        expected[old_path[1]][old_path[2]] = {"type": "text", "text": EMPTY_TEXT_PLACEHOLDER}
    else:
        del expected[old_path[1]][old_path[2]]
    assert strip(patched) == expected and patched["system"] == old["system"] and patched["tools"] == old["tools"]
