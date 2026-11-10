"""E: the four ctx tools, the MCP stub, and the summary-only-claim-to-refetch route end to end."""

from __future__ import annotations

import pytest

from context_ledger.anthropic.client import LedgerClient
from context_ledger.anthropic.fake import FakeAnthropicServer
from context_ledger.anthropic.request import TOOL_NAME
from context_ledger.fold import ReReadEstimator, SummarySentence
from context_ledger.invariants import check
from context_ledger.mcp_stub import call_tool, list_tools
from context_ledger.prompts import SYSTEM_PROMPT, answer_text
from context_ledger.tools import CTX_TOOLS, dispatch, search_tokens
from m2util import MODULE, MODULE_PATH, reply, sdk_client, text, tool_use, v1_session

NAMES = ["retrying", "backoff_delay", "InventoryClient.__init__", "InventoryClient.url_for", "DEFAULT_TIMEOUT", "MAX_RETRIES"]


def loaded(budget=100_000, **policy):
    s = v1_session(budget, **policy)
    s.append_raw("system", SYSTEM_PROMPT, pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    s.spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    s.append_raw("user", "Tell me about the client.")
    s.segs = {}
    for name in NAMES:
        s.segs[name] = s.append_span(s.spans[name].span_id)
        s.call()
    return s


def alias(s, name):
    return s.state.aliases.alias_of(s.spans[name].span_id)


def run(s, name, **input):
    return dispatch(s, name, input, turn_id="t1")


def test_tool_definitions():
    assert [t["name"] for t in CTX_TOOLS] == ["ctx_fold", "ctx_refetch", "ctx_search", "ctx_status"]
    for tool in CTX_TOOLS:
        assert TOOL_NAME.match(tool["name"]) and tool["description"]
        assert tool["input_schema"]["type"] == "object" and tool["input_schema"]["additionalProperties"] is False
    assert list_tools() == CTX_TOOLS


@pytest.mark.parametrize(
    ("name", "input", "first"),
    [
        ("ctx_explode", {}, "error: unknown_tool ctx_explode"),
        ("ctx_status", {"verbose": True}, "error: invalid_input"),
        ("ctx_refetch", {"refs": []}, "error: invalid_input"),
        ("ctx_refetch", {"refs": ["@1"] * 9}, "error: invalid_input"),
        ("ctx_search", {"query": "x", "limit": 11}, "error: invalid_input"),
        ("ctx_search", {"query": ""}, "error: invalid_input"),
        ("ctx_fold", {"targets": ["@1"], "summary": [{"text": "x" * 401}]}, "error: invalid_input"),
        ("ctx_fold", {"targets": ["@1"]}, "error: invalid_input"),
    ],
)
def test_bad_input_is_a_typed_error_never_an_exception(name, input, first):
    s = loaded()
    before = len(s.events)
    outcome = dispatch(s, name, input, turn_id="t1")
    assert outcome.is_error and outcome.text.startswith(first) and len(s.events) == before


def test_status_is_a_pure_read_and_lists_foldable_runs():
    s = loaded(budget=1200, estimator=ReReadEstimator("fixed", 40))
    before = len(s.events)
    outcome = run(s, "ctx_status")
    assert not outcome.is_error and len(s.events) == before
    lines = outcome.text.splitlines()
    assert lines[0].startswith("budget 1200 tokens; window ") and "zone " in lines[0]
    assert lines[1].startswith("knowledge: verbatim ")
    candidate = lines[3].strip()
    assert candidate.startswith("1. @") and " frees " in candidate and " J $" in candidate
    targets = candidate.split(":")[0][3:].split()
    assert run(s, "ctx_fold", targets=targets, summary=[{"text": "client setup code"}]).text.startswith("ok: folded")
    assert lines[-3:] == ["summary_only spans cited this turn: none", "pending refetch obligations: none", "turn cost so far: $0.000000"]


def test_fold_success_and_every_refusal():
    s = loaded(estimator=ReReadEstimator("fixed", 64))
    ok = run(s, "ctx_fold", targets=[alias(s, "retrying"), alias(s, "backoff_delay")],
             summary=[{"text": "retry decorator with capped exponential backoff", "cites": [alias(s, "retrying"), alias(s, "backoff_delay")]}])  # fmt: skip
    assert not ok.is_error and ok.text.startswith("ok: folded 2 segments, freed ") and ok.text.split()[-1].startswith("@")
    fold = next(iter(s.state.folds.values()))
    assert ok.text.endswith(f"new alias {s.state.aliases.alias_of(f'fold:{fold.fold_id}')}")
    before = list(s.events)
    bloated = run(s, "ctx_fold", targets=[alias(s, "InventoryClient.url_for")], summary=[{"text": "url " * 99}] * 3)
    assert bloated.is_error and bloated.text.startswith("error: fold_would_grow_window delta=-") and s.events == before
    tail = run(s, "ctx_fold", targets=[alias(s, "MAX_RETRIES")], summary=[{"text": "retries"}])
    assert tail.text == f"error: infeasible_run position={len(s.state.segments) - 1} why=tail"
    gap = run(s, "ctx_fold", targets=[alias(s, "InventoryClient.__init__"), alias(s, "DEFAULT_TIMEOUT")], summary=[{"text": "x"}])
    assert gap.text.startswith("error: infeasible_run why=not_contiguous")
    pinned = run(s, "ctx_fold", targets=["@0"], summary=[{"text": "x"}])
    assert pinned.text == "error: infeasible_run position=0 why=pinned"
    unknown = run(s, "ctx_fold", targets=["@ZZZ"], summary=[{"text": "x"}])
    assert unknown.text == "error: unknown_alias @ZZZ"
    outside = run(s, "ctx_fold", targets=[alias(s, "InventoryClient.__init__")], summary=[{"text": "x", "cites": [alias(s, "MAX_RETRIES")]}])
    assert outside.text.startswith("error: cites_outside_victims")
    assert s.events == before
    stingy = loaded(estimator=ReReadEstimator("fixed", 1))
    refused = run(stingy, "ctx_fold", targets=[alias(stingy, "retrying")], summary=[{"text": "retry decorator"}])
    assert refused.text.startswith("error: payback_not_met J=-$") and "rstar=" in refused.text and refused.text.endswith("rhat=1")
    assert check(s.events, s.blobs) == []


def test_refetch_spans_folds_and_source_edits():
    s = loaded(estimator=ReReadEstimator("fixed", 64))
    run(s, "ctx_fold", targets=[alias(s, "retrying"), alias(s, "backoff_delay")], summary=[{"text": "retry helpers", "cites": [alias(s, "retrying"), alias(s, "backoff_delay")]}])
    fold = next(iter(s.state.folds.values()))
    sentence = s.state.aliases.alias_of(f"fold:{fold.fold_id}#0")
    before = len(s.state.segments)
    outcome = dispatch(s, "ctx_refetch", {"refs": [sentence]}, turn_id="t1", tool_use_id="toolu_1")
    assert outcome.text.splitlines() == [f"{sentence} {alias(s, 'retrying')} OK", f"{sentence} {alias(s, 'backoff_delay')} OK"]
    assert len(outcome.appended) == 2 and len(s.state.segments) == before + 2
    assert all(seg.tool_use_id == "toolu_1" for seg in s.state.segments[-2:])
    s.update_source(s.spans["DEFAULT_TIMEOUT"].source_id, "# moved down\n" + MODULE.replace("37.0", "41.0"))
    edited = run(s, "ctx_refetch", refs=[alias(s, "DEFAULT_TIMEOUT"), alias(s, "MAX_RETRIES"), "@ZZZ"])
    lines = edited.text.splitlines()
    assert lines[0] == "@ZZZ UNKNOWN" and lines[1] == f"{alias(s, 'DEFAULT_TIMEOUT')} STALE"
    assert lines[2].startswith(f"{alias(s, 'MAX_RETRIES')} MOVED @") and not edited.is_error
    assert run(s, "ctx_refetch", refs=["@ZZZ"]).text.startswith("error: nothing_to_refetch")


def test_refetch_through_a_fold_is_capped_at_eight_in_first_appearance_order():
    s = v1_session()
    s.append_raw("user", "q")
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = s.register_python_spans(src, MODULE_PATH)[:10]
    segs = [s.append_span(sp.span_id) for sp in spans]
    s.call()
    fold = s.fold([g.segment_id for g in segs], [SummarySentence("the module", tuple(sp.span_id for sp in spans))])
    outcome = run(s, "ctx_refetch", refs=[s.state.aliases.alias_of(f"fold:{fold.fold_id}")])
    lines = outcome.text.splitlines()
    assert len(lines) == 9 and lines[-1] == "capped at 8: 2 more not re-fetched"
    assert [ln.split()[1] for ln in lines[:8]] == [s.state.aliases.alias_of(sp.span_id) for sp in spans[:8]]
    assert check(s.events, s.blobs) == []


def test_search_returns_metadata_only_and_assigns_aliases():
    s = loaded()
    unseen = s.spans["InventoryClient.restock_threshold"].span_id
    assert s.state.aliases.alias_of(unseen) is None
    before = [e.type for e in s.events]
    outcome = run(s, "ctx_search", query="restock threshold", limit=3)
    first = outcome.text.splitlines()[0]
    assert first == f"{s.state.aliases.alias_of(unseen)} {MODULE_PATH}#InventoryClient.restock_threshold (python function)"
    assert "Reorder when stock" not in outcome.text  # no span content, so no false exposure
    assert [e.type for e in s.events[len(before) :]] and {e.type for e in s.events[len(before) :]} == {"alias_assigned"}
    assert run(s, "ctx_search", query="restock threshold", limit=3).text == outcome.text  # deterministic
    assert run(s, "ctx_search", query="zebra").text == "no matches for 'zebra'"
    assert search_tokens("InventoryClient.url_for DEFAULT_TIMEOUT HTTPServer") == [
        "inventory", "client", "url", "for", "default", "timeout", "http", "server",
    ]  # fmt: skip
    assert check(s.events, s.blobs) == []


def test_mcp_stub_and_anthropic_path_give_identical_text():
    a, b = loaded(estimator=ReReadEstimator("fixed", 64)), loaded(estimator=ReReadEstimator("fixed", 64))
    calls = [
        ("ctx_status", {}),
        ("ctx_search", {"query": "timeout"}),
        ("ctx_refetch", {"refs": [alias(a, "backoff_delay")]}),
        ("ctx_fold", {"targets": [alias(a, "retrying")], "summary": [{"text": "x " * 300}]}),
        ("ctx_fold", {"targets": [alias(a, "retrying")], "summary": [{"text": "retry decorator"}]}),
        ("ctx_status", {}),
    ]
    for n, (name, arguments) in enumerate(calls):
        mcp = call_tool(a, name, arguments, turn_id="t1")
        native = dispatch(b, name, arguments, turn_id="t1", tool_use_id=None)
        assert mcp == {"content": [{"type": "text", "text": native.text}], "isError": native.is_error}, name


def test_summary_only_claim_routes_to_refetch_end_to_end():
    """Through the real SDK: the model cites the summary, is told to re-fetch, calls ctx_refetch,
    and answers again from verbatim text; the grounded claim supersedes the first."""
    s = loaded(estimator=ReReadEstimator("fixed", 64))
    folded = run(s, "ctx_fold", targets=[alias(s, "retrying"), alias(s, "backoff_delay")],
                 summary=[{"text": "retry decorator; backoff_delay caps the wait", "cites": [alias(s, "backoff_delay")]}])  # fmt: skip
    assert folded.text.startswith("ok: folded")
    fold = next(iter(s.state.folds.values()))
    sentence = s.state.aliases.alias_of(f"fold:{fold.fold_id}#0")
    span = alias(s, "backoff_delay")
    guess = answer_text([{"text": "The backoff wait is capped at 600 seconds.", "citations": [{"ref": sentence, "quote": "backoff_delay caps the wait"}]}])
    right = answer_text([{"text": "The backoff wait is capped at 300 seconds.", "citations": [{"ref": span, "quote": "return min(2 ** attempt, 300)"}]}])

    def after_verification(request):
        assert "no longer hold verbatim (summary_only)" in request["messages"][-1]["content"][-1]["text"]
        return {"content": [tool_use("toolu_r", "ctx_refetch", refs=[sentence])], "stop_reason": "tool_use"}

    server = FakeAnthropicServer([reply(text(guess)), after_verification, reply(text(right))], pricing=s.pricing)
    result = LedgerClient(s, sdk_client(server), model="claude-haiku-5-5", max_tokens=512).run_turn("What caps the backoff?", turn_id="t1")
    assert result.reason == "answered" and result.refetch_routes == 1
    first, second = result.claims
    assert (first.status, second.status, second.supersedes) == ("needs_refetch", "grounded", first.claim_id)
    kinds = [e.type for e in s.events]
    order = [kinds.index("claim", kinds.index("fold")), max(i for i, k in enumerate(kinds) if k == "refetch"), len(kinds) - 1]
    assert order == sorted(order) and kinds[-1] == "answer_emitted" and s.events[-1].payload["claim_ids"] == [second.claim_id]
    assert check(s.events, s.blobs) == []
