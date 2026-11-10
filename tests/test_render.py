from __future__ import annotations

import pytest

from context_ledger.fold import BackstopPolicy, BackstopPolicyV1
from context_ledger.ids import canonical_json
from context_ledger.model import SummarySentence
from context_ledger.render import (
    APPROX,
    counter_for,
    render_request,
    render_segment,
    rendered_segment_tokens,
    rendered_window_tokens,
)
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.window import approx_tokens, segment_tokens, window_tokens
from m2util import MODULE, MODULE_PATH, pricing


def session(policy):
    return Session(Ledger(":memory:"), MemoryBlobStore(), policy.budget_tokens, policy, pricing=pricing())


def fill(s: Session):
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    note = s.register_source("docs/runbook.txt", "Staging runs 3 replicas; production runs 9.\n")
    spans["note"] = s.register_text_span(note, 0, 43)
    segs = {name: s.append_span(spans[name].span_id) for name in ["DEFAULT_TIMEOUT", "backoff_delay", "note"]}
    segs["question"] = s.append_raw("user", "What is the default timeout?")
    return spans, segs


def test_span_header_raw_and_fold_rendering():
    s = session(BackstopPolicyV1(budget_tokens=10_000))
    spans, segs = fill(s)
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    assert render_segment(s.state, segs["DEFAULT_TIMEOUT"], s.blobs) == f"[{alias} {MODULE_PATH}#DEFAULT_TIMEOUT]\nDEFAULT_TIMEOUT = 37.0"
    note_alias = s.state.aliases.alias_of(spans["note"].span_id)
    assert render_segment(s.state, segs["note"], s.blobs).startswith(f"[{note_alias} docs/runbook.txt#0-43]\nStaging")
    assert render_segment(s.state, segs["question"], s.blobs) == "What is the default timeout?"
    s.call()
    fold = s.fold([segs["backoff_delay"].segment_id], [SummarySentence("backoff", (spans["backoff_delay"].span_id,))])
    fold_seg = next(g for g in s.state.segments if g.ref_kind == "fold")
    assert render_segment(s.state, fold_seg, s.blobs) == s.blobs.get(fold.summary_hash).decode()


def test_rendered_tokens_sum_and_fold_delta_use_rendered_text():
    s = session(BackstopPolicyV1(budget_tokens=10_000))
    spans, segs = fill(s)
    total = rendered_window_tokens(s.state, s.blobs)
    assert total == sum(approx_tokens(render_segment(s.state, g, s.blobs)) for g in s.state.segments)
    assert total > window_tokens(s.state, s.blobs)  # headers cost tokens
    s.call()
    victim = segs["backoff_delay"]
    before = rendered_window_tokens(s.state, s.blobs)
    victim_tokens = rendered_segment_tokens(s.state, victim, s.blobs)
    fold = s.fold([victim.segment_id], [SummarySentence("backoff", (spans["backoff_delay"].span_id,))])
    assert fold.freed_tokens == victim_tokens - approx_tokens(s.blobs.get(fold.summary_hash).decode())
    assert before - rendered_window_tokens(s.state, s.blobs) == fold.freed_tokens


def test_v0_sessions_render_exactly_as_m1():
    s = session(BackstopPolicy(budget_tokens=10_000))
    fill(s)
    assert not s.state.aliases.count
    assert all(render_segment(s.state, g, s.blobs) == s.blobs.get(g.text_hash).decode() for g in s.state.segments)
    assert rendered_window_tokens(s.state, s.blobs) == window_tokens(s.state, s.blobs)
    assert all(rendered_segment_tokens(s.state, g, s.blobs) == segment_tokens(g, s.blobs) for g in s.state.segments)


def test_recorded_request_carries_headers_and_verbatim_text():
    s = session(BackstopPolicyV1(budget_tokens=10_000))
    spans, segs = fill(s)
    result = s.call()
    assert result.request == render_request(s.state, s.blobs)
    serialized = canonical_json(result.request).decode()
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    assert f"[{alias} {MODULE_PATH}#DEFAULT_TIMEOUT]\\nDEFAULT_TIMEOUT = 37.0" in serialized
    assert result.request["system"] == ["Answer only from cited spans."]


def test_token_counter_registry():
    assert counter_for("approx-bytes4") is APPROX and APPROX.count("abcde") == 2
    with pytest.raises(ValueError):
        counter_for("tiktoken")
