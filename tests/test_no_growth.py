"""A.7: no fold may grow the window (deviation 8 of M1)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from context_ledger.fold import BackstopPolicy, BackstopPolicyV1, FoldError, extractive_summary, make_fold
from context_ledger.invariants import check
from context_ledger.model import SummarySentence
from context_ledger.render import rendered_window_tokens
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.window import window_tokens
from m2util import MODULE, MODULE_PATH, pricing

# Small spans whose M1 summary lines each carry a 64-hex span id: the M1 demo's growing fold.
SMALL = [
    "InventoryClient.__init__",
    "InventoryClient.url_for",
    "InventoryClient.restock_threshold",
    "backoff_delay",
    "MAX_RETRIES",
    "USER_AGENT",
    "DEFAULT_TIMEOUT",
]


def loaded(policy):
    s = Session(Ledger(":memory:"), MemoryBlobStore(), policy.budget_tokens, policy, pricing=pricing())
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    segs = [s.append_span(spans[name].span_id) for name in SMALL]
    s.call()
    return s, [g.segment_id for g in segs]


def test_m1_growing_fold_is_refused_and_shrinks_with_aliases():
    v0, victims = loaded(BackstopPolicy(budget_tokens=100_000))
    before = list(v0.events)
    with pytest.raises(FoldError) as info:
        v0.fold(victims, extractive_summary(v0.state, victims, v0.blobs))
    assert info.value.reason == "fold_would_grow_window" and info.value.details["delta"] < 0
    assert v0.events == before

    v1, victims = loaded(BackstopPolicyV1(budget_tokens=100_000))
    tokens = rendered_window_tokens(v1.state, v1.blobs)
    fold = v1.fold(victims, extractive_summary(v1.state, victims, v1.blobs))
    assert fold.freed_tokens > 0
    assert rendered_window_tokens(v1.state, v1.blobs) == tokens - fold.freed_tokens
    assert check(v1.events, v1.blobs) == []


@pytest.mark.parametrize("policy", [BackstopPolicy(budget_tokens=100_000), BackstopPolicyV1(budget_tokens=100_000)])
@pytest.mark.parametrize("trigger", ["model", "backstop"])
def test_crafted_fold_with_nonpositive_delta_is_refused(policy, trigger):
    s, victims = loaded(policy)
    recorded = {"id": "crafted"} if trigger == "backstop" else None
    deltas = set()
    for length in range(0, 400):
        sentence = [SummarySentence("x" * length, ())]
        try:
            make_fold(s.state, 1, victims[:1], sentence, trigger, recorded, blobs=s.blobs, aliased=s.aliased)
        except FoldError as exc:
            assert exc.reason == "fold_would_grow_window" and exc.details["delta"] <= 0
            deltas.add(exc.details["delta"])
    assert 0 in deltas and min(deltas) < 0  # delta == 0 is refused too


@settings(max_examples=60, deadline=None)
@given(
    sizes=st.lists(st.integers(1, 300), min_size=2, max_size=12),
    run=st.tuples(st.integers(0, 11), st.integers(1, 6)),
    cite=st.booleans(),
    summary=st.text(alphabet="abc xyz", min_size=1, max_size=200),
    aliased=st.booleans(),
)
def test_random_folds_never_record_nonpositive_freed_tokens(sizes, run, cite, summary, aliased):
    policy = BackstopPolicyV1(budget_tokens=100_000) if aliased else BackstopPolicy(budget_tokens=100_000)
    s = Session(Ledger(":memory:"), MemoryBlobStore(), 100_000, policy, pricing=pricing())
    s.append_raw("system", "sys", pinned=True)
    src = s.register_source("doc.txt", "".join(f"{i:04d} " + "w" * n + "\n" for i, n in enumerate(sizes)))
    text = s.blobs.get(src.content_hash).decode()
    segs, start = [], 0
    for line in text.splitlines(keepends=True):
        segs.append(s.append_span(s.register_text_span(src, start, start + len(line)).span_id))
        start += len(line)
    s.call()
    i = min(run[0], len(segs) - 1)
    chosen = segs[i : i + run[1]]
    cites = tuple(g.ref for g in chosen) if cite else ()
    before = (len(s.events), rendered_window_tokens(s.state, s.blobs), window_tokens(s.state, s.blobs))
    try:
        fold = s.fold([g.segment_id for g in chosen], [SummarySentence(summary, cites)])
    except FoldError as exc:
        assert exc.reason == "fold_would_grow_window" and len(s.events) == before[0]
        return
    assert fold.freed_tokens > 0
    after = rendered_window_tokens(s.state, s.blobs) if aliased else window_tokens(s.state, s.blobs)
    assert after == (before[1] if aliased else before[2]) - fold.freed_tokens
    assert check(s.events, s.blobs) == []
