from __future__ import annotations

import pytest

from context_ledger.model import Event, FoldRecord, Knowledge, SummarySentence
from context_ledger.window import (
    ReplayError,
    WindowState,
    approx_tokens,
    down,
    kappa,
    manifest,
    manifest_hash,
    parse_fold_ref,
    read_map,
    replay,
    state_at_call,
)

TEXT = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
LONG = TEXT * 4  # for tests that fold spans: M2 refuses folds that do not free tokens


def test_unseen_verbatim_summary_verbatim(builder):
    src = builder.source("note.txt", LONG)
    span = builder.span(src, 0, 200)  # long enough that folding it frees tokens
    first = builder.call()
    seg = builder.append(span)
    seen = builder.call()
    builder.fold([seg.segment_id], [SummarySentence("about alpha", (span.span_id,))])
    folded = builder.call()
    builder.append(span)  # re-fetched verbatim
    refetched = builder.call()
    events = builder.events()
    assert kappa(state_at_call(events, first), span.span_id) is Knowledge.UNSEEN
    assert kappa(state_at_call(events, seen), span.span_id) is Knowledge.VERBATIM
    assert kappa(state_at_call(events, folded), span.span_id) is Knowledge.SUMMARY_ONLY
    assert kappa(state_at_call(events, refetched), span.span_id) is Knowledge.VERBATIM


def test_dropped_after_evict_only_if_exposed(builder):
    src = builder.source("note.txt", TEXT)
    shown, hidden = builder.span(src, 0, 5), builder.span(src, 6, 10)
    shown_seg = builder.append(shown)
    builder.call()
    hidden_seg = builder.append(hidden)
    builder.evict([shown_seg.segment_id, hidden_seg.segment_id])
    call = builder.call()
    state = state_at_call(builder.events(), call)
    assert kappa(state, shown.span_id) is Knowledge.DROPPED
    assert kappa(state, hidden.span_id) is Knowledge.UNSEEN


def test_stale_after_source_update_overrides_verbatim(builder):
    src = builder.source("note.txt", TEXT)
    span = builder.span(src, 0, 10)
    builder.append(span)
    before = builder.call()
    builder.update(src, "changed " + TEXT)
    after = builder.call()
    events = builder.events()
    assert kappa(state_at_call(events, before), span.span_id) is Knowledge.VERBATIM
    assert kappa(state_at_call(events, after), span.span_id) is Knowledge.STALE


def test_snippet_marks_parent(builder):
    src = builder.source("note.txt", TEXT)
    parent = builder.span(src, 0, 30)
    child = builder.span(src, 6, 10)
    builder.append(child, parent=parent.span_id)
    state = replay(builder.events())
    assert kappa(state, child.span_id) is Knowledge.VERBATIM
    assert kappa(state, parent.span_id) is Knowledge.SNIPPET
    assert read_map(state) == {child.span_id: "verbatim", parent.span_id: "snippet"}


def test_down_through_fold_of_fold(builder):
    src = builder.source("note.txt", LONG)
    s1, s2, s3 = builder.span(src, 0, 100), builder.span(src, 101, 200), builder.span(src, 201, 216)
    g1, g2, g3 = builder.append(s1), builder.append(s2), builder.append(s3)
    inner = builder.fold(
        [g1.segment_id, g2.segment_id],
        [SummarySentence("one", (s1.span_id,)), SummarySentence("two", (s2.span_id, "inference"))],
    )
    outer = builder.fold(
        [inner.summary_segment_id, g3.segment_id],
        [SummarySentence("both", (f"fold:{inner.fold_id}#0", f"fold:{inner.fold_id}#1")), SummarySentence("three", (s3.span_id,))],
    )
    state = replay(builder.events())
    assert down(state, f"fold:{outer.fold_id}#0") == {s1.span_id, s2.span_id}
    assert down(state, f"fold:{outer.fold_id}#1") == {s3.span_id}
    assert down(state, s1.span_id) == {s1.span_id}
    assert down(state, "inference") == down(state, "raw:seg_x") == down(state, "fold:nope#0") == frozenset()
    assert all(kappa(state, s.span_id) is Knowledge.SUMMARY_ONLY for s in (s1, s2, s3))
    assert [s.ref_kind for s in state.segments] == ["fold"]


def test_down_is_cycle_safe():
    state = WindowState()
    loop = FoldRecord("f1", 0, ("v",), (SummarySentence("x", ("fold:f1#0", "span_a")),), "model", None, 0, "seg", "h")
    state.folds["f1"] = loop
    assert down(state, "fold:f1#0") == {"span_a"}


def test_parse_fold_ref():
    assert parse_fold_ref("fold:fold_ab#3") == ("fold_ab", 3)
    assert parse_fold_ref("fold:x#") is None and parse_fold_ref("fold:#1") is None and parse_fold_ref("abc") is None


def test_manifest_hash_changes_iff_window_changes(builder):
    src = builder.source("note.txt", LONG)
    span = builder.span(src, 0, 200)
    other = builder.span(src, 11, 20)
    seg = builder.append(span)
    builder.call()
    builder.append(other)
    builder.update(src, LONG + " nu")
    builder.fold([seg.segment_id], [SummarySentence("alpha", (span.span_id,))])
    builder.call()
    events = builder.events()
    previous = manifest_hash(replay([]))
    for i, event in enumerate(events):
        before = replay(events[:i])
        after = replay(events[: i + 1])
        changed = manifest(before) != manifest(after)
        assert (manifest_hash(after) != previous) == changed, event.type
        assert changed == (event.type in {"segment_appended", "fold", "evict"})
        previous = manifest_hash(after)


def test_replay_rejects_impossible_events(builder):
    src = builder.source("note.txt", TEXT)
    builder.append(builder.span(src, 0, 5))
    events = builder.events()
    bad = Event(len(events) + 1, "evict", {"segment_ids": ["seg_missing"], "reason": "x"}, "", "")
    with pytest.raises(ReplayError):
        replay(events + [bad])
    with pytest.raises(ReplayError):
        replay(events + [Event(len(events) + 1, "fold", {"fold": "not a dict"}, "", "")])
    with pytest.raises(ReplayError):
        replay(events + [events[-1]])  # the same segment twice


def test_replay_upto_seq(builder):
    src = builder.source("note.txt", TEXT)
    builder.append(builder.span(src, 0, 5))
    builder.append(builder.span(src, 6, 10))
    events = builder.events()
    assert len(replay(events, upto_seq=4).segments) == 1
    assert len(replay(events).segments) == 2
    with pytest.raises(KeyError):
        state_at_call(events, "call_missing")


def test_approx_tokens():
    assert [approx_tokens(t) for t in ["", "a", "abcd", "abcde", "é"]] == [0, 1, 1, 2, 1]
