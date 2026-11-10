from __future__ import annotations

import pytest

from context_ledger.alias import AliasMap, UnknownAlias, alias_for, b32, is_alias
from context_ledger.fold import BackstopPolicyV1
from context_ledger.invariants import check
from context_ledger.model import Citation, FoldRecord, SummarySentence
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.window import fold_sentence_refs, replay
from m2util import v1_session

MODULE = '''DEFAULT_TIMEOUT = 37


def backoff(attempt):
    """Seconds to wait before retry number `attempt`."""
    return min(2 ** attempt, 300)
'''


def test_b32_sequence():
    assert [b32(n) for n in (0, 1, 9, 10, 17, 31, 32, 33, 1023, 1024)] == ["0", "1", "9", "A", "H", "Z", "10", "11", "ZZ", "100"]
    assert [alias_for(n) for n in range(34)][-3:] == ["@Z", "@10", "@11"]
    assert all(is_alias(alias_for(n)) for n in range(2000))
    assert not any(is_alias(t) for t in ["@", "@I", "@l", "@O", "@U", "3", "@retrying", "@a"])
    with pytest.raises(ValueError):
        b32(-1)


def test_alias_map():
    m = AliasMap()
    assert m.next_alias() == "@0" and m.introduce("@0", "span_a")
    assert m.alias_of("span_a") == "@0" and m.target_of("@0") == "span_a"
    assert m.resolve("@0") == "span_a" and m.resolve("span_b") == "span_b"
    with pytest.raises(UnknownAlias):
        m.resolve("@7")
    assert not m.introduce("@1", "span_a")  # target already aliased: counted, not mapped
    assert not m.introduce("@0", "span_c")  # alias reused
    assert m.count == 3 and len(m) == 1 and m.next_alias() == "@3"


def aliased_session(budget: int = 10_000) -> Session:
    return v1_session(budget)


def test_session_assigns_dense_aliases_and_replay_rebuilds_them():
    s = aliased_session()
    system = s.append_raw("system", "Answer from cited spans.", pinned=True)
    src = s.register_source("pkg/mod.py", MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, "pkg/mod.py")}
    seg_t = s.append_span(spans["DEFAULT_TIMEOUT"].span_id)
    seg_b = s.append_span(spans["backoff"].span_id)
    s.append_span(spans["backoff"].span_id)  # appended again: no new alias
    aliases = s.state.aliases
    assert aliases.alias_of(f"raw:{system.segment_id}") == "@0"
    assert aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id) == "@1"
    assert aliases.alias_of(spans["backoff"].span_id) == "@2" and aliases.count == 3
    order = [e.type for e in s.events if e.type in ("alias_assigned", "segment_appended")]
    assert order[:4] == ["segment_appended", "alias_assigned", "alias_assigned", "segment_appended"]
    s.call()
    fold = s.fold([seg_t.segment_id, seg_b.segment_id], [SummarySentence("timeout and backoff", (spans["DEFAULT_TIMEOUT"].span_id, spans["backoff"].span_id))])
    assert fold.aliases == (("@3", f"fold:{fold.fold_id}"), ("@4", f"fold:{fold.fold_id}#0"))
    text = s.blobs.get(fold.summary_hash).decode()
    assert text == "[fold @3]\n- (@4) timeout and backoff [@1 @2]"
    assert FoldRecord.from_dict(fold.to_dict()) == fold
    assert replay(s.events).aliases == s.state.aliases
    assert check(s.events, s.blobs) == []


def test_fold_introduces_aliases_for_unaliased_cites():
    s = aliased_session()
    src = s.register_source("notes.txt", "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 4)
    parent = s.register_text_span(src, 0, 200)
    child = s.register_text_span(src, 6, 160)
    seg = s.append_span(child.span_id, parent=parent.span_id)
    s.call()
    fold = s.fold([seg.segment_id], [SummarySentence("beta gamma", (child.span_id, parent.span_id))])
    targets = [t for _, t in fold.aliases]
    assert targets == [f"fold:{fold.fold_id}", f"fold:{fold.fold_id}#0", parent.span_id]
    assert check(s.events, s.blobs) == []


def test_claims_resolve_aliases_and_unknown_alias_is_ungrounded():
    s = aliased_session()
    src = s.register_source("pkg/mod.py", MODULE, media_type="text/x-python")
    span = next(sp for sp in s.register_python_spans(src, "pkg/mod.py") if sp.anchor.qualname == "DEFAULT_TIMEOUT")
    s.append_span(span.span_id)
    s.call()
    alias = s.state.aliases.alias_of(span.span_id)
    claim, verdict = s.claim("DEFAULT_TIMEOUT is 37", [Citation(alias, "DEFAULT_TIMEOUT = 37")])
    assert claim.status == "grounded" and claim.citations == (Citation(span.span_id, "DEFAULT_TIMEOUT = 37", alias),)
    bad, verdict = s.claim("DEFAULT_TIMEOUT is 37", [Citation("@Z9", "DEFAULT_TIMEOUT = 37")])
    assert (bad.status, verdict.reason) == ("ungrounded", "unknown_ref")
    assert bad.citations[0].ref == "@Z9" == bad.citations[0].alias
    assert check(s.events, s.blobs) == []


def test_whole_fold_alias_reduces_to_every_sentence():
    s = aliased_session()
    src = s.register_source("pkg/mod.py", MODULE, media_type="text/x-python")
    spans = s.register_python_spans(src, "pkg/mod.py")
    segs = [s.append_span(sp.span_id) for sp in spans]
    s.call()
    fold = s.fold([g.segment_id for g in segs], [SummarySentence("module", tuple(sp.span_id for sp in spans))])
    from context_ledger.window import down

    whole = s.state.aliases.target_of(fold.aliases[0][0])
    assert whole == f"fold:{fold.fold_id}"
    assert down(s.state, whole) == {sp.span_id for sp in spans} == down(s.state, fold_sentence_refs(fold)[0])
