"""C.2: refetch obligations are typed outcomes when a claim is recorded, and an error at egress."""

from __future__ import annotations

import pytest

from context_ledger.invariants import check
from context_ledger.model import Citation, SummarySentence
from context_ledger.session import RefetchObligationError, SessionError, UngroundedClaimError
from context_ledger.verify import RefetchRequired
from m2util import MODULE, MODULE_PATH, v1_session


@pytest.fixture
def world():
    """A v1 session: backoff_delay and retrying folded into one summary, DEFAULT_TIMEOUT and
    MAX_RETRIES verbatim, a question asked and a call made."""
    s = v1_session()
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    segs = {n: s.append_span(spans[n].span_id) for n in ["retrying", "backoff_delay", "DEFAULT_TIMEOUT", "MAX_RETRIES"]}
    s.call()
    fold = s.fold(
        [segs["retrying"].segment_id, segs["backoff_delay"].segment_id],
        [SummarySentence("retry decorator and a capped backoff", (spans["retrying"].span_id, spans["backoff_delay"].span_id))],
    )
    s.append_raw("user", "What is the backoff cap?")
    s.call()
    return s, spans, segs, fold


def test_summary_only_claim_is_recorded_as_needs_refetch(world):
    s, spans, segs, fold = world
    sentence = s.state.aliases.alias_of(f"fold:{fold.fold_id}#0")
    claim, verdict = s.claim("The cap is 600 seconds.", [Citation(sentence)], turn_id="t1")
    assert claim.status == "needs_refetch" and s.events[-1].payload["claim"] == claim.to_dict()
    assert isinstance(verdict, RefetchRequired) and verdict.claim_id == claim.claim_id
    assert verdict.reason == "summary_only" and verdict.action == "refetch"
    assert verdict.obligations == tuple(sorted([spans["retrying"].span_id, spans["backoff_delay"].span_id]))
    assert claim.verdict["obligations"] == list(verdict.obligations)
    assert check(s.events, s.blobs) == []


def test_reason_priority_stale_over_dropped_over_summary_only(world):
    s, spans, segs, fold = world
    s.evict([segs["MAX_RETRIES"].segment_id], "test")
    s.call()
    _, dropped = s.claim("MAX_RETRIES is 4", [Citation(spans["MAX_RETRIES"].span_id, "MAX_RETRIES = 4")])
    assert (dropped.reason, dropped.obligations) == ("dropped", (spans["MAX_RETRIES"].span_id,))
    s.update_source(spans["DEFAULT_TIMEOUT"].source_id, MODULE.replace("37.0", "41.0"))
    s.call()
    _, stale = s.claim(
        "both", [Citation(spans["MAX_RETRIES"].span_id, "MAX_RETRIES = 4"), Citation(spans["DEFAULT_TIMEOUT"].span_id, "DEFAULT_TIMEOUT = 37.0")]
    )
    assert stale.reason == "stale" and set(stale.obligations) >= {spans["DEFAULT_TIMEOUT"].span_id}


def test_unmet_p6_obligation_is_needs_refetch(world):
    s, spans, segs, fold = world
    s.append_span(spans["backoff_delay"].span_id)  # verbatim again, but not by a refetch
    s.call()
    claim, verdict = s.claim(
        "The cap is 300.",
        [Citation(f"fold:{fold.fold_id}#0"), Citation(spans["backoff_delay"].span_id, "return min(2 ** attempt, 300)")],
    )
    assert claim.status == "needs_refetch" and verdict.reason == "refetch_obligation"


def test_full_lifecycle_supersede_and_emit(world):
    s, spans, segs, fold = world
    sentence = s.state.aliases.alias_of(f"fold:{fold.fold_id}#0")
    guess, _ = s.claim("The cap is 600 seconds.", [Citation(sentence)], turn_id="t1")
    with pytest.raises(RefetchObligationError) as info:
        s.emit_answer("t1", [guess.claim_id])
    assert info.value.obligations == guess.verdict["obligations"] and info.value.claim_ids == [guess.claim_id]
    with pytest.raises(RefetchObligationError):  # an unsuperseded needs_refetch blocks even an abstaining answer
        s.emit_answer("t1", [], abstained=True)
    assert not any(e.type == "answer_emitted" for e in s.events)
    s.refetch(spans["backoff_delay"].span_id)
    s.call()
    fresh = s.state.aliases.alias_of(spans["backoff_delay"].span_id)
    good, verdict = s.claim(
        "The backoff cap is 300 seconds.",
        [Citation(sentence), Citation(fresh, "return min(2 ** attempt, 300)")],
        turn_id="t1",
        supersedes=guess.claim_id,
    )
    assert good.status == "grounded" and good.supersedes == guess.claim_id
    event = s.emit_answer("t1", [good.claim_id])
    assert event.payload == {"turn_id": "t1", "claim_ids": [good.claim_id], "abstained": False}
    assert s.unresolved_claims("t1") == []
    assert check(s.events, s.blobs) == []


def test_ungrounded_claims_block_egress_unless_superseded_or_abstained(world):
    s, spans, segs, fold = world
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    wrong, _ = s.claim("DEFAULT_TIMEOUT is 41.", [Citation(alias, "DEFAULT_TIMEOUT = 37.0")], turn_id="t2")
    assert wrong.status == "ungrounded"
    with pytest.raises(UngroundedClaimError):
        s.emit_answer("t2", [wrong.claim_id], abstained=True)  # listed ungrounded claims never leave
    with pytest.raises(UngroundedClaimError):
        s.emit_answer("t2", [])
    gave_up = s.abstain("No verified answer.", turn_id="t2")
    s.emit_answer("t2", [gave_up.claim_id], abstained=True)  # the abstention covers the bad claim
    assert check(s.events, s.blobs) == []


def test_claims_are_never_refused_and_supersedes_is_checked(world):
    s, spans, segs, fold = world
    made_up, verdict = s.claim("It is 9.", [Citation("f" * 64, "made up")], turn_id="t3")
    assert (made_up.status, verdict.reason) == ("ungrounded", "unknown_ref")
    grounded, _ = s.claim("MAX_RETRIES is 4", [Citation(spans["MAX_RETRIES"].span_id, "MAX_RETRIES = 4")], turn_id="t3")
    with pytest.raises(SessionError):
        s.claim("x", [], supersedes=grounded.claim_id)  # only needs_refetch or ungrounded claims
    with pytest.raises(SessionError):
        s.claim("x", [], supersedes="claim_missing")
    s.claim("MAX_RETRIES is 4", [Citation(spans["MAX_RETRIES"].span_id, "MAX_RETRIES = 4")], turn_id="t3", supersedes=made_up.claim_id)
    with pytest.raises(SessionError):
        s.claim("again", [], supersedes=made_up.claim_id)  # already superseded
    with pytest.raises(SessionError):
        s.emit_answer("other-turn", [grounded.claim_id])
    s.emit_answer("t3", [grounded.claim_id])
    assert check(s.events, s.blobs) == []
