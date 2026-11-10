"""Mutation tests for the M2 invariants: each tamper (re-chained, so P11 holds) must produce
exactly the targeted violation code."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from context_ledger.fold import BackstopPolicy, BackstopPolicyV1, fold_delta, fold_id_for, plan_fold_aliases, render_fold
from context_ledger.ids import new_id
from context_ledger.invariants import check
from context_ledger.model import Citation, Event, FoldRecord, SummarySentence
from context_ledger.session import BudgetExceeded, Session
from context_ledger.store import Ledger, MemoryBlobStore, rechain
from context_ledger.render import rendered_window_tokens
from context_ledger.window import replay
from m2util import MODULE, MODULE_PATH, index_of, mutate, props, v1_session

BUDGET = 100_000


@dataclass
class Scenario:
    events: list[Event]
    blobs: MemoryBlobStore
    spans: dict


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    """An aliased (backstop-v1) session: spans and raw messages with aliases, a model fold whose
    summary cites aliases, claims citing aliases (grounded and unknown)."""
    ledger, blobs = Ledger(":memory:"), MemoryBlobStore()
    s = v1_session(BUDGET, ledger, blobs)
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    segs = {name: s.append_span(spans[name].span_id) for name in ["retrying", "backoff_delay", "DEFAULT_TIMEOUT"]}
    s.append_raw("user", "What is the default timeout?")
    s.call()
    s.fold(
        [segs["retrying"].segment_id, segs["backoff_delay"].segment_id],
        [SummarySentence("retry decorator and backoff", (spans["retrying"].span_id, spans["backoff_delay"].span_id))],
    )
    s.call()
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    claim, _ = s.claim("The default timeout is 37 seconds.", [Citation(alias, "DEFAULT_TIMEOUT = 37.0")])
    assert claim.status == "grounded"
    s.claim("It is 41.", [Citation("@ZZ", "DEFAULT_TIMEOUT = 41")])
    # A turn whose first claim rests on the summary: needs_refetch, refetch, grounded, emitted.
    sentence = s.state.aliases.alias_of(f"fold:{next(iter(s.state.folds))}#0")
    guess, _ = s.claim("The cap is 600 seconds.", [Citation(sentence)], turn_id="t1")
    assert guess.status == "needs_refetch"
    s.refetch(spans["backoff_delay"].span_id)
    s.call()
    good, _ = s.claim(
        "The cap is 300 seconds.",
        [Citation(sentence), Citation(s.state.aliases.alias_of(spans["backoff_delay"].span_id), "return min(2 ** attempt, 300)")],
        turn_id="t1",
        supersedes=guess.claim_id,
    )
    assert good.status == "grounded"
    s.emit_answer("t1", [good.claim_id])
    s.append_span(spans["MAX_RETRIES"].span_id)
    s.call()
    events = ledger.events()
    ledger.close()
    return Scenario(events, blobs, spans)


def test_unmutated_m2_ledger_reports_nothing(scenario):
    assert check(scenario.events, scenario.blobs) == []


def test_p12_freed_tokens_changed(scenario):
    i = index_of(scenario.events, "fold")
    events = mutate(scenario.events, i, lambda p: p["fold"].update(freed_tokens=p["fold"]["freed_tokens"] + 1))
    assert props(events, scenario.blobs) == {"P12"}


def growing_fold(events, blobs, aliased: bool) -> Event:
    """A fold event, valid in every other respect, that folds the last segment into a summary
    longer than it."""
    state = replay(events)
    victim = state.segments[-1]
    sentences = (SummarySentence("a summary much longer than the victim " * 4, ()),)
    fold_id = fold_id_for("model", len(state.segments) - 1, [victim.segment_id])
    aliases = plan_fold_aliases(state.aliases, fold_id, sentences) if aliased else ()
    text = render_fold(fold_id, sentences, aliases, state.aliases)
    freed = fold_delta(state, [victim], text, blobs, aliased=aliased)
    assert freed < 0
    record = FoldRecord(fold_id, len(state.segments) - 1, (victim.segment_id,), sentences, "model", None, freed,
                        new_id("seg", "fold", fold_id), blobs.put(text.encode()), aliases)  # fmt: skip
    return Event(0, "fold", {"fold": record.to_dict()}, "", "")


def test_p12_a_fold_that_grows(scenario):
    blobs = scenario.blobs.copy()
    events = rechain(scenario.events + [growing_fold(scenario.events, blobs, aliased=True)])
    found = check(events, blobs)
    assert {v.prop for v in found} == {"P12"} and {v.seq for v in found} == {len(events)}


def test_p12_flags_a_growing_fold_in_an_m1_format_ledger():
    """M1 ledgers replay unchanged; one that recorded a growing fold (as M1's backstop could)
    now reports exactly P12."""
    ledger, blobs = Ledger(":memory:"), MemoryBlobStore()
    s = Session(ledger, blobs, BUDGET, BackstopPolicy(budget_tokens=BUDGET))
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    span = next(sp for sp in s.register_python_spans(src, MODULE_PATH) if sp.anchor.qualname == "MAX_RETRIES")
    s.append_span(span.span_id)
    s.call()
    assert check(s.events, blobs) == []
    events = rechain(s.events + [growing_fold(s.events, blobs, aliased=False)])
    assert props(events, blobs) == {"P12"}


def test_p14_answer_lists_a_needs_refetch_claim(scenario):
    i = index_of(scenario.events, "answer_emitted")
    guess = scenario.events[index_of(scenario.events, "claim", lambda p: p["claim"]["status"] == "needs_refetch")]
    events = mutate(scenario.events, i, lambda p: p["claim_ids"].append(guess.payload["claim"]["claim_id"]))
    assert props(events, scenario.blobs) == {"P14"}


def test_p14_answer_leaves_a_needs_refetch_claim_unsuperseded(scenario):
    i = index_of(scenario.events, "claim", lambda p: p["claim"].get("supersedes") is not None)
    events = mutate(scenario.events, i, lambda p: p["claim"].pop("supersedes"))
    assert props(events, scenario.blobs) == {"P14"}


def test_p13_duplicate_alias(scenario):
    i = index_of(scenario.events, "alias_assigned", last=True)
    first = index_of(scenario.events, "alias_assigned")
    events = mutate(scenario.events, i, lambda p: p.update(alias=scenario.events[first].payload["alias"]))
    assert props(events, scenario.blobs) == {"P13"}


def test_p13_skipped_counter(scenario):
    i = index_of(scenario.events, "alias_assigned", last=True)
    events = mutate(scenario.events, i, lambda p: p.update(alias="@ZZZ"))
    assert props(events, scenario.blobs) == {"P13"}


def test_p13_alias_to_unknown_target(scenario):
    i = index_of(scenario.events, "alias_assigned", last=True)
    events = mutate(scenario.events, i, lambda p: p.update(target="0" * 64))
    assert props(events, scenario.blobs) == {"P13"}


def test_p13_citation_alias_mismatch(scenario):
    i = index_of(scenario.events, "claim", lambda p: p["claim"]["status"] == "grounded")
    other = scenario.events[index_of(scenario.events, "alias_assigned")].payload["alias"]
    events = mutate(scenario.events, i, lambda p: p["claim"]["citations"][0].update(alias=other))
    assert props(events, scenario.blobs) == {"P13"}


@pytest.fixture(scope="module")
def backstop_scenario() -> Scenario:
    """A v1 session whose window grows into the mandatory zone, so prepare_call records a
    backstop-v1 fold with its decision record; a call follows it."""
    ledger, blobs = Ledger(":memory:"), MemoryBlobStore()
    s = v1_session(10**7, ledger, blobs)
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    for name in ["retrying", "backoff_delay", "InventoryClient.__init__", "InventoryClient.restock_threshold", "DEFAULT_TIMEOUT"]:
        s.append_span(spans[name].span_id)
        s.call()
    used = rendered_window_tokens(s.state, s.blobs)
    s.policy = BackstopPolicyV1(budget_tokens=used * 100 // 200)  # no run can reach hard: max delta wins
    s.budget_tokens = s.policy.budget_tokens
    with pytest.raises(BudgetExceeded):  # BUILD-V2b: after that fold nothing else is feasible and the window is still mandatory
        s.call()
    s.policy = BackstopPolicyV1(budget_tokens=10**7)
    s.budget_tokens = s.policy.budget_tokens
    s.call()
    (fold,) = s.state.folds.values()
    assert fold.trigger == "backstop" and fold.policy["zone"] == "mandatory" and len(fold.victims) >= 2
    events = ledger.events()
    ledger.close()
    return Scenario(events, blobs, spans)


def test_unmutated_backstop_v1_ledger_reports_nothing(backstop_scenario):
    assert check(backstop_scenario.events, backstop_scenario.blobs) == []


def test_p9_v1_changed_rhat(backstop_scenario):
    i = index_of(backstop_scenario.events, "fold")
    events = mutate(backstop_scenario.events, i, lambda p: p["fold"]["policy"].update(rhat=p["fold"]["policy"]["rhat"] + 1))
    assert props(events, backstop_scenario.blobs) == {"P9"}


def test_p9_v1_changed_victims(backstop_scenario):
    """Dropping a victim also changes what the fold frees (P12), leaves a sentence citing a span
    that is no longer a victim (P3), and changes the window every later call saw (P2, P4)."""
    i = index_of(backstop_scenario.events, "fold")
    events = mutate(backstop_scenario.events, i, lambda p: p["fold"]["victims"].pop())
    assert props(events, backstop_scenario.blobs) == {"P9", "P3", "P12", "P2", "P4"}
