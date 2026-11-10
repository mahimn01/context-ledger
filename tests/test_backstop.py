"""Payoff-gated backstop selection (A.3 to A.6)."""

from __future__ import annotations

import time
from fractions import Fraction

import pytest
from hypothesis import event, example, given, settings
from hypothesis import strategies as st

from context_ledger.backstop import (
    BudgetExceeded,
    FoldRejected,
    PayoffModel,
    rstar,
    s_cache,
    select_backstop_v1,
    zone_of,
)
from context_ledger.fold import (
    BackstopPolicyV1,
    ReReadEstimator,
    extractive_summary_v1,
    fold_id_for,
    payback_calls,
    plan_fold_aliases,
    refs_of,
    render_fold,
)
from context_ledger.invariants import check
from context_ledger.model import Citation, Event, SummarySentence
from context_ledger.render import APPROX, render_segment
from context_ledger.window import approx_tokens
from m2util import pricing, v1_session

TABLE = pricing()


# A.3 cost model


def test_formulas_by_hand():
    # L=1000, P_i=200, delta=300, p_w=125000, p_r=10000, R-hat=4: the next call rewrites 500 tokens.
    assert s_cache(4, 300, 1000, 200, 125000, 10000) == 4 * 300 * 10000 - 500 * 115000
    assert s_cache(4, 300, 1000, 200, 125000, 10000, cold=True) == 4 * 300 * 10000
    assert rstar(300, 1000, 200, 125000, 10000) == Fraction(500 * 115000, 300 * 10000) - 1
    assert rstar(300, 1000, 200, 125000, 10000, cold=True) == -1
    assert rstar(0, 1000, 200, 125000, 10000) is None
    assert rstar(300, 1000, 200, 125000, 10000) == payback_calls(1000, 200, 300, 125000, 10000)
    assert payback_calls(32000, 4000, 8000, Fraction(5, 4), Fraction(1, 10)) == Fraction(111, 4)


@settings(max_examples=300, deadline=None)
@given(
    rhat=st.integers(1, 200),
    delta=st.integers(1, 10_000),
    rewrite=st.integers(0, 50_000),
    p_r=st.integers(1, 10**6),
    extra=st.integers(0, 10**7),
    G=st.integers(0, 10**12),
    cold=st.booleans(),
)
def test_sign_of_saving_matches_break_even(rhat, delta, rewrite, p_r, extra, G, cold):
    """sign(S_cache) == sign(R-hat - 1 - R*) whenever delta > 0."""
    p_w = p_r + extra
    P_i = 100
    L = P_i + delta + rewrite
    s = s_cache(rhat, delta, L, P_i, p_w, p_r, G, cold)
    r = rstar(delta, L, P_i, p_w, p_r, G, cold)
    sign = lambda x: (x > 0) - (x < 0)  # noqa: E731
    assert sign(s) == sign(rhat - 1 - r)


def test_zones_use_integer_basis_points():
    policy = BackstopPolicyV1(budget_tokens=1000)
    assert [zone_of(u, policy) for u in (700, 701, 850, 851, 950, 951)] == [
        "normal", "soft", "soft", "discretionary", "discretionary", "mandatory",
    ]  # fmt: skip


# Scenarios through a session


def window(n: int = 6, calls_between: bool = True, size: int = 400, **policy):
    """A v1 session (budget large enough that nothing folds yet) with a pinned system prompt
    and n multi-line note spans, with a call after each but the last when `calls_between`."""
    s = v1_session(10**7, **policy)
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    lines = [f"section {k}: " + "lorem ipsum dolor sit amet " * (size // 27) for k in range(n)]
    src = s.register_source("notes.txt", "\n".join(lines) + "\n")
    start = 0
    segs = []
    for k, line in enumerate(lines):
        span = s.register_text_span(src, start, start + len(line))
        segs.append(s.append_span(span.span_id))
        start += len(line) + 1
        if calls_between and k < n - 1:
            s.call()
    return s, segs


def retune(s, percent: int, **policy) -> BackstopPolicyV1:
    """Give the session a budget so that its window is `percent`% of it."""
    used = sum(approx_tokens(render_segment(s.state, g, s.blobs)) for g in s.state.segments)
    s.policy = BackstopPolicyV1(budget_tokens=used * 100 // percent, **policy)
    s.budget_tokens = s.policy.budget_tokens
    return s.policy


def test_discretionary_rejects_until_reads_justify_then_folds():
    """With one call behind it R-hat (lindy) is 1 and the risk term wins; as calls accumulate
    R-hat grows, the victims age, and a fold pays."""
    s, segs = window(calls_between=False)
    s.call()
    retune(s, 90)
    analysis = s.payoff_model().analysis()
    assert analysis.zone == "discretionary" and analysis.rhat == 1
    assert analysis.runs and all(r.J <= 0 for r in analysis.runs) and analysis.decision is None
    assert s.prepare_call() is None
    for _ in range(12):
        s.append_raw("user", "ok")
        s.call()
        if s.state.folds:
            break
    (fold,) = s.state.folds.values()
    assert fold.trigger == "backstop" and fold.policy["zone"] == "discretionary"
    assert Fraction(*map(int, fold.policy["J"].split("/"))) > 0 and fold.policy["rhat"] > 1
    assert fold.freed_tokens == fold.policy["delta"] > 0
    assert check(s.events, s.blobs) == []


def test_discretionary_prefers_runs_reaching_target():
    s, segs = window(size=600)
    retune(s, 90, estimator=ReReadEstimator("fixed", 50))
    analysis = s.payoff_model().analysis()
    assert analysis.zone == "discretionary"
    reach = lambda r: (analysis.used - r.delta) * 10000 <= 5000 * s.budget_tokens  # noqa: E731
    positive = [r for r in analysis.runs if r.J > 0]
    best = analysis.decision
    assert any(reach(r) for r in positive) and any(not reach(r) for r in positive)
    chosen = next(r for r in analysis.runs if (r.i, r.j) == (best.i, best.j))
    assert reach(chosen) and chosen.J == max(r.J for r in positive if reach(r))


def test_mandatory_waives_payback_and_prefers_reaching_hard():
    s, segs = window()
    retune(s, 110)
    analysis = s.payoff_model().analysis()
    assert analysis.zone == "mandatory" and any(r.J <= 0 for r in analysis.runs)
    decision = analysis.decision
    hard = [r for r in analysis.runs if (analysis.used - r.delta) * 10000 <= 8500 * s.budget_tokens]
    if hard:
        assert (decision.i, decision.j) in {(r.i, r.j) for r in hard}
        assert decision.J == max(r.J for r in hard)
    else:
        assert decision.delta == max(r.delta for r in analysis.runs)
    record = s.prepare_call()
    assert record.trigger == "backstop" and record.policy["zone"] == "mandatory"
    assert check(s.events, s.blobs) == []


def test_mandatory_falls_back_to_the_largest_delta():
    s, segs = window()
    retune(s, 200)  # no run can bring the window down to hard
    analysis = s.payoff_model().analysis()
    assert not [r for r in analysis.runs if (analysis.used - r.delta) * 10000 <= 8500 * s.budget_tokens]
    top = max(r.delta for r in analysis.runs)
    assert analysis.decision.delta == top
    assert analysis.decision.J == max(r.J for r in analysis.runs if r.delta == top)


def test_mandatory_without_a_feasible_run_raises_and_records_nothing():
    s, segs = window(n=3, calls_between=False)  # nothing was called yet: every segment is recent
    retune(s, 110)
    before = list(s.events)
    with pytest.raises(BudgetExceeded) as info:
        s.call()
    assert info.value.used * 100 >= 105 * s.budget_tokens and info.value.reason == "no_feasible_run"
    assert s.events == before


def test_feasibility_exclusions_and_cited_spans():
    s, segs = window(tail_keep=2, cold_m=2)
    s.call()
    model = s.payoff_model()
    assert model.why[0] == "pinned" and model.why[-2:] == ["tail", "tail"]
    s.claim("section 1", [Citation(s.state.aliases.alias_of(segs[1].ref), s.blobs.get(segs[1].text_hash).decode()[:40])])
    s.append_raw("user", "fresh")
    model = s.payoff_model()
    assert model.why[2] == "cited" and model.why[-1] == "tail"
    s.call()
    s.call()
    s.call()  # the claim's call is no longer among the last cold_m calls
    assert s.payoff_model().why[2] is None


def test_cache_cold_rule_and_risk_ageing():
    s, segs = window(n=4)
    s.call()
    events = list(s.events)
    cold = PayoffModel(s.state, events, s.blobs, s.policy, TABLE, APPROX)
    assert cold.cache_cold
    warm_events = events + [Event(len(events) + 1, "model_call", {"usage": {"cache_read_input_tokens": 900}}, "", "")]
    warm = PayoffModel(s.state, warm_events, s.blobs, s.policy, TABLE, APPROX)
    assert not warm.cache_cold
    run_cold, run_warm = cold.evaluate(1, 2, 50), warm.evaluate(1, 2, 50)
    assert run_cold.J - run_warm.J == (cold.L - cold.prefix[1] - 50) * (cold.p_w - cold.p_r)
    # Risk halves every half_life_calls calls since the segment was last touched.
    q = lambda age: Fraction(2000, 10000) / 2 ** (age // 4)  # noqa: E731
    for _ in range(8):
        s.call()
    aged = s.payoff_model()
    risk = Fraction(aged.risk[2] - aged.risk[1], aged.D)
    assert risk == q(4 + 8) * (aged.tok[1] + 150) * aged.p_w  # appended before 4 of the calls


def test_gated_model_fold_errors():
    s, segs = window(estimator=ReReadEstimator("fixed", 1))
    s.call()
    alias = s.state.aliases.alias_of
    with pytest.raises(FoldRejected) as info:
        s.gated_fold([segs[1].segment_id, segs[3].segment_id], [SummarySentence("x", ())])
    assert info.value.reason == "infeasible_run" and info.value.details["why"] == "not_contiguous"
    with pytest.raises(FoldRejected) as info:
        s.gated_fold([segs[-1].segment_id], [SummarySentence("x", ())])
    assert info.value.details == {"position": 6, "why": "tail"}
    with pytest.raises(FoldRejected) as info:
        s.gated_fold([segs[0].segment_id], [SummarySentence("y" * 2000, ())])
    assert info.value.reason == "fold_would_grow_window" and info.value.details["delta"] <= 0
    with pytest.raises(FoldRejected) as info:  # R-hat = 1 and risk: no payback below mandatory
        s.gated_fold([segs[0].segment_id, segs[1].segment_id], [SummarySentence("two sections", (segs[0].ref,))])
    assert info.value.reason == "payback_not_met" and info.value.details["J"] <= 0 and info.value.details["rhat"] == 1
    events = len(s.events)
    rich, segs = window(estimator=ReReadEstimator("fixed", 64))
    rich.call()
    fold = rich.gated_fold([segs[0].segment_id, segs[1].segment_id], [SummarySentence("two sections", (segs[0].ref,))])
    assert fold.trigger == "model" and fold.freed_tokens > 0 and len(s.events) == events
    assert alias(segs[0].ref) and check(rich.events, rich.blobs) == []


# The optimizer against an independent, naive, all-Fraction reference


def reference(state, events, blobs, policy, table):
    """Brute force: every run, full render each time, every quantity a Fraction. Random windows
    hold no tool blocks and no answered turns, so BUILD-V2b's completed-pair and cited-waiver
    rules never apply here (test_enforce_w.py covers them)."""
    price = table.price(policy.model)
    p_w, p_r = Fraction(price.cache_write(policy.ttl)), Fraction(price.cache_read)
    segs = state.segments
    tok = [approx_tokens(render_segment(state, g, blobs)) for g in segs]
    L = sum(tok)
    calls = [e for e in events if e.type == "call"]
    recent = {e.payload["call_id"] for e in calls[max(0, len(calls) - policy.cold_m) :]} if policy.cold_m else set()
    cited = {
        c["ref"]
        for e in events
        if e.type == "claim" and e.payload["claim"]["call_id"] in recent
        for c in e.payload["claim"]["citations"]
    }
    entered, touch, n_calls, last_read = {}, {}, 0, None
    current_question, awaiting = None, True
    for e in events:
        if e.type == "call":
            n_calls += 1
        elif e.type == "answer_emitted":
            awaiting = True
        elif e.type == "segment_appended":
            entered[e.payload["segment"]["segment_id"]] = (e.seq, n_calls)
            if awaiting and e.payload["segment"]["role"] == "user":
                current_question, awaiting = e.payload["segment"]["segment_id"], False
        elif e.type == "fold":
            entered[e.payload["fold"]["summary_segment_id"]] = (e.seq, n_calls)
        elif e.type == "claim":
            for c in e.payload["claim"]["citations"]:
                touch[c["ref"]] = n_calls
        elif e.type == "refetch":
            for r in (e.payload["span_id"], e.payload["new_span_id"]):
                if r:
                    touch[r] = n_calls
        elif e.type == "model_call":
            last_read = e.payload["usage"]["cache_read_input_tokens"]
    cold = not last_read
    # tail_keep counts only the current question's items (BUILD-V2b): a segment before the
    # current question is never tail, however close to the end of the window it is
    q_pos = next((p for p, g in enumerate(segs) if g.segment_id == current_question), 0)

    def foldable(p):
        g = segs[p]
        return (
            not g.pinned
            and g.role != "system"
            and g.segment_id != current_question
            and (p < len(segs) - policy.tail_keep or p < q_pos)
            and bool(calls)
            and entered[g.segment_id][0] <= calls[-1].seq
            and not (refs_of(g, state) & cited)
        )

    def age(p):
        g = segs[p]
        return n_calls - max([entered[g.segment_id][1]] + [touch[r] for r in refs_of(g, state) if r in touch])

    rhat = policy.estimator.rhat(n_calls)
    budget = Fraction(policy.budget_tokens)
    frac = lambda bp: Fraction(bp, 10000) * budget  # noqa: E731
    zone = "mandatory" if L > frac(policy.mandatory_bp) else "discretionary" if L > frac(policy.hard_bp) else "other"
    runs = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs) + 1):
            if not all(foldable(p) for p in range(i, j)):
                continue
            victims = [g.segment_id for g in segs[i:j]]
            sentences = extractive_summary_v1(state, victims, blobs)
            fold_id = fold_id_for("backstop", i, victims)
            text = render_fold(fold_id, sentences, plan_fold_aliases(state.aliases, fold_id, sentences), state.aliases)
            delta = sum(tok[i:j]) - approx_tokens(text)
            if delta <= 0:
                continue
            P_i = sum(tok[:i])
            rewrite = 0 if cold else L - P_i - delta
            S = rhat * delta * p_r - rewrite * (p_w - p_r)
            risk = sum(
                Fraction(policy.q0_bp, 10000) / 2 ** (age(p) // policy.half_life_calls) * (tok[p] + policy.tau_tokens) * p_w
                for p in range(i, j)
            )
            rs = rewrite * (p_w - p_r) / (delta * p_r) - 1
            runs.append(dict(i=i, j=j, delta=delta, J=S - risk, rstar=rs, P_i=P_i))
    key = lambda r: (r["J"], r["i"], -(r["j"] - r["i"]), -r["j"])  # noqa: E731
    if zone == "discretionary":
        positive = [r for r in runs if r["J"] > 0]
        pool = [r for r in positive if L - r["delta"] <= frac(policy.target_bp)] or positive
        best = max(pool, key=key, default=None)
    elif zone == "mandatory":
        pool = [r for r in runs if L - r["delta"] <= frac(policy.hard_bp)]
        best = max(pool, key=key) if pool else max(runs, key=lambda r: (r["delta"], *key(r)), default=None)
    else:
        best = None
    if best is None:
        return None
    return dict(zone=zone, L=L, rhat=rhat, cache_cold=cold, victims=tuple(g.segment_id for g in segs[best["i"] : best["j"]]), **best)


@st.composite
def random_window(draw):
    s = v1_session(10**7)
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    sizes = draw(st.lists(st.integers(1, 260), min_size=2, max_size=14))
    units = [f"unit {k}\n" + "abc defgh " * (n // 10) + "z" * (n % 10) + "\n" for k, n in enumerate(sizes)]
    src = s.register_source("doc.txt", "".join(units))
    start = 0
    for unit in units:
        kind = draw(st.sampled_from(["span", "span", "raw", "pinned"]))
        if kind == "raw":
            s.append_raw(draw(st.sampled_from(["user", "assistant", "tool"])), unit * draw(st.integers(1, 3)))
        else:
            span = s.register_text_span(src, start, start + len(unit) - 1)
            s.append_span(span.span_id, pinned=kind == "pinned")
        start += len(unit)
        if draw(st.booleans()):
            s.call()
        if s.state.calls and draw(st.integers(0, 4)) == 0:
            spans = [g for g in s.state.segments if g.ref_kind == "span"]
            if spans:
                g = draw(st.sampled_from(spans))
                s.claim("cited", [Citation(g.ref, s.blobs.get(g.text_hash).decode())])
        if draw(st.integers(0, 6)) == 0:
            free = [g.segment_id for g in s.state.segments if not g.pinned]
            if len(free) >= 2:
                try:
                    s.fold(free[:2], [SummarySentence("folded", ())])
                except Exception:
                    pass
    if draw(st.booleans()) and s.state.calls:
        s.call()
    return s


def found_v2b_window():
    """The window Hypothesis found after BUILD-V2b (with budget_scale 84, tail_keep 2, cold_m 0):
    the span before the current question is among the last tail_keep segments, yet foldable,
    because tail_keep counts only the current question's items."""
    s = v1_session(10**7)
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    src = s.register_source("doc.txt", "unit 0\nz\nunit 1\nz\nunit 2\nabc defgh zzzzzzzzz\nunit 3\nz\n")
    first = [s.append_span(s.register_text_span(src, a, b).span_id).segment_id for a, b in ((0, 8), (9, 17))]
    s.fold(first, [SummarySentence("folded", ())])
    g = s.append_span(s.register_text_span(src, 18, 44).span_id)
    s.call()
    cite = [Citation(g.ref, s.blobs.get(g.text_hash).decode())]
    s.claim("cited", cite)
    s.append_raw("user", "unit 3\nz\n")
    s.claim("cited", cite)
    return s


@settings(max_examples=120, deadline=None)
@given(
    s=random_window(),
    budget_scale=st.one_of(st.integers(55, 140), st.integers(86, 95)),
    cold_m=st.integers(0, 3),
    tail_keep=st.integers(0, 3),
    q0_bp=st.integers(0, 10000),
    half_life=st.integers(1, 5),
    tau=st.integers(0, 300),
    estimator=st.sampled_from(
        [ReReadEstimator("fixed", 3), ReReadEstimator("fixed", 80), ReReadEstimator("lindy", 4), ReReadEstimator("horizon", 9), ReReadEstimator()]
    ),
    ttl=st.sampled_from(["5m", "1h"]),
    model=st.sampled_from(["claude-haiku-5-5", "claude-sonnet-5-5"]),
    model_call=st.sampled_from([None, 0, 500]),
)
@example(
    s=found_v2b_window(), budget_scale=84, cold_m=0, tail_keep=2, q0_bp=0, half_life=1, tau=0,
    estimator=ReReadEstimator("fixed", 3), ttl="5m", model="claude-haiku-5-5", model_call=None,
)  # fmt: skip
def test_optimizer_matches_naive_reference(s, budget_scale, cold_m, tail_keep, q0_bp, half_life, tau, estimator, ttl, model, model_call):
    used = sum(approx_tokens(render_segment(s.state, g, s.blobs)) for g in s.state.segments)
    budget = max(1, used * 100 // budget_scale)
    policy = BackstopPolicyV1(
        budget_tokens=budget, cold_m=cold_m, tail_keep=tail_keep, q0_bp=q0_bp, half_life_calls=half_life,
        tau_tokens=tau, estimator=estimator, ttl=ttl, model=model,
    )  # fmt: skip
    events = list(s.events)
    if model_call is not None:
        events.append(Event(len(events) + 1, "model_call", {"usage": {"cache_read_input_tokens": model_call}}, "", ""))
    got = select_backstop_v1(s.state, events, s.blobs, policy, TABLE, APPROX)
    want = reference(s.state, events, s.blobs, policy, TABLE)
    event(f"zone {zone_of(used, policy)}: {'fold' if want else 'no fold'}")
    if want is None:
        assert got is None
        return
    assert got is not None
    assert (got.zone, got.L, got.P_i, got.i, got.j, got.delta, got.rhat, got.cache_cold) == (
        want["zone"], want["L"], want["P_i"], want["i"], want["j"], want["delta"], want["rhat"], want["cache_cold"],
    )  # fmt: skip
    assert (got.J, got.rstar, got.victims) == (want["J"], want["rstar"], want["victims"])


def test_four_hundred_segments_select_quickly():
    s = v1_session(10**7)
    s.append_raw("system", "Answer only from cited spans.", pinned=True)
    rows = [f"row {k}\n" + "data " * (5 + k % 17) + "\n" for k in range(399)]  # two lines each
    src = s.register_source("big.txt", "".join(rows))
    start = 0
    for row in rows:
        s.append_span(s.register_text_span(src, start, start + len(row) - 1).span_id)
        start += len(row)
    s.call()
    used = sum(approx_tokens(render_segment(s.state, g, s.blobs)) for g in s.state.segments)
    policy = BackstopPolicyV1(budget_tokens=used * 100 // 97, estimator=ReReadEstimator("fixed", 40))
    began = time.perf_counter()
    decision = select_backstop_v1(s.state, s.events, s.blobs, policy, TABLE, APPROX)
    elapsed = time.perf_counter() - began
    assert len(s.state.segments) == 400 and decision is not None and decision.zone == "mandatory"
    assert elapsed < 20, elapsed


def test_the_current_question_is_never_a_fold_victim():
    """Found in the M4 live re-smoke: a backstop fold took the current question with it and the
    model then answered an earlier question. The first user segment since the last
    answer_emitted is protected; earlier, answered questions stay foldable."""
    s, segs = window(n=6)
    s.append_raw("user", "first question?")
    first = s.state.segments[-1].segment_id
    s.call()
    s.emit_answer("q1", [], abstained=True)
    s.append_raw("user", "second question?")
    second = s.state.segments[-1].segment_id
    for k in range(4):
        s.append_raw("tool", f"result {k} " + "lorem ipsum " * 40)
        s.call()
    why = dict(zip((g.segment_id for g in s.state.segments), s.payoff_model().why))
    assert why[second] == "current_question" and why[first] is None
    retune(s, 110, tail_keep=0, cold_m=0)
    analysis = s.payoff_model().analysis()
    assert analysis.zone == "mandatory" and analysis.runs
    model = s.payoff_model()
    assert all(second not in model.victims(r.i, r.j) for r in analysis.runs)
    assert any(first in model.victims(r.i, r.j) for r in analysis.runs)
    record = s.prepare_call()
    assert record.trigger == "backstop" and second not in record.victims
    assert second in {g.segment_id for g in s.state.segments}
    assert check(s.events, s.blobs) == []
