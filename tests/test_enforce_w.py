"""BUILD-V2b item 2: enforce W. Before a call the backstop keeps folding while the window is in the
discretionary or mandatory zone, until it is at or below hard; completed tool pairs are foldable
except the current turn's last tail_keep items; the current question never is; with nothing
feasible in the mandatory zone the answered turns' citations stop protecting, and only then
BudgetExceeded. B4 is unchanged."""

from __future__ import annotations

import hashlib
import json

import pytest

from context_ledger.backstop import BudgetExceeded
from context_ledger.fold import BackstopPolicyV1
from context_ledger.invariants import check
from context_ledger.model import Citation
from context_ledger.pricing import PricingTable
from context_ledger.render import rendered_window_tokens
from context_ledger.store import Ledger, SqliteBlobStore
from m2util import MODULE, MODULE_PATH
from test_one_w_measure import HAIKU, session, v3

TABLE = PricingTable.verified_default()


def reads(budget: int = 100_000, *, emit: bool = False):
    """Five completed corpus_read pairs of two spans each, a call, and a claim (turn t0) citing a
    span of the third pair; with `emit`, t0's answer is emitted. Then a new question."""
    s = session(budget)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = s.register_python_spans(src, MODULE_PATH)
    s.append_raw("user", "What is the default timeout?")
    for n in range(5):
        tid = f"toolu_{n:03d}"
        s.append_blocks([{"type": "text", "text": "reading"}, {"type": "tool_use", "id": tid, "name": "corpus_read", "input": {"n": n}}])
        for span in spans[2 * n : 2 * n + 2]:
            s.append_span(span.span_id, tool_use_id=tid)
        s.append_raw("tool", "read 2 spans", tool_use_id=tid)
    s.call()
    cited = spans[4]
    claim, _ = s.claim("It is cited.", [Citation(s.state.aliases.alias_of(cited.span_id), s.blobs.get(cited.text_hash).decode()[:40])], turn_id="t0")
    assert claim.status == "grounded"
    s.append_raw("assistant", "ANSWER: 37.0")
    if emit:
        s.emit_answer("t0", [claim.claim_id])
    s.append_raw("user", "Next: what is MAX_RETRIES?")
    return s, cited


def retune(s, budget: int) -> None:
    s.policy = BackstopPolicyV1(budget_tokens=budget, model=HAIKU, token_counter=s.counter.name)
    s.budget_tokens = budget


def test_one_call_folds_until_the_window_is_at_or_below_hard():
    s, cited = reads()
    L = rendered_window_tokens(s.state, s.blobs)
    retune(s, 900)
    assert L * 10000 > 9500 * 900  # mandatory, and no single run reaches hard (the cited pair splits the old reads)
    first = s.prepare_call()
    folds = s.last_folds
    assert len(folds) >= 2 and first is folds[0] and folds[0].policy["zone"] == folds[1].policy["zone"] == "mandatory"
    assert s.last_window == rendered_window_tokens(s.state, s.blobs) and s.last_window * 10000 <= 8500 * 900
    held = {seg.ref for seg in s.state.segments if seg.ref_kind == "span"}
    assert cited.span_id in held  # the open turn's citation still protects
    s.record_call(s._request())
    assert check(s.events, s.blobs) == []  # P9 re-runs every fold of the loop


def test_completed_pairs_are_foldable_but_the_current_turns_tail_and_question_are_not():
    s, _ = reads()
    model = s.payoff_model()
    n = len(s.state.segments)
    assert model.why[n - 1] == "current_question" or model.why[n - 1] == "tail"
    # a completed pair appended after the latest call, followed by more of the turn, is foldable
    tid = "toolu_late"
    s.append_blocks([{"type": "tool_use", "id": tid, "name": "corpus_read", "input": {"n": 9}}])
    s.append_raw("tool", "read 0 spans", tool_use_id=tid)
    s.append_raw("user", "a note")
    s.append_raw("user", "another note")
    s.append_raw("user", "a last note")
    why = s.payoff_model().why
    m = len(s.state.segments)
    assert why[m - 5] is None and why[m - 4] is None  # the late pair: not "recent"
    assert why[m - 3] == "recent" and why[m - 2:] == ["tail", "tail"]  # plain messages since the call stay recent
    question = next(p for p, seg in enumerate(s.state.segments) if s.payoff_model().current_question == seg.segment_id)
    assert why[question] == "current_question"


def test_a_turn_ended_without_an_answer_call_leaves_its_last_pair_foldable():
    s, _ = reads()
    s.call()
    tid = "toolu_last"
    s.append_blocks([{"type": "tool_use", "id": tid, "name": "corpus_read", "input": {"n": 7}}])
    s.append_raw("tool", "read 0 spans", tool_use_id=tid)
    claim = s.abstain("abstained: budget_exceeded", turn_id="t1", reason="budget_exceeded")
    s.emit_answer("t1", [claim.claim_id], abstained=True)
    s.append_raw("user", "A new question?")
    why = s.payoff_model().why
    n = len(s.state.segments)
    assert why[n - 1] in ("tail", "current_question") and why[n - 3] is None and why[n - 2] is None  # tail_keep counts the new turn only


def test_answered_citations_are_waived_only_when_nothing_else_is_feasible():
    s, cited = reads(emit=True)
    retune(s, 300)
    with pytest.raises(BudgetExceeded):
        s.prepare_call()
    waived = [f for f in s.last_folds if f.policy.get("cited_waived")]
    assert waived and all(f.policy["zone"] == "mandatory" for f in s.last_folds)
    assert s.last_folds.index(waived[0]) > 0  # only after every other feasible run was folded
    assert cited.span_id not in {seg.ref for seg in s.state.segments if seg.ref_kind == "span"}
    assert check(s.events, s.blobs) == []

    open_turn, cited = reads(emit=False)
    retune(open_turn, 300)
    with pytest.raises(BudgetExceeded):
        open_turn.prepare_call()
    assert not any(f.policy.get("cited_waived") for f in open_turn.last_folds)
    assert cited.span_id in {seg.ref for seg in open_turn.state.segments if seg.ref_kind == "span"}
    assert check(open_turn.events, open_turn.blobs) == []


# An L episode whose reads add up to 3 W never sends a request above mandatory W.


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from context_ledger.eval.corpus import build_corpus
    from context_ledger.eval.questions import build_questions
    from pilot_fixture import write_fixture_tarball

    tmp = tmp_path_factory.mktemp("enforce-w")
    tar, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    return manifest, build_questions(manifest, seed=9, episodes=2, runs_dir=tmp / "runs")


def run(tmp_path, corpus, arms: list[str], window: int, episodes: int = 1):
    from context_ledger.eval.runner import offline_client, run_pilot

    manifest, rows = corpus
    client, guard, _ = offline_client(tmp_path, run_id="e", pricing=TABLE, budget_usd="0.75")
    try:
        return run_pilot(manifest=manifest, rows=rows, arms=arms, rho=10, model=HAIKU, window=window, episodes=episodes, per_episode=8, smoke=False,
                         run_dir=tmp_path, run_id="e", pricing=TABLE, estimator=v3(), client=client, live=False, budget_usd="0.75")  # fmt: skip
    finally:
        guard.close()


W = 1200


def test_reads_of_three_windows_never_send_a_request_above_mandatory(tmp_path, corpus):
    summary = run(tmp_path, corpus, ["L"], W, episodes=2)
    tools, estimator = summary["tool_definitions"]["tokens"]["L"], v3()
    read, calls, folds_before = 0, [], []
    for db in sorted((tmp_path / "L").glob("episode-*.sqlite")):
        events, blobs = Ledger(db).events(), SqliteBlobStore(db)
        read += sum(estimator.count(blobs.get(e.payload["segment"]["text_hash"]).decode(), "python")
                    for e in events if e.type == "segment_appended" and e.payload["segment"]["ref_kind"] == "span")  # fmt: skip
        calls += [e.payload for e in events if e.type == "model_call"]
        count = 0
        for e in events:
            if e.type == "fold" and e.payload["fold"]["trigger"] == "backstop":
                count += 1
            elif e.type == "call":
                folds_before.append(count)
                count = 0
        assert check(events, blobs) == []
    assert read >= 3 * W
    assert calls and all(c["predicted"]["prompt"] - tools == c["predicted"]["window"] for c in calls)
    assert max(c["predicted"]["window"] for c in calls) * 10000 <= 9500 * W
    assert max(folds_before) >= 2  # several folds before one call
    results = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert sum(r["reason"] == "answered" for r in results) >= 8  # held to W, L still answers


# B4's transcript and results at W = 1500 (17 compactions). BUILD-V3 left the transcript byte-identical
# and every earlier results field unchanged; each results row gained the keys condition,
# forced_violation, forced_fallback and L's gate_reason, gate_target, gate_reasons and cleared_failures.
# BUILD-KEY added key_profile to every results row and nothing else: the hash is taken with it removed.
B4_SHA256 = "9982daa294aa10558843427c9cc7b60c4d174da3e3d51040cb33f5c746eea979"


def test_b4_is_unchanged(tmp_path, corpus):
    run(tmp_path, corpus, ["B4"], 1500, episodes=2)
    digest = hashlib.sha256()
    for path in sorted((tmp_path / "B4").glob("*.jsonl")):
        digest.update(path.read_bytes())
    results = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert {r.pop("key_profile") for r in results} == {"research"}
    digest.update("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in results).encode("utf-8"))
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert sum(c["kind"] == "summary" for c in calls) == 17 and digest.hexdigest() == B4_SHA256
