"""BUILD-V2c commit 1: L's floor stays bounded (backstop-v1 rules "v2c").

Pilot v2b (runs/pilot-v2b-haiku-w3000, rho 10) had L abstain with budget_exceeded on 36 of its
40 questions, 30 of them before any call. A turn refused by BudgetExceeded makes no call, so "recent" never
expired for the backstop's own folds and the later questions; and one summary line per victim
is as large as a one-line span and larger than a short question, so such runs never shrank.
Under v2c "recent" protects the current turn only and never a fold, a refold line names the
earlier fold by one alias, and in the mandatory zone a digest line (the summary-of-summaries)
is the last resort. Invariant: L returns budget_exceeded before its first model call on a
question only when the system prompt, the question and the tool-free fixed parts alone are
over hard."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from pathlib import Path

import pytest

from context_ledger.backstop import BudgetExceeded
from context_ledger.fold import BackstopPolicyV1, digest_sentence, extractive_summary_v1, fold_delta, fold_id_for, plan_fold_aliases, policy_from_dict, render_fold
from context_ledger.invariants import check
from context_ledger.model import Event, SummarySentence
from context_ledger.pricing import PricingTable
from context_ledger.render import rendered_window_tokens, tool_blocks
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore, SqliteBlobStore
from context_ledger.tools import dispatch
from context_ledger.window import down, kappa, replay
from pilot_fixture import TOKEN_NAMES, token_table
from test_one_w_measure import HAIKU, v3

TABLE = PricingTable.verified_default()
ROOT = Path(__file__).resolve().parents[1]
TOKENTAB = "Lib/tokentab.py"


def session(budget: int, rules: str) -> Session:
    estimator = v3()
    policy = BackstopPolicyV1(budget_tokens=budget, model=HAIKU, token_counter=estimator.name, rules=rules)
    s = Session(Ledger(":memory:"), MemoryBlobStore(), budget, policy, pricing=TABLE, tokenizer=estimator)
    s.append_raw("system", "Answer only from cited spans. " * 20, pinned=True)
    return s


def retune(s: Session, budget: int) -> None:
    s.policy = BackstopPolicyV1(budget_tokens=budget, model=HAIKU, token_counter=s.counter.name, rules=s.policy.rules)
    s.budget_tokens = budget


def read_turn(s: Session, question: str, n: int = 0) -> None:
    """A question, a call, and the model's corpus_read of the token table: every one-line
    constant a span of the one tool result, then the receipt."""
    s.append_raw("user", question)
    s.call()
    tid = f"toolu_{n:03d}"
    s.append_blocks([{"type": "text", "text": "Reading the token table."}, {"type": "tool_use", "id": tid, "name": "corpus_read", "input": {"n": n}}])
    source = next((src for src in s.state.sources.values() if src.uri == TOKENTAB), None) or s.register_source(TOKENTAB, token_table(), media_type="text/x-python")
    for span in s.register_python_spans(source, TOKENTAB):
        s.append_span(span.span_id, tool_use_id=tid)
    s.append_raw("tool", f"read {TOKENTAB}:1-70", tool_use_id=tid)


def refused(s: Session, turn: str) -> None:
    """What the client does when BudgetExceeded refuses the turn's call: abstain, emit."""
    claim = s.abstain("abstained: budget_exceeded", turn_id=turn, reason="budget_exceeded")
    s.emit_answer(turn, [claim.claim_id], abstained=True)


def mandatory_after_a_read(rules: str) -> Session:
    """The pilot's episode 16 in miniature: the first question's read puts the window in the
    mandatory zone with the read itself protected, so the turn is refused."""
    s = session(100_000, rules)
    read_turn(s, "What is the value of the module constant `tokentab.RSQB`?")
    retune(s, rendered_window_tokens(s.state, s.blobs) * 100 // 97)
    with pytest.raises(BudgetExceeded):
        s.prepare_call()
    refused(s, "q0")
    return s


def test_v2b_locks_the_window_after_a_refused_turn_and_v2c_does_not():
    old = mandatory_after_a_read("v2b")
    old.append_raw("user", "What is the value of the module constant `tokentab.COMMA`?")
    with pytest.raises(BudgetExceeded):
        old.prepare_call()
    refused(old, "q1")
    old.append_raw("user", "What is the value of the module constant `tokentab.SEMI`?")
    with pytest.raises(BudgetExceeded):
        old.prepare_call()
    assert old.last_folds == []  # no line-per-victim fold of tiny spans frees tokens, and the rest is "recent": for good
    assert old.payoff_model().analysis().runs == ()

    s = mandatory_after_a_read("v2c")
    s.append_raw("user", "What is the value of the module constant `tokentab.COMMA`?")
    s.prepare_call()  # no BudgetExceeded: the refused turn and the backstop's folds are foldable
    assert s.last_window * 10000 <= 8500 * s.budget_tokens
    assert any(f.policy.get("digest") for f in s.last_folds)
    s.record_call(s._request())
    assert check(s.events, s.blobs) == []  # P3, P9, P10, P12 and P13 hold for the digest


def test_recent_protects_only_the_current_turn_and_never_a_fold():
    s = mandatory_after_a_read("v2c")
    s.append_raw("user", "What is the value of the module constant `tokentab.COMMA`?")
    why = s.payoff_model().why
    assert why[0] == "pinned" and why[-1] == "tail" and all(w is None for w in why[1:-1])
    s.prepare_call()
    folded = [p for p, seg in enumerate(s.state.segments) if seg.ref_kind == "fold"]
    s.append_raw("user", "a harness note appended after the folds")
    s.append_raw("user", "another note")
    s.append_raw("user", "a last note")
    why = s.payoff_model().why
    assert folded and all(why[p] is None for p in folded)  # backstop folds made since the call
    assert why[-3] == "recent"  # the current turn's own new message still is


def test_a_digest_keeps_every_span_summary_only_and_refetchable():
    s = mandatory_after_a_read("v2c")
    s.append_raw("user", "What is the value of the module constant `tokentab.COMMA`?")
    s.prepare_call()
    digest = next(f for f in s.last_folds if f.policy.get("digest"))
    assert len(digest.sentences) == 1 and digest.sentences[0].text.startswith("earlier context: ")
    spans = {span_id for span_id in s.state.spans}
    reach = down(s.state, f"fold:{digest.fold_id}#0")
    assert reach == spans and all(kappa(s.state, span_id).value == "summary_only" for span_id in reach)
    text = s.blobs.get(digest.summary_hash).decode()
    assert len(text) < 400 and text.count("\n") == 1  # one line, one alias per span
    line_alias = s.state.aliases.alias_of(f"fold:{digest.fold_id}#0")
    s.call()
    s.append_blocks([{"type": "tool_use", "id": "toolu_r", "name": "ctx_refetch", "input": {"refs": [line_alias]}}])
    out = dispatch(s, "ctx_refetch", {"refs": [line_alias]}, turn_id="q1", tool_use_id="toolu_r")
    assert not out.is_error and out.text.count(" OK") == 8  # ctx_refetch brings back up to 8 spans behind it
    s.append_raw("tool", out.text, tool_use_id="toolu_r")
    assert check(s.events, s.blobs) == []


def test_a_refold_line_names_the_earlier_fold_by_one_alias():
    s = session(100_000, "v2c")
    read_turn(s, "What is the value of the module constant `tokentab.RSQB`?")
    s.call()
    spans = [seg.ref for seg in s.state.segments if seg.ref_kind == "span"]
    thirds = [SummarySentence(f"token constants, part {n}", tuple(spans[n::3])) for n in range(3)]
    first = s.fold([seg.segment_id for seg in s.state.segments[2:]], thirds, k=2)
    seg = next(seg for seg in s.state.segments if seg.ref == first.fold_id)
    for rules, cites in (("v2c", (f"fold:{first.fold_id}",)), ("v2b", tuple(f"fold:{first.fold_id}#{n}" for n in range(len(first.sentences))))):
        assert extractive_summary_v1(s.state, [seg.segment_id], s.blobs, rules=rules)[0].cites == cites
    refold = s.fold([seg.segment_id], extractive_summary_v1(s.state, [seg.segment_id], s.blobs, rules="v2c"))
    alias = s.state.aliases.alias_of(f"fold:{first.fold_id}")
    assert s.blobs.get(refold.summary_hash).decode().endswith(f"earlier fold {alias} ({len(first.sentences)} sentences) [{alias}]")
    assert down(s.state, f"fold:{refold.fold_id}#0") == down(s.state, f"fold:{first.fold_id}")
    assert check(s.events, s.blobs) == []  # a whole-fold cite of an earlier fold passes P3 and P10


def test_the_digest_delta_is_the_fold_delta_exactly():
    """Over a window of a digest fold, answered turns and a second read: every run a digest
    applies to frees exactly what make_fold would record."""
    s = mandatory_after_a_read("v2c")
    s.append_raw("user", "What is `tokentab.COMMA`?")
    s.prepare_call()
    s.record_call(s._request())
    s.append_raw("assistant", "ANSWER: 12")
    refused(s, "q1")
    read_turn(s, "What is `tokentab.PLUS`?", 1)
    s.append_raw("assistant", "ANSWER: 14")
    refused(s, "q2")
    s.append_raw("user", "What is `tokentab.STAR`?")
    model = s.payoff_model()
    blocks = tool_blocks(s.state, s.blobs)
    checked = 0
    for i in range(1, model.n):
        for j in range(i + 1, model.n + 1):
            if blocks[i][0] != i or blocks[j - 1][1] != j or not model.digest_applies(i, j):
                continue
            victims = s.state.segments[i:j]
            sentences = [digest_sentence(victims)]
            fold_id = fold_id_for("backstop", i, [v.segment_id for v in victims])
            text = render_fold(fold_id, sentences, plan_fold_aliases(s.state.aliases, fold_id, sentences), s.state.aliases)
            assert model.delta(i, j, True) == fold_delta(s.state, victims, text, s.blobs, aliased=True, k=i)
            checked += 1
    assert checked >= 30


def test_rules_are_recorded_only_when_they_are_v2c():
    old, new = BackstopPolicyV1(budget_tokens=3000), BackstopPolicyV1(budget_tokens=3000, rules="v2c")
    assert "rules" not in old.to_dict() and new.to_dict()["rules"] == "v2c"
    assert policy_from_dict(old.to_dict()) == old and policy_from_dict(new.to_dict()) == new
    with pytest.raises(ValueError):
        policy_from_dict({**old.to_dict(), "rules": "v2b"})
    with pytest.raises(ValueError):
        BackstopPolicyV1(budget_tokens=3000, rules="v9")


def test_budget_exceeded_before_a_call_means_system_and_question_alone_are_over_hard():
    s = mandatory_after_a_read("v2c")
    s.append_raw("user", "What is the value of the module constant `tokentab.COMMA`? " + "Please be careful. " * 400)
    with pytest.raises(BudgetExceeded):
        s.prepare_call()
    held = [seg for seg in s.state.segments if seg.pinned or seg.segment_id == s.payoff_model().current_question]
    assert len(s.state.segments) == len(held) + 1  # everything else is down to one line
    floor = replay([e for e in s.events if e.type in ("tokenizer_set", "pricing_set")])
    floor.segments = held
    assert rendered_window_tokens(floor, s.blobs) * 10000 > 8500 * s.budget_tokens


# The read cap counts the alias headers; the receipt lists at most 8 aliases.


def test_an_l_read_of_small_spans_costs_at_most_the_cap(tmp_path):
    from context_ledger.eval.tools import LedgerCorpusTools, RECEIPT_ALIASES
    from context_ledger.render import render_segment
    from types import SimpleNamespace

    index = SimpleNamespace(paths={TOKENTAB}, text=lambda path: token_table())
    tools = LedgerCorpusTools(index, cap=500)
    s = session(100_000, "v2c")
    s.append_raw("user", "What is the value of the module constant `tokentab.RSQB`?")
    s.call()
    s.append_blocks([{"type": "tool_use", "id": "toolu_1", "name": "corpus_read", "input": {}}])
    out = tools(s, "corpus_read", {"path": TOKENTAB, "start_line": 1, "end_line": 70}, turn_id="q0", tool_use_id="toolu_1")
    segments = [seg for seg in s.state.segments if seg.segment_id in set(out.appended)]
    rendered = sum(s.counter.count(render_segment(s.state, seg, s.blobs), "python") for seg in segments)
    assert 10 < len(segments) < len(TOKEN_NAMES) and rendered <= 500
    assert out.text.count("@") == RECEIPT_ALIASES and f"and {len(segments) - RECEIPT_ALIASES} more" in out.text and "truncated" in out.text


# An offline 20-question L episode at W = 2000 and W = 3000, the pilot's counter gap simulated.


INFLATE = (3, 2)  # live count_tokens counted ~1.45x calib-v3 on L's content in pilot v2b


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from context_ledger.eval.corpus import build_corpus
    from pilot_fixture import v2_fixture_files, write_fixture_tarball

    tmp = tmp_path_factory.mktemp("bounded-floor")
    tar, sha = write_fixture_tarball(tmp, {**v2_fixture_files(), TOKENTAB: token_table()})
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=400, rhos=(10, 20), n_gold=8, runs_dir=tmp / "runs", cache_dir=tmp / "cache",
                            progress=lambda m: None, version="v2")  # fmt: skip
    assert any(f["path"] == TOKENTAB and f["rho"] <= 20 for f in manifest["files"])
    texts = [f"What is the value of the module constant `tokentab.{name}`?" for name in TOKEN_NAMES[:14]]
    texts += [f"Which exception classes does `spool.drain_spool_step{k}` raise directly (in its own body)?" for k in range(3)]
    texts += [f"What is the default value of parameter `retries` of `garden.prune_garden_step{k}`?" for k in range(3)]
    random.Random(1).shuffle(texts)
    rows = [{"qid": f"bf{p:02d}", "episode": 1, "pos": p, "template": "constant", "text": t, "answerable": True, "fold_probe": False, "probe_of": None}
            for p, t in enumerate(texts)]  # fmt: skip
    return manifest, rows


def episode(tmp_path, corpus, window: int, monkeypatch=None, rules: str | None = None):
    import context_ledger.eval.arms as arms
    from context_ledger.eval.runner import offline_client, run_pilot

    if rules is not None:  # the old rules, for contrast
        monkeypatch.setattr(arms, "BackstopPolicyV1", lambda **kw: BackstopPolicyV1(**{**kw, "rules": rules}))
    manifest, rows = corpus
    client, guard, server = offline_client(tmp_path, run_id="bf", pricing=TABLE, budget_usd="2")
    counted = server.count_tokens
    server.count_tokens = lambda body: {"input_tokens": counted(body)["input_tokens"] * INFLATE[0] // INFLATE[1]}
    try:
        summary = run_pilot(manifest=manifest, rows=rows, arms=["L"], rho=20, model=HAIKU, window=window, episodes=1, per_episode=None, smoke=False,
                            run_dir=tmp_path, run_id="bf", pricing=TABLE, estimator=v3(), client=client, live=False, budget_usd="2",
                            questions_version="v2")  # fmt: skip
    finally:
        guard.close()
    results = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    return summary, results, calls, server


@pytest.mark.parametrize("window", [2000, 3000])
def test_a_20_question_episode_never_abstains_for_budget_before_a_call(tmp_path, corpus, window):
    from context_ledger.eval.prompts import system_prompt

    summary, results, calls, server = episode(tmp_path, corpus, window)
    assert len(results) == 20 and summary["l_mechanism"] == "l-v3"
    for r in results:  # the invariant, checked on the tool-free request of system prompt and question alone
        if r["reason"] == "budget_exceeded" and r["calls"] == 0:
            text = next(q["text"] for q in corpus[1] if q["qid"] == r["qid"])
            floor = server.count_tokens({"model": HAIKU, "system": [{"type": "text", "text": system_prompt("L", summary["prompts"]["version"])}],
                                         "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]})["input_tokens"]  # fmt: skip
            assert floor * 10000 > 8500 * window
    assert not any(r["reason"] == "budget_exceeded" and r["calls"] == 0 for r in results)  # and here it never does
    tools = summary["tool_definitions"]["tokens"]["L"]
    assert calls and all(c["predicted_prompt"] - tools == c["window"] <= window for c in calls)  # counted prompt minus tools <= W
    assert sum(r["reason"] == "answered" for r in results) >= 10
    digests = 0
    for db in sorted((tmp_path / "L").glob("episode-*.sqlite")):
        events = Ledger(db).events()
        assert check(events, SqliteBlobStore(db)) == []  # P9 re-runs every v2c decision, digests included
        digests += sum(e.type == "fold" and bool((e.payload["fold"]["policy"] or {}).get("digest")) for e in events)
    assert digests >= 1


def test_the_same_episode_under_the_v2b_rules_collapses(tmp_path, corpus, monkeypatch):
    _, results, _, _ = episode(tmp_path, corpus, 2000, monkeypatch, rules="v2b")
    assert sum(r["reason"] == "budget_exceeded" and r["calls"] == 0 for r in results) >= 15


# The pilot's own deadlocked window, read-only.


PILOT_LEDGER = ROOT / "runs" / "pilot-v2b-haiku-w3000" / "rho-10" / "L" / "episode-16.sqlite"


@pytest.mark.skipif(not PILOT_LEDGER.exists(), reason="the v2b pilot artifacts are not in this checkout")
def test_the_pilots_locked_window_is_recovered_under_v2c_read_only():
    """runs/pilot-v2b-haiku-w3000 rho 10, episode 16, the fourth question (seq 222): under the
    recorded v2b policy nothing is foldable (all "recent") and the window, at the pilot's
    counted/estimated ratio (3386 / 2337 at the fold of seq 215), is mandatory; under v2c the
    backstop brings it to hard. The ledger is opened immutable and is unchanged."""
    digest = hashlib.sha256(PILOT_LEDGER.read_bytes()).hexdigest()
    siblings = sorted(p.name for p in PILOT_LEDGER.parent.iterdir())
    con = sqlite3.connect(f"file:{PILOT_LEDGER}?mode=ro&immutable=1", uri=True)
    try:
        rows = con.execute("SELECT seq, type, payload, prev_hash, hash FROM events WHERE seq <= 222 ORDER BY seq").fetchall()
        blobs = MemoryBlobStore({h: bytes(d) for h, d in con.execute("SELECT hash, data FROM blobs")})
    finally:
        con.close()
    events = [Event(seq, t, json.loads(p), a, b) for seq, t, p, a, b in rows]
    recorded = policy_from_dict(next(e.payload["fold"]["policy"]["params"] for e in events if e.type == "fold"))
    measure = lambda s: lambda: rendered_window_tokens(s.state, s.blobs) * 3386 // 2337  # noqa: E731
    for rules in ("v2b", "v2c"):
        ledger = Ledger(":memory:")
        for e in events:
            ledger.append(e.type, e.payload)
        s = Session(ledger, blobs, recorded.budget_tokens, BackstopPolicyV1(**{**recorded.__dict__, "rules": rules}))
        assert s.payoff_model().current_question == events[-1].payload["segment"]["segment_id"]
        if rules == "v2b":
            with pytest.raises(BudgetExceeded):
                s.prepare_call(measure(s))
            assert s.last_folds == []
        else:
            s.prepare_call(measure(s))
            assert s.last_folds and s.last_window * 10000 <= 8500 * recorded.budget_tokens
    assert hashlib.sha256(PILOT_LEDGER.read_bytes()).hexdigest() == digest
    assert sorted(p.name for p in PILOT_LEDGER.parent.iterdir()) == siblings
