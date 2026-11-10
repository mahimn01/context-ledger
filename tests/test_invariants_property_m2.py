"""M1's random-session state machine, extended to an aliased (backstop-v1) session with M2 rules.
The ledger must satisfy every invariant after every step, and replay must rebuild the session's
alias map exactly."""

from __future__ import annotations

from hypothesis import event, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from context_ledger.anthropic.client import LedgerClient
from context_ledger.anthropic.fake import FakeAnthropicServer
from context_ledger.anthropic.request import InvalidWindow
from context_ledger.backstop import BudgetExceeded
from context_ledger.fold import BackstopPolicyV1, FoldError, refs_of
from context_ledger.model import Citation, Knowledge, SummarySentence
from context_ledger.session import EgressError, Session, SessionError
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.window import down, fold_sentence_refs, kappa, replay, state_at_call
from m2util import pricing, reply, text, tool_use
from context_ledger.tools import segment_alias, dispatch
from test_client import LOOKUP
from test_invariants_property import SENTENCE_TEXT, TEXTS, LedgerMachine

BUDGET = 200


class ForgivingSession(Session):
    """A bare call that hits BudgetExceeded records no call (only the backstop folds made before
    it, BUILD-V2b) and returns None, so the inherited M1 rules can keep calling."""

    def call(self):
        before = len(self.events)
        try:
            result = super().call()
        except BudgetExceeded:
            assert all(e.type == "fold" and e.payload["fold"]["trigger"] == "backstop" for e in self.events[before:])
            event("budget exceeded")
            return None
        event(f"zone {self.last_backstop.zone}")
        if self.last_backstop.decision is not None:
            event("backstop-v1 fold")
        return result


class LedgerMachineM2(LedgerMachine):
    def __init__(self) -> None:
        RuleBasedStateMachine.__init__(self)
        self.ledger = Ledger(":memory:")
        self.blobs = MemoryBlobStore()
        policy = BackstopPolicyV1(budget_tokens=BUDGET, cold_m=1, tail_keep=1, q0_bp=500, tau_tokens=20)
        self.session = ForgivingSession(self.ledger, self.blobs, BUDGET, policy, pricing=pricing())
        self.session.append_raw("system", "Answer only from cited spans.", pinned=True)
        self.sources = []

    def refused(self, fn, *args) -> bool:
        """Run a fold or evict; a refusal for splitting a tool use from its results (or, as in
        M1, for growing the window) records nothing."""
        before = len(self.session.events)
        try:
            fn(*args)
            return False
        except (FoldError, SessionError) as exc:
            assert getattr(exc, "reason", None) in ("tool_pair", "fold_would_grow_window") or "tool use" in str(exc)
            assert len(self.session.events) == before
            return True

    @precondition(lambda self: self.unpinned())
    @rule(data=st.data())
    def model_fold(self, data):
        """M1's model fold (random victims and cites), tolerating every M2 refusal."""
        victims = data.draw(st.lists(st.sampled_from(self.unpinned()), min_size=1, max_size=3, unique=True))
        by_id = {s.segment_id: s for s in self.state.segments}
        refs = sorted(set().union(*(refs_of(by_id[v], self.state) for v in victims)) | {"inference"})
        cites = st.lists(st.sampled_from(refs), max_size=3, unique=True).map(tuple)
        sentences = data.draw(st.lists(st.builds(SummarySentence, SENTENCE_TEXT, cites), min_size=1, max_size=3))
        if not self.refused(self.session.fold, victims, sentences):
            event("model fold")

    @precondition(lambda self: self.unpinned())
    @rule(data=st.data())
    def evict(self, data):
        """M1's evict, tolerating a refusal for splitting a tool pair."""
        self.refused(LedgerMachine.evict, self, data)

    @rule(data=st.data(), kind=st.sampled_from(["text", "tool_use", "two_tool_uses"]))
    def model_step(self, data, kind):
        """One LedgerClient.step() against the fake client; tool uses are answered at once."""
        n = len([e for e in self.session.events if e.type == "model_call"])
        uses = {"text": 0, "tool_use": 1, "two_tool_uses": 2}[kind]
        content = [text("noted")] + [tool_use(f"toolu_{n}_{k}", "lookup", ref="@0") for k in range(uses)]
        server = FakeAnthropicServer([reply(*content)], pricing=self.session.pricing)
        client = LedgerClient(self.session, server.as_fake_client(), model="claude-haiku-5-5", max_tokens=64, tools=[LOOKUP])
        try:
            step = client.step()
        except (InvalidWindow, BudgetExceeded) as exc:
            event(f"model step refused: {type(exc).__name__}")
            return
        for block in step.content:
            if block["type"] == "tool_use":
                self.session.append_raw("tool", data.draw(TEXTS), tool_use_id=block["id"], is_error=data.draw(st.booleans()))
        event(f"model step: {kind}")

    @rule(data=st.data(), tool=st.sampled_from(["ctx_status", "ctx_search", "ctx_refetch", "ctx_fold", "ctx_fold"]))
    def ctx_tool(self, data, tool):
        """A ctx tool call as the model would make it; it never raises, errors are typed."""
        aliases = [a for a, _ in self.state.aliases.items()] + ["@ZZZZ"]
        if tool == "ctx_status":
            arguments = {}
        elif tool == "ctx_search":
            arguments = {"query": data.draw(st.sampled_from(["timeout", "area size", "Box", "limit 10", "zzz"])), "limit": data.draw(st.integers(1, 10))}
        elif tool == "ctx_refetch":
            arguments = {"refs": data.draw(st.lists(st.sampled_from(aliases), min_size=1, max_size=3))}
        else:
            runs = self.session.payoff_model().analysis().runs
            if runs and data.draw(st.booleans()):  # the best advisory run, as ctx_status lists it
                run = runs[0]
                targets = [segment_alias(self.session, p) for p in range(run.i, run.j)]
            else:
                targets = data.draw(st.lists(st.sampled_from(aliases), min_size=1, max_size=3))
            cites = data.draw(st.lists(st.sampled_from(targets), max_size=2))
            arguments = {"targets": targets, "summary": [{"text": data.draw(SENTENCE_TEXT), "cites": cites}]}
        before = len(self.session.events)
        outcome = dispatch(self.session, tool, arguments, turn_id="t1")
        assert outcome.text and (not outcome.is_error or outcome.text.startswith("error: "))
        assert tool != "ctx_status" or len(self.session.events) == before
        event(f"{tool} {outcome.text.split()[1] if outcome.is_error else 'ok'}")

    @precondition(lambda self: any(down(self.state, r) for f in self.state.folds.values() for r in fold_sentence_refs(f)))
    @rule(data=st.data())
    def refetch_then_claim(self, data):
        """M1's rule, tolerating a call refused with BudgetExceeded."""
        fold_refs = sorted(r for f in self.state.folds.values() for r in fold_sentence_refs(f) if down(self.state, r))
        fold_ref = data.draw(st.sampled_from(fold_refs))
        resolution = self.session.refetch(data.draw(st.sampled_from(sorted(down(self.state, fold_ref)))))
        result = self.session.call()
        if result is None:
            return
        citations = [Citation(fold_ref)]
        if resolution.status in ("OK", "MOVED"):
            new_id = next(e.payload["new_span_id"] for e in reversed(self.session.events) if e.type == "refetch")
            if kappa(state_at_call(self.session.events, result.call_id), new_id) is Knowledge.VERBATIM:
                citations.append(Citation(new_id, self.blobs.get(self.state.spans[new_id].text_hash).decode()))
        self.session.claim("supported by the quoted text", citations)

    @precondition(lambda self: self.state.spans)
    @rule(data=st.data(), calls=st.integers(1, 3))
    def load_and_call(self, data, calls):
        """Push the window toward the budget: append a few spans, calling after each."""
        for _ in range(calls):
            self.session.append_span(data.draw(st.sampled_from(sorted(self.state.spans))))
            self.session.call()

    @precondition(lambda self: self.state.calls)
    @rule(data=st.data(), unknown=st.sampled_from([False, False, False, True]))
    def alias_claim(self, data, unknown):
        """Cite a verbatim span of the latest call by its alias (or an alias nobody introduced)."""
        at_call = state_at_call(self.session.events, self.state.calls[-1])
        refs = sorted({s.ref for s in at_call.segments if s.ref_kind == "span"})
        if not refs or unknown:
            self.session.claim("supported", [Citation("@ZZZZ", "nothing")])
            return
        span = at_call.spans[data.draw(st.sampled_from(refs))]
        text = self.blobs.get(span.text_hash).decode()
        self.session.claim("supported by the quoted text", [Citation(self.state.aliases.alias_of(span.span_id), text)])

    @precondition(lambda self: self.state.calls)
    @rule(data=st.data(), turn=st.sampled_from(["t1", "t2"]), abstained=st.booleans())
    def answer(self, data, turn, abstained):
        """Claim within a turn (any citation kind), then try to emit; a refused egress records
        nothing, and the harness then abstains over every open claim and emits that."""
        fold_refs = sorted(r for f in self.state.folds.values() for r in fold_sentence_refs(f))
        spans = sorted({s.ref for s in self.state.segments if s.ref_kind == "span"} | set(self.state.spans))
        ref = data.draw(st.sampled_from(fold_refs + spans))
        text = self.blobs.get(self.state.spans[ref].text_hash).decode() if ref in self.state.spans else None
        claim, _ = self.session.claim("supported by the quoted text", [Citation(ref, text)], turn_id=turn)
        before = len(self.session.events)
        try:
            self.session.emit_answer(turn, [claim.claim_id], abstained=abstained)
            event(f"emitted {claim.status}")
            return
        except EgressError:
            assert len(self.session.events) == before
        cover = [self.session.abstain("no verified answer", turn_id=turn, supersedes=c.claim_id) for c in self.session.unresolved_claims(turn)]
        self.session.emit_answer(turn, [c.claim_id for c in cover], abstained=True)
        event("emitted after abstaining")

    @invariant()
    def replay_rebuilds_the_alias_map(self):
        assert replay(self.ledger.events()).aliases == self.state.aliases


LedgerMachineM2.TestCase.settings = settings(max_examples=100, stateful_step_count=40, deadline=None)
TestLedgerInvariantsM2 = LedgerMachineM2.TestCase
