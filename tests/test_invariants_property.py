from __future__ import annotations

import ast

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from context_ledger.anchor.text import normalize_for_match
from context_ledger.fold import BackstopPolicy, FoldError, refs_of
from context_ledger.invariants import check
from context_ledger.model import Citation, Knowledge, SummarySentence
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.window import down, fold_sentence_refs, kappa, state_at_call

BUDGET = 60
PY_V1 = '''LIMIT = 10
NAME = "svc"


def area(w, h):
    return w * h


class Box:
    def size(self):
        return LIMIT
'''
PY_VALID = [
    PY_V1,
    PY_V1.replace("LIMIT = 10", "LIMIT = 12"),
    "# header\n\n" + PY_V1.replace("def area(w, h):", "def area(\n    w,\n    h,\n):"),
    PY_V1.replace("def area(", "def surface("),
]
PY_VERSIONS = PY_VALID + [PY_V1 + "\ndef broken(:\n"]
TEXTS = st.text(alphabet="abcdefgh ijk\n0123456789.=_", min_size=1, max_size=120)
SENTENCE_TEXT = st.text(alphabet="abcdefg xyz0123", min_size=1, max_size=30)
NOT_OFTEN = st.sampled_from([False, False, False, True])


def parses(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


class LedgerMachine(RuleBasedStateMachine):
    """Random sessions; the ledger must satisfy every invariant after every step."""

    def __init__(self) -> None:
        super().__init__()
        self.ledger = Ledger(":memory:")
        self.blobs = MemoryBlobStore()
        policy = BackstopPolicy(budget_tokens=BUDGET, cold_m=1, tail_keep=1)
        self.session = Session(self.ledger, self.blobs, BUDGET, policy)
        self.session.append_raw("system", "Answer only from cited spans.", pinned=True)
        self.sources: list[tuple[str, bool]] = []  # (source_id, is_python)

    def teardown(self) -> None:
        self.ledger.close()

    # helpers

    @property
    def state(self):
        return self.session.state

    def current_text(self, source_id: str) -> str:
        return self.blobs.get(self.state.sources[source_id].content_hash).decode()

    def unpinned(self) -> list[str]:
        return [s.segment_id for s in self.state.segments if not s.pinned]

    def fresh_spans(self) -> list[str]:
        """Spans of their source's current version, long enough to hold a sub-span."""
        return sorted(
            sid
            for sid, span in self.state.spans.items()
            if span.source_hash == self.state.sources[span.source_id].content_hash and span.n_chars >= 2
        )

    # rules

    @initialize(text=TEXTS, module=st.sampled_from(PY_VALID))
    def seed_sources(self, text, module):
        self.register_text_source(text)
        self.register_python_source(module)

    @rule(text=TEXTS)
    def register_text_source(self, text):
        source = self.session.register_source(f"note{len(self.sources)}.txt", text)
        self.sources.append((source.source_id, False))

    @rule(text=st.sampled_from(PY_VALID))
    def register_python_source(self, text):
        uri = f"mod{len(self.sources)}.py"
        source = self.session.register_source(uri, text, media_type="text/x-python")
        self.sources.append((source.source_id, True))
        self.session.register_python_spans(source, uri)

    @precondition(lambda self: self.sources)
    @rule(data=st.data())
    def register_text_span(self, data):
        source_id, _ = data.draw(st.sampled_from(self.sources))
        text = self.current_text(source_id)
        start = data.draw(st.integers(0, len(text) - 1))
        end = data.draw(st.integers(start + 1, min(len(text), start + 80)))
        self.session.register_text_span(self.state.sources[source_id], start, end)

    @precondition(lambda self: any(py and parses(self.current_text(sid)) for sid, py in self.sources))
    @rule(data=st.data())
    def register_python_spans(self, data):
        options = [sid for sid, py in self.sources if py and parses(self.current_text(sid))]
        source_id = data.draw(st.sampled_from(options))
        self.session.register_python_spans(self.state.sources[source_id], self.state.sources[source_id].uri)

    @precondition(lambda self: self.state.spans)
    @rule(data=st.data(), role=st.sampled_from(["tool", "user", "assistant"]), pinned=NOT_OFTEN, then_call=st.booleans())
    def append_span(self, data, role, pinned, then_call):
        span_id = data.draw(st.sampled_from(sorted(self.state.spans)))
        self.session.append_span(span_id, role=role, pinned=pinned)
        if then_call:
            self.session.call()

    @precondition(lambda self: self.fresh_spans())
    @rule(data=st.data())
    def append_snippet(self, data):
        parent = self.state.spans[data.draw(st.sampled_from(self.fresh_spans()))]
        start = data.draw(st.integers(parent.anchor.start, parent.anchor.end - 1))
        end = data.draw(st.integers(start + 1, parent.anchor.end))
        child = self.session.register_text_span(self.state.sources[parent.source_id], start, end)
        if child.span_id != parent.span_id:
            self.session.append_span(child.span_id, parent=parent.span_id)

    @rule(role=st.sampled_from(["user", "assistant", "tool"]), text=TEXTS, pinned=NOT_OFTEN, then_call=st.booleans())
    def append_raw(self, role, text, pinned, then_call):
        self.session.append_raw(role, text, pinned=pinned)
        if then_call:
            self.session.call()

    @precondition(lambda self: self.unpinned())
    @rule(data=st.data())
    def model_fold(self, data):
        victims = data.draw(st.lists(st.sampled_from(self.unpinned()), min_size=1, max_size=3, unique=True))
        by_id = {s.segment_id: s for s in self.state.segments}
        refs = sorted(set().union(*(refs_of(by_id[v], self.state) for v in victims)) | {"inference"})
        cites = st.lists(st.sampled_from(refs), max_size=3, unique=True).map(tuple)
        sentences = data.draw(st.lists(st.builds(SummarySentence, SENTENCE_TEXT, cites), min_size=1, max_size=3))
        before = len(self.session.events)
        try:
            self.session.fold(victims, sentences)
        except FoldError as exc:  # M2: a fold that frees no tokens is refused and records nothing
            assert exc.reason == "fold_would_grow_window" and len(self.session.events) == before

    @precondition(lambda self: self.unpinned())
    @rule(data=st.data())
    def evict(self, data):
        ids = data.draw(st.lists(st.sampled_from(self.unpinned()), min_size=1, max_size=2, unique=True))
        self.session.evict(ids, "test")

    @precondition(lambda self: self.sources)
    @rule(data=st.data())
    def update_source(self, data):
        source_id, is_python = data.draw(st.sampled_from(self.sources))
        if is_python:
            new_text = data.draw(st.sampled_from(PY_VERSIONS))
        else:
            text = self.current_text(source_id)
            pos = data.draw(st.integers(0, len(text)))
            cut = data.draw(st.integers(pos, min(len(text), pos + 10)))
            new_text = text[:pos] + data.draw(st.text(alphabet="xyz\n", max_size=8)) + text[cut:]
            if not new_text:
                new_text = "x"
        self.session.update_source(source_id, new_text)

    @rule()
    def call(self):
        self.session.call()

    @precondition(lambda self: self.state.spans)
    @rule(data=st.data())
    def refetch(self, data):
        self.session.refetch(data.draw(st.sampled_from(sorted(self.state.spans))))

    def quoted_citation(self, data) -> Citation | None:
        """Cite a span segment of the latest call's window with a real quote from it."""
        at_call = state_at_call(self.session.events, self.state.calls[-1])
        refs = sorted({s.ref for s in at_call.segments if s.ref_kind == "span"})
        if not refs:
            return None
        span = at_call.spans[data.draw(st.sampled_from(refs))]
        text = self.blobs.get(span.text_hash).decode()
        quote = text[data.draw(st.integers(0, len(text) - 1)) :]
        if len(normalize_for_match(quote)) < min(20, len(normalize_for_match(text))):
            quote = text
        return Citation(span.span_id, quote)

    @precondition(lambda self: self.state.calls)
    @rule(data=st.data(), kind=st.sampled_from(["quote", "fold", "fold+quote", "none", "inference", "raw", "abstain"]))
    def claim(self, data, kind):
        if kind == "abstain":
            self.session.abstain("not enough information")
            return
        citations: list[Citation] = []
        fold_refs = sorted(r for f in self.state.folds.values() for r in fold_sentence_refs(f))
        if kind in ("fold", "fold+quote") and fold_refs:
            citations.append(Citation(data.draw(st.sampled_from(fold_refs))))
        if kind in ("quote", "fold+quote"):
            quoted = self.quoted_citation(data)
            if quoted is not None:
                citations.append(quoted)
        if kind == "inference":
            citations.append(Citation("inference"))
        if kind == "raw":
            raw = sorted(s.segment_id for s in self.state.segments if s.ref_kind == "raw")
            if raw:
                citations.append(Citation(f"raw:{data.draw(st.sampled_from(raw))}"))
        self.session.claim("supported by the quoted text", citations)

    @precondition(lambda self: any(down(self.state, r) for f in self.state.folds.values() for r in fold_sentence_refs(f)))
    @rule(data=st.data())
    def refetch_then_claim(self, data):
        """The harness loop: a claim rests on a fold sentence, so re-fetch a span behind it, call
        again, and claim with the fold sentence plus a quote from the re-fetched span."""
        fold_refs = sorted(r for f in self.state.folds.values() for r in fold_sentence_refs(f) if down(self.state, r))
        fold_ref = data.draw(st.sampled_from(fold_refs))
        resolution = self.session.refetch(data.draw(st.sampled_from(sorted(down(self.state, fold_ref)))))
        result = self.session.call()
        citations = [Citation(fold_ref)]
        fresh = self.session.events[-1].payload["exposures"]
        if resolution.status in ("OK", "MOVED"):
            new_id = next(e.payload["new_span_id"] for e in reversed(self.session.events) if e.type == "refetch")
            at_call = state_at_call(self.session.events, result.call_id)
            if new_id in fresh and kappa(at_call, new_id) is Knowledge.VERBATIM:
                citations.append(Citation(new_id, self.blobs.get(self.state.spans[new_id].text_hash).decode()))
        self.session.claim("supported by the quoted text", citations)

    @invariant()
    def ledger_satisfies_every_invariant(self):
        assert check(self.ledger.events(), self.blobs) == []


LedgerMachine.TestCase.settings = settings(max_examples=150, stateful_step_count=30, deadline=None)
TestLedgerInvariants = LedgerMachine.TestCase
