"""M4 item 1: Tier 1 splits identifiers from values. Values need the verbatim quote; an
identifier may instead be witnessed by the anchored path of a passing citation."""

from __future__ import annotations

import string

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from context_ledger.anchor.python import anchored_path, module_parts
from context_ledger.fold import BackstopPolicy
from context_ledger.model import Citation, Claim
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.verify.tier1 import IDENTIFIER
from context_ledger.verify import is_anchored, tier0, tier1, value_tokens_typed, verify_claim
from context_ledger.window import replay

PATH = "Lib/multiprocessing/context.py"
SOURCE = """\
class BaseContext(object):
    def violet_juniper_sparrow(self, module_names, *, on_error='igoore'):
        '''Configure the start method.'''
        violet_juniper_sparrow = 'igoore'
        return violet_juniper_sparrow


class SpawnContext(BaseContext):
    def cpu_count(self):
        return 12345
"""
SIGNATURE = "def violet_juniper_sparrow(self, module_names, *, on_error='igoore'):"


class World:
    """An M1-style session over SOURCE with BaseContext.violet_juniper_sparrow and
    SpawnContext.cpu_count appended verbatim and one call made."""

    def __init__(self, source: str = SOURCE, path: str = PATH) -> None:
        self.blobs = MemoryBlobStore()
        self.session = Session(Ledger(":memory:"), self.blobs, 100_000, BackstopPolicy(budget_tokens=100_000))
        src = self.session.register_source(path, source, media_type="text/x-python")
        self.spans = {s.anchor.qualname: s for s in self.session.register_python_spans(src, path)}
        self.segments = {}

    def show(self, *qualnames: str) -> World:
        for q in qualnames:
            self.segments[q] = self.session.append_span(self.spans[q].span_id)
        self.session.call()
        return self

    def verdict(self, text: str, *cites: tuple[str, str | None]):
        claim = Claim("c", self.session.state.calls[-1], text, tuple(Citation(self.spans[q].span_id, quote) for q, quote in cites), "ungrounded")
        return tier1(claim, replay(self.session.events), self.blobs)


@pytest.fixture
def world():
    return World().show("BaseContext.violet_juniper_sparrow", "SpawnContext.cpu_count")


METHOD = "BaseContext.violet_juniper_sparrow"


def test_anchored_path_drops_lib_py_init_and_locals():
    assert module_parts("Lib/multiprocessing/context.py") == ["multiprocessing", "context"]
    assert module_parts("Lib/email/__init__.py") == ["email"]
    assert module_parts("Tools/scripts/x.py") == ["Tools", "scripts", "x"]
    w = World("def outer():\n    def inner():\n        return 1\n    return inner\n", "Lib/pkg/__init__.py")
    assert anchored_path(w.spans["outer.<locals>.inner"].anchor) == ("pkg", "outer", "inner")


def test_identifier_and_value_kinds():
    tokens = value_tokens_typed("`mod.f` parameter `p` defaults to `3`, `'x y'`, `a + b`, 'q', f() and 3.15.0rc1")
    assert tokens == [
        ("identifier", "mod.f"),
        ("identifier", "p"),
        ("number", "3"),
        ("exact", "x y"),
        ("exact", "a + b"),
        ("exact", "q"),
        ("identifier", "f"),
        ("exact", "3.15.0rc1"),
    ]


@pytest.mark.parametrize("ident", [METHOD, "context.BaseContext", "multiprocessing.context", "violet_juniper_sparrow", "multiprocessing.context.BaseContext.violet_juniper_sparrow()"])
def test_right_identifier_and_right_value_pass(world, ident):
    v = world.verdict(f"`{ident}` parameter `on_error` defaults to `'igoore'`", (METHOD, SIGNATURE))
    assert (v.passed, v.reason) == (True, "quotes_verified")


@pytest.mark.parametrize("ident", ["SpawnContext.violet_juniper_sparrow", "BaseCont", "Context.violet", "BaseContext.cpu_count", "multiprocessing.violet_juniper_sparrow"])
def test_wrong_class_function_or_partial_component_fails(world, ident):
    v = world.verdict(f"`{ident}` parameter `on_error` defaults to `'igoore'`", (METHOD, SIGNATURE))
    assert (v.passed, v.reason) == (False, "identifier_not_anchored")


def test_a_trailing_module_prefix_witnesses_an_anchored_name(world):
    """BUILD-V2 item 1 (module-prefix witness) changed this case: `context` is a trailing
    component of the citation's module multiprocessing.context and violet_juniper_sparrow is
    anchored, so `context.violet_juniper_sparrow` passes (M4 rejected it as a non-contiguous run)."""
    v = world.verdict("`context.violet_juniper_sparrow` parameter `on_error` defaults to `'igoore'`", (METHOD, SIGNATURE))
    assert (v.passed, v.reason) == (True, "quotes_verified")


def test_right_identifier_wrong_value_fails_as_value_not_covered(world):
    v = world.verdict(f"`{METHOD}` parameter `on_error` defaults to `'ignore'`", (METHOD, SIGNATURE))
    assert (v.passed, v.reason) == (False, "value_not_covered")
    # The brief's example: the anchored name never vouches for the value.
    v = world.verdict("`BaseContext.violet_juniper_sparrow = 'ignore'`", (METHOD, "violet_juniper_sparrow = 'igoore'"))
    assert (v.passed, v.reason) == (False, "value_not_covered")
    v = world.verdict("BaseContext.violet_juniper_sparrow = 'ignore'", (METHOD, "violet_juniper_sparrow = 'igoore'"))
    assert (v.passed, v.reason) == (False, "value_not_covered")
    v = world.verdict("BaseContext.violet_juniper_sparrow = 'igoore'", (METHOD, "violet_juniper_sparrow = 'igoore'"))
    assert v.passed


def test_upstream_value_fails_against_perturbed_quote(world):
    """The unperturbed CPython value ('ignore') is what memory would answer: it must fail."""
    v = world.verdict(f"`{METHOD}` parameter `on_error` defaults to 'ignore'.", (METHOD, SIGNATURE))
    assert (v.passed, v.reason) == (False, "value_not_covered")


def test_number_canonicalization_with_anchored_identifier(world):
    for value in ("12345", "12345.0", "12_345", "1.2345e4"):
        v = world.verdict(f"`SpawnContext.cpu_count` returns `{value}`", ("SpawnContext.cpu_count", "def cpu_count(self):\n        return 12345"))
        assert v.passed, value
    v = world.verdict("`SpawnContext.cpu_count` returns `1234`", ("SpawnContext.cpu_count", "def cpu_count(self):\n        return 12345"))
    assert v.reason == "value_not_covered"


def test_identifier_anchored_only_by_a_citation_failing_tier0_fails():
    w = World().show(METHOD, "SpawnContext.cpu_count")
    text = f"`SpawnContext.cpu_count` is defined next to `{METHOD}`"
    cites = ((METHOD, SIGNATURE), ("SpawnContext.cpu_count", "def cpu_count(self):"))
    assert w.verdict(text, *cites).passed  # control: both spans verbatim
    w.session.evict([w.segments["SpawnContext.cpu_count"].segment_id], "test")  # dropped: fails Tier 0
    w.session.call()
    claim = Claim("c", w.session.state.calls[-1], text, tuple(Citation(w.spans[q].span_id, quote) for q, quote in cites), "ungrounded")
    state = replay(w.session.events)
    assert tier0(claim, state).passed  # the claim still cites one verbatim span directly
    assert (verify_claim(claim, state, w.blobs).reason) == "identifier_not_anchored"


def test_identifier_covered_by_quote_needs_no_anchor():
    """Text spans have no anchor: only quote coverage (rule a) applies."""
    w = World()
    src = w.session.state.sources[next(iter(w.session.state.sources))]
    span = w.session.register_text_span(src, 0, SOURCE.index("\n\n\n"))
    w.session.append_span(span.span_id)
    w.session.call()
    state = replay(w.session.events)

    def check(text: str):
        claim = Claim("c", w.session.state.calls[-1], text, (Citation(span.span_id, SIGNATURE),), "ungrounded")
        return tier1(claim, state, w.blobs)

    assert check("`violet_juniper_sparrow` takes `on_error`, default `'igoore'`").passed
    assert check(f"`{METHOD}` takes `on_error`").reason == "identifier_not_anchored"


_COMPONENT = st.text(string.ascii_lowercase, min_size=3, max_size=8).map(lambda s: "z" + s)


@settings(max_examples=150, deadline=None)
@given(
    st.lists(_COMPONENT, min_size=1, max_size=3, unique=True),
    st.lists(_COMPONENT, min_size=1, max_size=3, unique=True),
    st.data(),
)
def test_runs_pass_and_single_char_mutations_fail(module, qual, data):
    assume(not set(module) & set(qual))
    *classes, func = qual
    lines, indent = [], ""
    for name in classes:
        lines.append(f"{indent}class {name}:")
        indent += "    "
    lines += [f"{indent}def {func}(self):", f"{indent}    return 98765432109876543210"]
    path = "Lib/" + "/".join(module) + ".py"
    w = World("\n".join(lines) + "\n", path).show(".".join(qual))
    full = (*module, *qual)
    assert anchored_path(w.spans[".".join(qual)].anchor) == full
    runs = {".".join(full[i:j]) for i in range(len(full)) for j in range(i + 1, len(full) + 1)}
    quote = "return 98765432109876543210"
    run = data.draw(st.sampled_from(sorted(runs)))
    assert w.verdict(f"`{run}` returns the value.", (".".join(qual), quote)).passed
    k = data.draw(st.integers(0, len(run) - 1))
    op = data.draw(st.sampled_from(["replace", "insert", "delete"]))
    ch = data.draw(st.sampled_from(string.ascii_lowercase + "_"))
    mutated = {"replace": run[:k] + ch + run[k + 1 :], "insert": run[:k] + ch + run[k:], "delete": run[:k] + run[k + 1 :]}[op]
    # BUILD-V2 module-prefix witness: a trailing module run, then any anchored run, also passes.
    prefixed = {".".join(module[len(module) - s :]) + "." + r for s in range(1, len(module) + 1) for r in runs}
    assume(mutated not in runs | prefixed and IDENTIFIER.fullmatch(mutated) is not None and mutated != "return")
    v = w.verdict(f"`{mutated}` returns the value.", (".".join(qual), quote))
    assert (v.passed, v.reason) == (False, "identifier_not_anchored")
    assert not is_anchored(mutated, full)


def test_innermost_anchor_witnesses_a_method_quoted_from_its_class_span():
    w = World().show("BaseContext")
    full = "multiprocessing.context.BaseContext.violet_juniper_sparrow"
    v = w.verdict(f"`{full}` parameter `on_error` defaults to `'igoore'`", ("BaseContext", SIGNATURE))
    assert v.passed  # the def line occurs once in the class span, inside the method's anchor
    v = w.verdict("`BaseContext.cpu_count` is defined here", ("BaseContext", SIGNATURE))
    assert v.reason == "identifier_not_anchored"  # the quote locates violet_juniper_sparrow, not cpu_count


def test_innermost_anchor_needs_a_unique_quote():
    source = "class Pool:\n    def first(self):\n        return 1234567890123\n\n    def second(self):\n        return 1234567890123\n"
    w = World(source, "Lib/pool.py").show("Pool")
    v = w.verdict("`Pool.first` returns `1234567890123`", ("Pool", "return 1234567890123"))
    assert v.reason == "identifier_not_anchored"  # the quoted line is in both methods: no single anchor
    v = w.verdict("`pool.Pool` returns `1234567890123`", ("Pool", "return 1234567890123"))
    assert v.passed
