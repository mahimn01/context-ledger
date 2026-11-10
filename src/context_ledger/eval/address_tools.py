"""The v4 address tools (BUILD-V4 item 2), over the address index (index/) and the rho's corpus:

    lookup(name)        exact addresses of a name (a dotted name, a bare function name or a file
                        path), else the nearest DIFFERENT identifiers, labelled
    read(address)       the lines of an address "path:start-end[#sha]": the hash, if given, must
                        still match the lines; then the same cap as corpus_read (200 lines or
                        min(2,000, W // 4) estimated tokens)
    outline(target)     the children of a directory, a file or a qualified name, with addresses
    search(query)       BM25 chunks, as corpus_search; at most MAX_SEARCHES per question

In L4 (`LedgerAddressTools`) read registers the file as a source and appends its lines as
verbatim spans, exactly as L's corpus_read does (eval/tools.LedgerCorpusTools: python spans of
the symbols a range covers, text spans otherwise, each counted with its alias header), so Tier 0,
Tier 1, the gate, folding and ctx_refetch treat them like any span; an evicted or folded span
keeps its address, so reading it again is exact. The ctx tools are dispatched as in L. B5
(`PlainAddressTools`) gets plain text, as B4 does. Nothing here reads a question's gold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..index import AddressIndex, parse_address
from ..model import PyAnchor
from ..tools import ToolOutcome, _check, _Invalid, dispatch as ctx_dispatch
from .corpus import CorpusIndex, lines_of
from .formats import format_read
from .tools import MAX_READ_TOKENS, RECEIPT_ALIASES, LedgerCorpusTools, clamp_read, search_text

MAX_SEARCHES = 2


def preresolve_cap(window: int) -> int:
    """The token cap of the harness's own reads (pre-resolution and repair) of the asked target's
    anchored span: W // 2, so the span is read whole and its anchored path witnesses the target
    (a def longer than the read cap would otherwise come as a text span of its first lines, which
    neither Tier 1 nor the gate can anchor). Reads the model makes keep corpus_read's cap."""
    return window // 2
ADDRESS_TOOLS: list[dict] = [
    {
        "name": "lookup",
        "description": (
            "Find the exact address of a name: a dotted name such as mod.Class.method or mod.CONSTANT, a bare function name, or a file "
            "path. Returns path:start-end#hash addresses. A name that is not found exactly is reported with the nearest DIFFERENT "
            "identifiers, never as the one asked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 300}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "outline",
        "description": "List what a directory, a file or a qualified name (a module or class) contains, one line each with its address.",
        "input_schema": {
            "type": "object",
            "properties": {"target": {"type": "string", "minLength": 1, "maxLength": 300}},
            "required": ["target"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read",
        "description": "Read the lines of an address (path:start-end, optionally with its #hash, as lookup and outline give it): at most 200 lines or about 2,000 tokens per call.",
        "input_schema": {
            "type": "object",
            "properties": {"address": {"type": "string", "minLength": 3, "maxLength": 400}},
            "required": ["address"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search",
        "description": f"Find source chunks by keywords (at most {MAX_SEARCHES} per question; prefer lookup). Returns path:start-end line ranges with the first line of each; read them with read.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]
B5_TOOL_NAMES = ("lookup", "outline", "read")
B5_TOOLS = [t for t in ADDRESS_TOOLS if t["name"] in B5_TOOL_NAMES]
_SCHEMAS = {t["name"]: t["input_schema"] for t in ADDRESS_TOOLS}


def _invalid(name: str, args: dict) -> str | None:
    try:
        _check(_SCHEMAS[name], args, "input")
    except _Invalid as exc:
        return f"error: invalid_input {exc}"
    return None


def resolve_read(corpus: CorpusIndex, address: AddressIndex, text: str) -> tuple[str, int, int] | str:
    """(path, start, end) of a read's address, or an error line: not an address, a path not in
    this corpus, or lines that no longer hash to the #sha it carries."""
    parsed = parse_address(text)
    if parsed is None:
        return f"error: not_an_address {text!r} (give path:start-end, e.g. from lookup)"
    path, start, end, sha = parsed
    if path not in corpus.paths:
        return f"error: unknown_path {path}"
    if sha is not None and not address.verify(path, start, end, sha, corpus.text(path)):
        return f"error: stale_address {text} (those lines no longer hash to #{sha}; lookup the name again)"
    return path, start, min(max(start, end), corpus.text(path).count("\n") + 1)


@dataclass
class LedgerAddressTools(LedgerCorpusTools):
    """L4's tools: the address tools registering spans in the session, plus the ctx tools.
    `searches` counts each question's search calls (by turn id); `calls` every address tool
    call by name, per turn."""

    address: AddressIndex | None = None
    searches: dict = field(default_factory=dict)
    calls: dict = field(default_factory=dict)

    def __call__(self, session, name: str, args: dict, *, turn_id: str, tool_use_id: str | None = None) -> ToolOutcome:
        self.calls.setdefault(turn_id, {}).setdefault(name, 0)
        self.calls[turn_id][name] += 1
        if name not in _SCHEMAS:
            return ctx_dispatch(session, name, args, turn_id=turn_id, tool_use_id=tool_use_id)
        problem = _invalid(name, args)
        if problem:
            return ToolOutcome(problem, True)
        if name == "lookup":
            return ToolOutcome(self.address.lookup(args["name"]).render())
        if name == "outline":
            text = self.address.outline(args["target"])
            return ToolOutcome(text, text.startswith("error:"))
        if name == "search":
            used = self.searches.get(turn_id, 0)
            if used >= MAX_SEARCHES:
                return ToolOutcome(f"error: search_limit {MAX_SEARCHES} searches per question are used; lookup a name or outline a file instead", True)
            self.searches[turn_id] = used + 1
            return ToolOutcome(search_text(self.index, args["query"], args.get("limit", 5)))
        where = resolve_read(self.index, self.address, args["address"])
        if isinstance(where, str):
            return ToolOutcome(where, True)
        return self.read(session, *where, tool_use_id=tool_use_id)

    def read(self, session, path: str, start: int, end: int, *, tool_use_id: str | None = None, role: str = "tool") -> ToolOutcome:
        """Append lines start..end of `path` as verbatim spans (corpus_read's cover and cap), as
        part of the tool result of `tool_use_id`, or (the harness's own reads: pre-resolution and
        repair) as `role` "user" segments; the receipt names each span's alias."""
        count = lambda t, cls: session.counter.count(t, cls)  # noqa: E731
        start, end, truncated = clamp_read(self.index, path, start, end, count, self.cap)
        span_ids, cut = self._cover(session, path, start, end)
        if cut is not None:
            truncated, end = True, cut - 1
        appended, aliases = [], []
        for span_id in span_ids:
            if tool_use_id is not None:
                appended.append(session.append_span(span_id, tool_use_id=tool_use_id).segment_id)
            else:
                appended.append(session.append_span(span_id, role=role).segment_id)
            span = session.state.spans[span_id]
            label = span.anchor.qualname if isinstance(span.anchor, PyAnchor) else "text"
            aliases.append(f"{session.state.aliases.alias_of(span_id)} ({label})")
        shown = aliases[:RECEIPT_ALIASES] + ([f"and {len(aliases) - RECEIPT_ALIASES} more"] if len(aliases) > RECEIPT_ALIASES else [])
        receipt = f"read {path}:{start}-{end} as " + (", ".join(shown) or "nothing (blank lines)")
        return ToolOutcome(receipt + (f"; truncated, continue at line {end + 1}" if truncated else ""), False, appended)

    def symbol_span(self, session, path: str, start: int, end: int) -> str | None:
        """The registered Python span whose lines are exactly start..end (an index address), or None."""
        _, symbols = self._source(session, path)
        return next((span_id for a, b, span_id in symbols if (a, b) == (start, end)), None)

    def span_tokens(self, session, span_id: str) -> int:
        """A span's tokens as the window carries it, alias header included (as _cover counts)."""
        from ..alias import alias_for
        from ..render import span_label
        from ..tokens import source_class

        span, aliases = session.state.spans[span_id], session.state.aliases
        alias = aliases.alias_of(span_id) or alias_for(aliases.count)
        text = session.blobs.get(span.text_hash).decode("utf-8")
        path = session.state.sources[span.source_id].uri
        return session.counter.count(f"[{alias} {span_label(session.state, span_id)}]\n" + text, source_class(path))

    def turn_counts(self, turn_id: str) -> dict:
        return dict(sorted(self.calls.get(turn_id, {}).items()))


@dataclass
class PlainAddressTools:
    """B5's tools: lookup, outline and read with plain-text results (read as B4's corpus_read)."""

    index: CorpusIndex
    address: AddressIndex
    count: Callable[[str, str], int]
    cap: int = MAX_READ_TOKENS

    def __call__(self, name: str, args: dict) -> tuple[str, bool]:
        if name not in B5_TOOL_NAMES:
            return f"error: unknown_tool {name}", True
        problem = _invalid(name, args)
        if problem:
            return problem, True
        if name == "lookup":
            return self.address.lookup(args["name"]).render(), False
        if name == "outline":
            text = self.address.outline(args["target"])
            return text, text.startswith("error:")
        where = resolve_read(self.index, self.address, args["address"])
        if isinstance(where, str):
            return where, True
        return self.read(*where), False

    def read(self, path: str, start: int, end: int, cap: int | None = None) -> str:
        start, end, truncated = clamp_read(self.index, path, start, end, self.count, self.cap if cap is None else cap)
        text = format_read(path, start, end, lines_of(self.index.text(path), start, end))
        return text + (f"\n(truncated; continue at line {end + 1})" if truncated else "")


__all__ = ["ADDRESS_TOOLS", "B5_TOOLS", "MAX_SEARCHES", "LedgerAddressTools", "PlainAddressTools", "preresolve_cap", "resolve_read"]
