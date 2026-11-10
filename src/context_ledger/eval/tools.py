"""The corpus tools shared by the agent arms: corpus_search and corpus_read.

    corpus_search(query, limit <= 10)            path:start-end ranges with a one-line snippet each
    corpus_read(path, start_line, end_line)      at most 200 lines or min(2,000, W // 4) estimated
                                                 tokens (BUILD-V2b: reads scale with W, the same
                                                 for B4 and L; the tool description still says
                                                 about 2,000, so the tool definitions are unchanged)

B4 gets plain text results. In L, corpus_read registers the file as a source and appends the
lines as verbatim spans (python spans of the symbols they cover for .py files, a text span
otherwise) inside the tool result, so knowledge states, refetch and Tier 0/1 work on them; its
own text is a one-line receipt listing the aliases. Search results never expose span content
beyond the one-line snippet, and in L they are raw text, never spans.

BUILD-V2c: in L the token cap counts each span as the window carries it, its alias header
"[@7 Lib/token.py#ENDMARKER]" included, and the receipt lists at most RECEIPT_ALIASES aliases
(each span's header names its own). A read of 56 one-line constants had cost about three
times the cap: one header per line, and an 861-byte receipt repeating them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..alias import alias_for
from ..model import PyAnchor
from ..render import span_label
from ..tools import ToolOutcome, _check, _Invalid, dispatch as ctx_dispatch
from ..tokens import source_class
from .corpus import CorpusIndex, lines_of
from .formats import format_read

MAX_READ_LINES = 200
MAX_READ_TOKENS = 2000
RECEIPT_ALIASES = 8
CORPUS_TOOLS: list[dict] = [
    {
        "name": "corpus_read",
        "description": "Read lines start_line..end_line (1-based, inclusive) of a corpus file: at most 200 lines or about 2,000 tokens per call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "maxLength": 300},
                "start_line": {"type": "integer", "minimum": 1, "maximum": 1000000},
                "end_line": {"type": "integer", "minimum": 1, "maximum": 1000000},
            },
            "required": ["path", "start_line", "end_line"],
            "additionalProperties": False,
        },
    },
    {
        "name": "corpus_search",
        "description": "Find source chunks by keywords. Returns path:start-end line ranges with the first line of each; read them with corpus_read.",
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
_SCHEMAS = {t["name"]: t["input_schema"] for t in CORPUS_TOOLS}


def read_cap(window: int) -> int:
    """corpus_read's token cap at window W: min(MAX_READ_TOKENS, W // 4)."""
    return min(MAX_READ_TOKENS, window // 4)


def cited_lines(state, blobs, ref: str) -> str | None:
    """A span ref as the shared citation format "path:start-end": its source's path and the
    1-based lines its characters run over in that source version; None for any other ref. A
    note's span (BUILD-V2c) is cited as its note, "note:<pos>", as B3 and B4 cite it."""
    span = state.spans.get(ref)
    source = state.sources.get(span.source_id) if span is not None else None
    if source is None:
        return None
    if source.uri.startswith("note:"):
        return source.uri
    text = blobs.get(span.source_hash).decode("utf-8")
    first = text.count("\n", 0, span.anchor.start) + 1
    last = text.count("\n", 0, max(span.anchor.start, span.anchor.end - 1)) + 1
    return f"{source.uri}:{first}-{last}"


def search_text(index: CorpusIndex, query: str, limit: int) -> str:
    hits = index.search(query, limit)
    if not hits:
        return f"no matches for {query!r}"
    return "\n".join(f"{h.path}:{h.start}-{h.end}  {h.snippet}" for h in hits)


def clamp_read(index: CorpusIndex, path: str, start: int, end: int, count: Callable[[str, str], int], cap: int = MAX_READ_TOKENS) -> tuple[int, int, bool]:
    """The line range actually read: within the file, at most MAX_READ_LINES lines and `cap`
    estimated tokens. Returns (start, end, truncated)."""
    n = index.text(path).count("\n") + 1
    start = min(max(1, start), n)
    end = min(max(start, end), n, start + MAX_READ_LINES - 1)
    truncated = end < min(max(start, end), n)
    while end > start and count(lines_of(index.text(path), start, end), source_class(path)) > cap:
        end = start + (end - start) // 2
        truncated = True
    return start, end, truncated


def _validate(name: str, args: dict) -> str | None:
    try:
        _check(_SCHEMAS[name], args, "input")
    except _Invalid as exc:
        return f"error: invalid_input {exc}"
    return None


@dataclass
class PlainCorpusTools:
    """B4's tools: plain-text results."""

    index: CorpusIndex
    count: Callable[[str, str], int]
    cap: int = MAX_READ_TOKENS

    def __call__(self, name: str, args: dict) -> tuple[str, bool]:
        if name not in _SCHEMAS:
            return f"error: unknown_tool {name}", True
        problem = _validate(name, args)
        if problem:
            return problem, True
        if name == "corpus_search":
            return search_text(self.index, args["query"], args.get("limit", 5)), False
        if args["path"] not in self.index.paths:
            return f"error: unknown_path {args['path']}", True
        start, end, truncated = clamp_read(self.index, args["path"], args["start_line"], args["end_line"], self.count, self.cap)
        text = format_read(args["path"], start, end, lines_of(self.index.text(args["path"]), start, end))
        return text + (f"\n(truncated; continue at line {end + 1})" if truncated else ""), False


@dataclass
class LedgerCorpusTools:
    """L's tools: the corpus tools registering spans in the session, plus the ctx tools."""

    index: CorpusIndex
    cap: int = MAX_READ_TOKENS
    sources: dict = field(default_factory=dict)  # path -> (Source, [(start_line, end_line, span_id)] of its python spans)

    def _source(self, session, path: str):
        if path not in self.sources:
            text = self.index.text(path)
            source = session.register_source(path, text, media_type="text/x-python" if path.endswith(".py") else "text/plain")
            symbols = []
            if path.endswith(".py"):
                for span in session.register_python_spans(source, path):
                    a, b = text.count("\n", 0, span.anchor.start) + 1, text.count("\n", 0, span.anchor.end) + 1
                    symbols.append((a, b, span.span_id))
            self.sources[path] = (source, symbols)
        return self.sources[path]

    def _cover(self, session, path: str, start: int, end: int) -> tuple[list[str], int | None]:
        """Span ids covering lines start..end, and the next unread line if the token cap cut it
        short. For .py files: at each line, the outermost symbol starting there that fits the
        read limits (it may run past `end`); lines no such symbol covers become text spans up to
        the next symbol start. Other files: one text span. Spans never overlap. A span counts
        against the cap with its alias header (the alias it has, or the one it will get)."""
        source, symbols = self._source(session, path)
        text = self.index.text(path)
        line_starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
        count = session.counter

        def text_span(a: int, b: int) -> str | None:
            lo, hi = line_starts[a - 1], (line_starts[b] - 1 if b < len(line_starts) else len(text))
            return session.register_text_span(source, lo, hi).span_id if text[lo:hi].strip() else None

        def tokens(span_id: str, new: int = 0) -> int:
            span, aliases = session.state.spans[span_id], session.state.aliases
            alias = aliases.alias_of(span_id) or alias_for(aliases.count + new)
            return count.count(f"[{alias} {span_label(session.state, span_id)}]\n" + text[span.anchor.start : span.anchor.end], source_class(path))

        def fits(span_id: str) -> bool:
            return tokens(span_id) <= self.cap

        out, line, used = [], start, 0
        while line <= end:
            here = [s for s in symbols if s[0] == line and s[1] - s[0] < MAX_READ_LINES and fits(s[2])]
            if here:
                a, b, span_id = max(here, key=lambda s: s[1])
            else:
                following = [s[0] for s in symbols if s[0] > line]
                b = min([end, *(f - 1 for f in following)])
                span_id = text_span(line, b)
            if span_id is not None:
                cost = tokens(span_id, sum(session.state.aliases.alias_of(s) is None for s in out))
                if out and used + cost > self.cap:
                    return out, line
                out.append(span_id)
                used += cost
            line = b + 1
        return out, None

    def __call__(self, session, name: str, args: dict, *, turn_id: str, tool_use_id: str | None = None) -> ToolOutcome:
        if name not in _SCHEMAS:
            return ctx_dispatch(session, name, args, turn_id=turn_id, tool_use_id=tool_use_id)
        problem = _validate(name, args)
        if problem:
            return ToolOutcome(problem, True)
        if name == "corpus_search":
            return ToolOutcome(search_text(self.index, args["query"], args.get("limit", 5)))
        path = args["path"]
        if path not in self.index.paths:
            return ToolOutcome(f"error: unknown_path {path}", True)
        count = lambda t, cls: session.counter.count(t, cls)  # noqa: E731
        start, end, truncated = clamp_read(self.index, path, args["start_line"], args["end_line"], count, self.cap)
        appended, aliases = [], []
        span_ids, cut = self._cover(session, path, start, end)
        if cut is not None:
            truncated, end = True, cut - 1
        for span_id in span_ids:
            appended.append(session.append_span(span_id, tool_use_id=tool_use_id).segment_id)
            span = session.state.spans[span_id]
            label = span.anchor.qualname if isinstance(span.anchor, PyAnchor) else "text"
            aliases.append(f"{session.state.aliases.alias_of(span_id)} ({label})")
        shown = aliases[:RECEIPT_ALIASES] + ([f"and {len(aliases) - RECEIPT_ALIASES} more"] if len(aliases) > RECEIPT_ALIASES else [])
        receipt = f"read {path}:{start}-{end} as " + (", ".join(shown) or "nothing (blank lines)")
        return ToolOutcome(receipt + (f"; truncated, continue at line {end + 1}" if truncated else ""), False, appended)
