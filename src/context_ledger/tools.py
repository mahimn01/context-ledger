"""Model-facing context tools: definitions and a transport-agnostic dispatcher.

    ctx_status   budget, zone, knowledge counts, the backstop's best fold candidates (advisory),
                 summary-only spans cited this turn, pending refetch obligations, turn cost
    ctx_fold     fold a contiguous run the model names by aliases, under the backstop-v1 gate
    ctx_refetch  bring spans back verbatim (a fold alias brings back the spans behind it)
    ctx_search   BM25 over registered spans; metadata only, never span content

Handlers never raise to the model: a failure is an `is_error` result whose first line is a
short, typed, machine-readable code, for example "error: fold_would_grow_window delta=-12".
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .alias import UnknownAlias
from .anchor.python import anchored_path
from .backstop import FoldRejected
from .fold import FoldError
from .model import PyAnchor, SummarySentence
from .pricing import usd
from .render import span_label
from .verify import value_tokens_typed
from .window import down, kappa, parse_fold_ref, parse_whole_fold_ref, read_map, ref_kind_of

_ALIASES = {"type": "array", "items": {"type": "string"}}
CTX_TOOLS: list[dict] = [
    {
        "name": "ctx_fold",
        "description": (
            "Replace one contiguous run of older context with your own short summary, to free budget. "
            "targets: the aliases of every segment in the run (ctx_status lists good runs). summary: sentences, "
            "each citing the aliases it summarizes. Refused if the summary would not free tokens or does not pay off."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "targets": {**_ALIASES, "minItems": 1, "maxItems": 64},
                "summary": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "minLength": 1, "maxLength": 400},
                            "cites": {**_ALIASES, "maxItems": 16},
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["targets", "summary"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ctx_refetch",
        "description": (
            "Bring spans back verbatim. refs: span aliases, or fold and summary-line aliases to bring back "
            "the spans behind them (at most 8 spans per call). Reports OK, MOVED (with the new alias), STALE or DANGLING."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"refs": {**_ALIASES, "minItems": 1, "maxItems": 8}},
            "required": ["refs"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ctx_search",
        "description": "Find registered spans by keywords. Returns aliases and locations only; use ctx_refetch to read them.",
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
    {
        "name": "ctx_status",
        "description": "Report the context budget, zone, knowledge states, fold candidates, pending refetch obligations and turn cost.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]
_SCHEMAS = {tool["name"]: tool["input_schema"] for tool in CTX_TOOLS}
MAX_REFETCH = 8


@dataclass(frozen=True)
class ToolOutcome:
    """What a tool call produced: the result text, whether it is an error, and the segment ids
    the handler appended to the window (e.g. re-fetched spans)."""

    text: str
    is_error: bool = False
    appended: list[str] = field(default_factory=list)


class _Invalid(Exception):
    pass


def _check(schema: dict, value: object, where: str) -> None:
    """The JSON Schema subset the ctx tools use: object, array, string, integer; required,
    additionalProperties false, min/max lengths, items and values."""
    kind = schema["type"]
    ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
    }[kind]
    if not ok:
        raise _Invalid(f"{where} must be {kind}")
    if kind == "object":
        missing = [k for k in schema.get("required", []) if k not in value]
        extra = [k for k in value if k not in schema["properties"]]
        if missing or extra:
            raise _Invalid(f"{where}: missing {missing}" if missing else f"{where}: unexpected {extra}")
        for key, item in value.items():
            _check(schema["properties"][key], item, f"{where}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            raise _Invalid(f"{where} must have {schema.get('minItems', 0)} to {schema.get('maxItems')} items")
        for n, item in enumerate(value):
            _check(schema["items"], item, f"{where}[{n}]")
    elif kind == "string" and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", len(value)):
        raise _Invalid(f"{where} must be {schema.get('minLength', 0)} to {schema.get('maxLength')} characters")
    elif kind == "integer" and not schema.get("minimum", value) <= value <= schema.get("maximum", value):
        raise _Invalid(f"{where} must be between {schema.get('minimum')} and {schema.get('maximum')}")


def dispatch(session, name: str, input: dict, *, turn_id: str, tool_use_id: str | None = None) -> ToolOutcome:
    """Run tool `name` with `input` against `session` (a backstop-v1 Session). Never raises:
    bad input or a refused operation is an is_error outcome with a typed first line.
    `tool_use_id`, if given, makes re-fetched spans part of that tool use's result."""
    schema = _SCHEMAS.get(name)
    if schema is None:
        return ToolOutcome(f"error: unknown_tool {name}", True)
    try:
        _check(schema, input, "input")
    except _Invalid as exc:
        return ToolOutcome(f"error: invalid_input {exc}", True)
    if name == "ctx_status":
        return ToolOutcome(_status(session, turn_id))
    if name == "ctx_fold":
        return _fold(session, input)
    if name == "ctx_refetch":
        return _refetch(session, input["refs"], tool_use_id)
    return ToolOutcome(_search(session, input["query"], input.get("limit", 5)))


def _show(session, ref: str) -> str:
    return session.state.aliases.alias_of(ref) or ref


def segment_alias(session, position: int) -> str:
    """The alias a model would use for the segment at `position` (what ctx_fold targets name)."""
    seg = session.state.segments[position]
    target = seg.ref if seg.ref_kind == "span" else f"fold:{seg.ref}" if seg.ref_kind == "fold" else f"raw:{seg.segment_id}"
    return session.state.aliases.alias_of(target) or f"#{position}"


def _status(session, turn_id: str) -> str:
    """Pure read of the session: appends nothing (the caller records the tool result)."""
    state, budget = session.state, session.budget_tokens
    model = session.payoff_model()
    used = model.L
    pct, rest = divmod(used * 10000 // budget, 100)
    counts = Counter(read_map(state).values())
    lines = [
        f"budget {budget} tokens; window {used} rendered tokens ({pct}.{rest:02d}%); zone {model.zone}",
        "knowledge: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in ("verbatim", "snippet", "summary_only", "dropped", "stale")),
        "fold candidates (advisory, best first):",
    ]
    runs = model.analysis().runs[:5]
    lines += [
        f"  {n}. {' '.join(segment_alias(session, p) for p in range(r.i, r.j))}: frees {r.delta} tokens, J {usd(r.J)}"
        for n, r in enumerate(runs, start=1)
    ] or ["  (none)"]
    claims = session.turn_claims(turn_id)
    cited = {s for c in claims for cite in c.citations for s in down(state, cite.ref)}
    summary_only = sorted(_show(session, s) for s in cited if kappa(state, s).value == "summary_only")
    pending = sorted(
        {_show(session, o) for c in session.unresolved_claims(turn_id) if c.status == "needs_refetch" for o in c.verdict["obligations"]}
    )
    last_answer = max((e.seq for e in session.events if e.type == "answer_emitted"), default=0)
    cost = sum(e.payload["cost_pusd"] for e in session.events if e.type == "model_call" and e.seq > last_answer)
    lines += [
        f"summary_only spans cited this turn: {' '.join(summary_only) or 'none'}",
        f"pending refetch obligations: {' '.join(pending) or 'none'}",
        f"turn cost so far: {usd(cost)}",
    ]
    return "\n".join(lines)


def _positions(session, ref: str) -> list[int]:
    """Window positions a canonical ref names: a span's segments, a raw segment, or a fold's
    segment (for the whole fold or one of its sentences)."""
    fold_id = parse_whole_fold_ref(ref) or (parse_fold_ref(ref) or (None,))[0]
    out = []
    for p, seg in enumerate(session.state.segments):
        if (
            (seg.ref_kind == "span" and seg.ref == ref)
            or (seg.ref_kind == "raw" and ref == f"raw:{seg.segment_id}")
            or (seg.ref_kind == "fold" and seg.ref == fold_id)
        ):
            out.append(p)
    return out


def _fold(session, input: dict) -> ToolOutcome:
    aliases = session.state.aliases
    try:
        targets = [aliases.target_of(a) for a in input["targets"]]
        sentences = [
            SummarySentence(item["text"], tuple(aliases.target_of(c) for c in item.get("cites", []))) for item in input["summary"]
        ]
    except UnknownAlias as exc:
        return ToolOutcome(f"error: unknown_alias {exc.alias}", True)
    positions = sorted({p for target in targets for p in _positions(session, target)})
    if not positions:
        return ToolOutcome("error: infeasible_run why=not_in_window", True)
    victims = [session.state.segments[p].segment_id for p in positions]
    try:
        fold = session.gated_fold(victims, sentences)
    except FoldRejected as exc:
        details = {k: (usd(v) if k == "J" else v) for k, v in exc.details.items()}
        return ToolOutcome(" ".join([f"error: {exc.reason}", *(f"{k}={v}" for k, v in details.items())]), True)
    except FoldError as exc:
        return ToolOutcome(f"error: {exc.reason or 'invalid_fold'} {exc}", True)
    handle = session.state.aliases.alias_of(f"fold:{fold.fold_id}")
    return ToolOutcome(f"ok: folded {len(fold.victims)} segments, freed {fold.freed_tokens} tokens, new alias {handle}")


def _refetch(session, refs: list[str], tool_use_id: str | None) -> ToolOutcome:
    state = session.state
    order = {span_id: n for n, span_id in enumerate(state.spans)}  # first appearance in the ledger
    wanted: list[tuple[str, str]] = []  # (alias asked for, span id)
    lines = []
    for alias in refs:
        try:
            target = state.aliases.target_of(alias)
        except UnknownAlias:
            lines.append(f"{alias} UNKNOWN")
            continue
        kind = ref_kind_of(target)
        if kind == "raw":
            lines.append(f"{alias} NOT_A_SPAN")
            continue
        spans = [target] if kind == "span" else sorted(down(state, target), key=order.__getitem__)
        wanted += [(alias, s) for s in spans if s not in {w for _, w in wanted}]
    before = {seg.segment_id for seg in state.segments}
    for alias, span_id in wanted[:MAX_REFETCH]:
        resolution = session.refetch(span_id, tool_use_id=tool_use_id)
        fresh = session.events[-1].payload["new_span_id"]
        shown = _show(session, span_id)
        prefix = shown if shown == alias else f"{alias} {shown}"
        lines.append(f"{prefix} {resolution.status}" + (f" {_show(session, fresh)}" if fresh and fresh != span_id else ""))
    if len(wanted) > MAX_REFETCH:
        lines.append(f"capped at {MAX_REFETCH}: {len(wanted) - MAX_REFETCH} more not re-fetched")
    appended = [seg.segment_id for seg in session.state.segments if seg.segment_id not in before]
    if not wanted:
        return ToolOutcome("error: nothing_to_refetch\n" + "\n".join(lines), True)
    return ToolOutcome("\n".join(lines), False, appended)


_WORD = re.compile(r"[A-Za-z0-9]+")
_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


def search_tokens(text: str) -> list[str]:
    """Identifier-split, case-folded tokens: "DEFAULT_TIMEOUT" -> default, timeout;
    "InventoryClient.url_for" -> inventory, client, url, for."""
    return [part.casefold() for word in _WORD.findall(text) for part in _PART.findall(word)]


def rank_spans(session, query: str) -> list[str]:
    """Span ids by BM25 (k1=1.2, b=0.75) of `query` over spans of their source's current
    version (identifier-split, case-folded tokens of the label and the text), best first, only
    those that score. Floats are used only to rank, never recorded; ties go to the smaller id."""
    state = session.state
    docs = {}
    for span_id, span in state.spans.items():
        if span.source_hash == state.sources[span.source_id].content_hash:
            text = session.blobs.get(span.text_hash).decode("utf-8")
            docs[span_id] = Counter(search_tokens(span_label(state, span_id)) + search_tokens(text))
    terms = set(search_tokens(query))
    if not docs or not terms:
        return []
    avg = sum(sum(c.values()) for c in docs.values()) / len(docs)
    df = {t: sum(t in c for c in docs.values()) for t in terms}
    idf = {t: math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5)) for t in terms}
    scores = []
    for span_id, counts in docs.items():
        length = sum(counts.values())
        score = sum(idf[t] * counts[t] * 2.2 / (counts[t] + 1.2 * (0.25 + 0.75 * length / avg)) for t in terms if counts[t])
        if score > 0:
            scores.append((-score, span_id))
    return [span_id for _, span_id in sorted(scores)]


def question_identifiers(question: str) -> list[str]:
    """The identifiers a question names, as Tier 1 reads them (backticked names and code-like
    tokens matching the IDENTIFIER grammar), plus the bare last component of each dotted name;
    in order, without repeats."""
    names = []
    for kind, token in value_tokens_typed(question):
        if kind == "identifier":
            name = token.removesuffix("()")
            names += [name, name.rsplit(".", 1)[-1]]
    return list(dict.fromkeys(names))


def refetch_candidates(session, question: str, limit: int = MAX_REFETCH) -> list[str]:
    """Spans the window holds only behind summary lines (summary_only) whose anchored path,
    source path or text names an identifier of `question` (`question_identifiers`, as a whole
    token), ranked by the ctx_search BM25 of those identifiers, at most `limit`. Reads only the
    ledger and the question text: never a gold answer."""
    names = question_identifiers(question)
    if not names:
        return []
    state = session.state

    def names_it(span_id: str) -> bool:
        span = state.spans[span_id]
        places = [state.sources[span.source_id].uri, span_label(state, span_id), session.blobs.get(span.text_hash).decode("utf-8")]
        if isinstance(span.anchor, PyAnchor):
            places.append(".".join(anchored_path(span.anchor)))
        return any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", place) for n in names for place in places)

    folded = [s for s in rank_spans(session, " ".join(names)) if kappa(state, s).value == "summary_only" and names_it(s)]
    return folded[:limit]


def _search(session, query: str, limit: int) -> str:
    """BM25 over registered spans (`rank_spans`); hits get aliases."""
    state = session.state
    hits = rank_spans(session, query)[:limit]
    if not hits:
        return f"no matches for {query!r}"
    lines = []
    for span_id in hits:
        anchor = state.spans[span_id].anchor
        kind = f"python {anchor.node_kind}" if isinstance(anchor, PyAnchor) else "text"
        lines.append(f"{session.assign_alias(span_id)} {span_label(state, span_id)} ({kind})")
    return "\n".join(lines)


__all__ = ["CTX_TOOLS", "MAX_REFETCH", "ToolOutcome", "dispatch", "question_identifiers", "rank_spans", "refetch_candidates", "search_tokens", "segment_alias"]
