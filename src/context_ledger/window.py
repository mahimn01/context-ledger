"""Window replay and knowledge states.

`replay` rebuilds the window W_t from the ledger. `kappa` gives the knowledge state of a span
per the formal model: stale > verbatim > snippet > summary_only > dropped > unseen, where
stale overrides everything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from .alias import AliasMap
from .ids import canonical_json, sha256_hex
from .model import Event, FoldRecord, Knowledge, Segment, Source, Span, SummarySentence
from .store.blobs import BlobStore
from .tokens import estimator_for, from_payload, segment_class


class ReplayError(Exception):
    """An event cannot be applied to the window (unknown id, duplicate, malformed payload)."""

    def __init__(self, seq: object, message: str) -> None:
        super().__init__(f"seq {seq}: {message}")
        self.seq = seq


@dataclass
class WindowState:
    segments: list[Segment] = field(default_factory=list)
    spans: dict[str, Span] = field(default_factory=dict)
    sources: dict[str, Source] = field(default_factory=dict)
    folds: dict[str, FoldRecord] = field(default_factory=dict)
    ever_exposed: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)
    last_call_seq: int | None = None
    aliases: AliasMap = field(default_factory=AliasMap)
    pricing: dict | None = None  # the price table in force (PricingTable.to_dict()) and its hash
    pricing_hash: str | None = None
    call_last: str | None = None  # last segment of the window at the latest call
    model_last: str | None = None  # last segment of the window at the latest model call
    tokenizer: dict | None = None  # the tokenizer_set payload in force (None: bytes4-v0)
    estimator: object = None  # its parsed estimator (tokens.estimator_for falls back to bytes4-v0)
    thinking_bound: dict[str, str] | None = None  # BUILD-V4FIX: thinking key -> the digest of what preceded it, from model_calls; None if none recorded
    _down_cache: dict[str, frozenset[str]] = field(default_factory=dict, init=False, repr=False, compare=False)

    def copy(self) -> WindowState:
        """Independent copy (records are immutable, so containers are copied shallowly)."""
        return WindowState(
            list(self.segments),
            dict(self.spans),
            dict(self.sources),
            dict(self.folds),
            set(self.ever_exposed),
            list(self.calls),
            self.last_call_seq,
            self.aliases.copy(),
            self.pricing,
            self.pricing_hash,
            self.call_last,
            self.model_last,
            self.tokenizer,
            self.estimator,
            None if self.thinking_bound is None else dict(self.thinking_bound),
        )


def ref_kind_of(ref: str) -> str:
    """Classify a citation ref as "inference", "raw", "fold" or "span"."""
    if ref == "inference":
        return "inference"
    if ref.startswith("raw:"):
        return "raw"
    if ref.startswith("fold:"):
        return "fold"
    return "span"


def parse_fold_ref(ref: str) -> tuple[str, int] | None:
    """Split "fold:<fold_id>#<i>" into (fold_id, i), or None if malformed."""
    if not ref.startswith("fold:"):
        return None
    fold_id, sep, index = ref[len("fold:") :].rpartition("#")
    if not sep or not fold_id or not index.isdigit() or not index.isascii():
        return None
    return fold_id, int(index)


def parse_whole_fold_ref(ref: str) -> str | None:
    """The fold id of a whole-fold ref "fold:<fold_id>" (no "#"), or None."""
    if not ref.startswith("fold:") or "#" in ref or len(ref) == len("fold:"):
        return None
    return ref[len("fold:") :]


def fold_sentence(state: WindowState, ref: str) -> SummarySentence | None:
    """The sentence a "fold:<fold_id>#<i>" ref names in this state, or None if it does not resolve."""
    parsed = parse_fold_ref(ref)
    fold = state.folds.get(parsed[0]) if parsed else None
    return fold.sentences[parsed[1]] if fold is not None and parsed[1] < len(fold.sentences) else None


def fold_segment(fold: FoldRecord) -> Segment:
    """The summary segment a fold inserts: role user, ref_kind "fold", fidelity "summary"."""
    return Segment(fold.summary_segment_id, "user", "fold", fold.fold_id, "summary", False, fold.summary_hash, None)


def _str_list(value: object, what: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise TypeError(f"{what} must be a list of strings")
    return value


def _apply_source_registered(state: WindowState, p: dict, seq: int) -> None:
    source = Source.from_dict(p["source"])
    if source.source_id in state.sources:
        raise ReplayError(seq, f"source {source.source_id} registered twice")
    state.sources[source.source_id] = source


def _apply_source_updated(state: WindowState, p: dict, seq: int) -> None:
    source_id, content_hash = p["source_id"], p["content_hash"]
    if not isinstance(content_hash, str):
        raise TypeError("content_hash must be a str")
    if source_id not in state.sources:
        raise ReplayError(seq, f"update of unknown source {source_id}")
    state.sources[source_id] = replace(state.sources[source_id], content_hash=content_hash)


def _apply_span_registered(state: WindowState, p: dict, seq: int) -> None:
    span = Span.from_dict(p["span"])
    if span.source_id not in state.sources:
        raise ReplayError(seq, f"span of unknown source {span.source_id}")
    if span.span_id in state.spans:
        raise ReplayError(seq, f"span {span.span_id} registered twice")
    state.spans[span.span_id] = span


def _apply_segment_appended(state: WindowState, p: dict, seq: int) -> None:
    seg = Segment.from_dict(p["segment"])
    if seg.ref_kind == "fold" or seg.fidelity != "verbatim":
        raise ReplayError(seq, "only verbatim span or raw segments can be appended")
    if seg.ref_kind == "span" and seg.ref not in state.spans:
        raise ReplayError(seq, f"segment of unknown span {seg.ref}")
    if seg.parent is not None and seg.parent not in state.spans:
        raise ReplayError(seq, f"snippet of unknown parent span {seg.parent}")
    if any(s.segment_id == seg.segment_id for s in state.segments):
        raise ReplayError(seq, f"segment {seg.segment_id} already in the window")
    state.segments.append(seg)


def _apply_fold(state: WindowState, p: dict, seq: int) -> None:
    fold = FoldRecord.from_dict(p["fold"])
    if fold.fold_id in state.folds:
        raise ReplayError(seq, f"fold {fold.fold_id} applied twice")
    if not 0 <= fold.k <= len(state.segments):
        raise ReplayError(seq, f"fold index {fold.k} outside the window")
    window_ids = [s.segment_id for s in state.segments]
    victims = set(fold.victims)
    if not victims or len(victims) != len(fold.victims) or not victims <= set(window_ids):
        raise ReplayError(seq, "fold victims are empty, repeated or not in the window")
    if fold.summary_segment_id in window_ids:
        raise ReplayError(seq, f"fold segment {fold.summary_segment_id} already in the window")
    remaining = [s for s in state.segments if s.segment_id not in victims]
    remaining.insert(fold.k, fold_segment(fold))
    state.segments[:] = remaining
    state.folds[fold.fold_id] = fold
    for alias, target in fold.aliases:
        state.aliases.introduce(alias, target)
    state._down_cache.clear()


def _apply_evict(state: WindowState, p: dict, seq: int) -> None:
    ids = _str_list(p["segment_ids"], "segment_ids")
    if not isinstance(p["reason"], str):
        raise TypeError("reason must be a str")
    gone = set(ids)
    if len(gone) != len(ids) or not gone <= {s.segment_id for s in state.segments}:
        raise ReplayError(seq, "evicted segments are repeated or not in the window")
    state.segments[:] = [s for s in state.segments if s.segment_id not in gone]


def _apply_call(state: WindowState, p: dict, seq: int) -> None:
    call_id = p["call_id"]
    exposures = _str_list(p["exposures"], "exposures")
    if not isinstance(call_id, str) or not isinstance(p["manifest_hash"], str) or not isinstance(p["read_map"], dict):
        raise TypeError("call needs str call_id, str manifest_hash and dict read_map")
    if call_id in state.calls:
        raise ReplayError(seq, f"call {call_id} recorded twice")
    state.calls.append(call_id)
    state.ever_exposed.update(exposures)
    state.last_call_seq = seq
    state.call_last = state.segments[-1].segment_id if state.segments else None


def _apply_model_call(state: WindowState, p: dict, seq: int) -> None:
    if not isinstance(p["call_id"], str):
        raise TypeError("model_call needs a str call_id")
    bound = p.get("thinking_bound")  # BUILD-V4FIX: the bindings of the response's thinking blocks (request.thinking_bindings)
    if bound is not None and (not isinstance(bound, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in bound.items())):
        raise TypeError("model_call thinking_bound must map str keys to str digests")
    state.model_last = state.call_last
    if bound is not None:
        state.thinking_bound = {**(state.thinking_bound or {}), **bound}


def _apply_alias_assigned(state: WindowState, p: dict, seq: int) -> None:
    if not isinstance(p["alias"], str) or not isinstance(p["target"], str):
        raise TypeError("alias and target must be strings")
    state.aliases.introduce(p["alias"], p["target"])


def _apply_pricing_set(state: WindowState, p: dict, seq: int) -> None:
    if not isinstance(p["pricing"], dict) or not isinstance(p["pricing_hash"], str):
        raise TypeError("pricing_set needs a dict pricing and a str pricing_hash")
    state.pricing, state.pricing_hash = p["pricing"], p["pricing_hash"]


def _apply_tokenizer_set(state: WindowState, p: dict, seq: int) -> None:
    estimator = from_payload(p)  # validates before anything changes
    state.tokenizer, state.estimator = dict(p), estimator


_HANDLERS = {
    "source_registered": _apply_source_registered,
    "source_updated": _apply_source_updated,
    "span_registered": _apply_span_registered,
    "segment_appended": _apply_segment_appended,
    "fold": _apply_fold,
    "evict": _apply_evict,
    "call": _apply_call,
    "alias_assigned": _apply_alias_assigned,
    "pricing_set": _apply_pricing_set,
    "model_call": _apply_model_call,
    "tokenizer_set": _apply_tokenizer_set,
}


def apply_event(state: WindowState, event: Event) -> None:
    """Apply one event to `state` in place; other event types are ignored.

    Each handler validates before it mutates, so on ReplayError `state` is unchanged."""
    seq = getattr(event, "seq", None)
    try:
        handler = _HANDLERS.get(event.type)
        if handler is not None:
            if not isinstance(event.payload, dict):
                raise TypeError("payload must be a dict")
            handler(state, event.payload, seq)
    except ReplayError:
        raise
    except Exception as exc:
        raise ReplayError(seq, f"malformed {getattr(event, 'type', '?')} event: {exc!r}") from exc


def replay(events: Sequence[Event], upto_seq: int | None = None) -> WindowState:
    """Pure: rebuild the window by applying events in order, up to and including `upto_seq`.

    Raises ReplayError on an impossible event."""
    state = WindowState()
    for event in events:
        if upto_seq is not None and event.seq > upto_seq:
            break
        apply_event(state, event)
    return state


def state_at_call(events: Sequence[Event], call_id: str) -> WindowState:
    """The window state including the `call` event with this call_id. KeyError if absent."""
    events = list(events)
    for i, event in enumerate(events):
        if event.type == "call" and isinstance(event.payload, dict) and event.payload.get("call_id") == call_id:
            return replay(events[: i + 1])
    raise KeyError(call_id)


def down(state: WindowState, ref: str) -> frozenset[str]:
    """Provenance reduction to span ids: down(span) = {span}; down("fold:<id>#<i>") = union of
    down(c) over the cites of sentence i of that fold; down("fold:<id>") = union over all its
    sentences; raw, inference and unresolvable refs give {}. Cycle-safe via a visited set,
    memoized per state until the next fold."""
    cached = state._down_cache.get(ref)
    if cached is not None:
        return cached
    found: set[str] = set()
    visited: set[str] = set()
    stack = [ref]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        kind = ref_kind_of(current)
        if kind == "span":
            found.add(current)
        elif kind == "fold":
            sentence = fold_sentence(state, current)
            whole = state.folds.get(parse_whole_fold_ref(current) or "")
            if sentence is not None:
                stack.extend(sentence.cites)
            elif whole is not None:
                stack.extend(fold_sentence_refs(whole))
    result = frozenset(found)
    state._down_cache[ref] = result
    return result


def fold_sentence_refs(fold: FoldRecord) -> list[str]:
    """["fold:<id>#0", "fold:<id>#1", ...] for every sentence of `fold`."""
    return [f"fold:{fold.fold_id}#{i}" for i in range(len(fold.sentences))]


def _knowledge_sets(state: WindowState) -> tuple[set[str], set[str], set[str]]:
    """(verbatim span ids, snippet parent ids, summary-only span ids) for the current window."""
    verbatim: set[str] = set()
    parents: set[str] = set()
    summary: set[str] = set()
    for seg in state.segments:
        if seg.ref_kind == "span":
            verbatim.add(seg.ref)
        if seg.fidelity == "verbatim" and seg.parent is not None:
            parents.add(seg.parent)
        if seg.ref_kind == "fold" and seg.ref in state.folds:
            for ref in fold_sentence_refs(state.folds[seg.ref]):
                summary |= down(state, ref)
    return verbatim, parents, summary


def _kappa(state: WindowState, span_id: str, sets: tuple[set[str], set[str], set[str]]) -> Knowledge:
    span = state.spans.get(span_id)
    source = state.sources.get(span.source_id) if span is not None else None
    if source is not None and source.content_hash != span.source_hash:
        return Knowledge.STALE
    verbatim, parents, summary = sets
    if span_id in verbatim:
        return Knowledge.VERBATIM
    if span_id in parents:
        return Knowledge.SNIPPET
    if span_id in summary:
        return Knowledge.SUMMARY_ONLY
    if span_id in state.ever_exposed:
        return Knowledge.DROPPED
    return Knowledge.UNSEEN


def kappa(state: WindowState, span_id: str) -> Knowledge:
    """Knowledge state of `span_id` in this window, per the formal model."""
    return _kappa(state, span_id, _knowledge_sets(state))


def read_map(state: WindowState) -> dict[str, str]:
    """Map every registered span whose state is not unseen to its knowledge value."""
    sets = _knowledge_sets(state)
    out = {}
    for span_id in sorted(state.spans):
        k = _kappa(state, span_id, sets)
        if k is not Knowledge.UNSEEN:
            out[span_id] = k.value
    return out


def exposures(state: WindowState) -> list[str]:
    """Sorted span ids exposed verbatim: the refs of every span segment, snippets included."""
    return sorted({seg.ref for seg in state.segments if seg.ref_kind == "span"})


def manifest(state: WindowState) -> list[list]:
    """[[segment_id, role, ref_kind, ref, fidelity, pinned, text_hash, parent], ...] in window order."""
    return [
        [s.segment_id, s.role, s.ref_kind, s.ref, s.fidelity, s.pinned, s.text_hash, s.parent] for s in state.segments
    ]


def manifest_hash(state: WindowState) -> str:
    return sha256_hex(canonical_json(manifest(state)))


def approx_tokens(text: str) -> int:
    """ceil(len(UTF-8 bytes) / 4)."""
    return (len(text.encode("utf-8")) + 3) // 4


def segment_tokens(segment: Segment, blobs: BlobStore, state: WindowState | None = None) -> int:
    """Tokens of the segment's blob text: approx_tokens without a state (M1), else the count of
    the estimator in force in `state` for the segment's class (bytes4-v0 gives the same)."""
    text = blobs.get(segment.text_hash).decode("utf-8")
    if state is None:
        return approx_tokens(text)
    return estimator_for(state).count(text, segment_class(state, segment))


def window_tokens(state: WindowState, blobs: BlobStore) -> int:
    """Token count of every segment's blob text, with the estimator in force."""
    return sum(segment_tokens(seg, blobs, state) for seg in state.segments)
