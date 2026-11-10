"""Fold records: replacing window segments with a cited summary, by the model or the backstop."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from .alias import AliasMap, alias_for
from .ids import new_id
from .model import Event, FoldRecord, PyAnchor, Segment, SummarySentence, TRIGGERS
from .render import content_blocks, fold_item, rendered_segment_tokens, splits_tool_block, window_costs, window_items
from .store.blobs import BlobStore
from .tokens import envelope_of, estimator_for
from .window import WindowState, fold_sentence_refs, segment_tokens, window_tokens

ROLE_LABEL = re.compile(r"^\s*(system|user|assistant|human|tool)\s*:", re.IGNORECASE | re.MULTILINE)
_FIRST_LINE_MAX = 120


class FoldError(Exception):
    """A fold violates a precondition (victims, pinning, cites, role labels or no-growth).

    `reason` is a short machine-readable code ("fold_would_grow_window", ...) when one applies,
    and `details` carries its numbers."""

    def __init__(self, message: str, reason: str | None = None, **details: object) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details


def has_role_label(text: str) -> bool:
    """True if any line of `text` looks like a chat role label ("user:", "System :", ...)."""
    return ROLE_LABEL.search(text) is not None


def refs_of(segment: Segment, state: WindowState) -> set[str]:
    """Refs a summary of `segment` may cite: a span segment gives its span id (and its parent
    if set); a fold segment gives "fold:<id>#<i>" per sentence; a raw one "raw:<segment_id>"."""
    if segment.ref_kind == "span":
        return {segment.ref} | ({segment.parent} if segment.parent else set())
    if segment.ref_kind == "fold":
        fold = state.folds.get(segment.ref)
        return set(fold_sentence_refs(fold)) if fold is not None else set()
    return {f"raw:{segment.segment_id}"}


def cite_refs_of(segment: Segment, state: WindowState) -> set[str]:
    """Refs a fold sentence summarizing `segment` may cite: `refs_of`, plus the whole-fold ref
    "fold:<id>" of a fold segment (BUILD-V2c: a refold names the earlier fold by its one alias;
    down() of it is every span behind every sentence)."""
    return refs_of(segment, state) | ({f"fold:{segment.ref}"} if segment.ref_kind == "fold" else set())


def fold_id_for(trigger: str, k: int, victims: Sequence[str]) -> str:
    """Deterministic fold id; unique because a segment can be a fold victim only once."""
    return new_id("fold", trigger, str(k), *victims)


def render_fold_text(fold_id: str, sentences: Sequence[SummarySentence]) -> str:
    """M1 format: header line "[fold <fold_id>]" plus one line "- <text> [<cites>]" per sentence."""
    lines = [f"[fold {fold_id}]"] + [f"- {s.text} [{', '.join(s.cites)}]" for s in sentences]
    return "\n".join(lines)


def plan_fold_aliases(aliases: AliasMap, fold_id: str, sentences: Sequence[SummarySentence]) -> tuple[tuple[str, str], ...]:
    """The [alias, target] pairs a new fold introduces, in counter order: its whole-fold handle
    "fold:<fold_id>", one per sentence "fold:<fold_id>#<i>", then each cited ref (not
    "inference") that has no alias yet, in order of first citation."""
    targets = [f"fold:{fold_id}"] + [f"fold:{fold_id}#{i}" for i in range(len(sentences))]
    seen = set(targets)
    for sentence in sentences:
        for cite in sentence.cites:
            if cite != "inference" and cite not in seen and aliases.alias_of(cite) is None:
                targets.append(cite)
                seen.add(cite)
    return tuple((alias_for(aliases.count + n), target) for n, target in enumerate(targets))


def render_fold(
    fold_id: str,
    sentences: Sequence[SummarySentence],
    fold_aliases: Sequence[tuple[str, str]] = (),
    aliases: AliasMap | None = None,
) -> str:
    """The fold segment's text. Without `fold_aliases` it is the M1 format. With them:
    "[fold @A]" then one line "- (@B) <text> [@3 @7]" per sentence, where every ref is shown
    by its alias (the fold's own aliases first, then `aliases`); "inference" stays as is."""
    if not fold_aliases:
        return render_fold_text(fold_id, sentences)
    planned = {target: alias for alias, target in fold_aliases}

    def show(ref: str) -> str:
        return planned.get(ref) or (aliases.alias_of(ref) if aliases is not None else None) or ref

    text = fold_header(show(f"fold:{fold_id}"))
    for i, s in enumerate(sentences):
        text += fold_line(show(f"fold:{fold_id}#{i}"), s.text, " ".join(show(c) for c in s.cites))
    return text


def fold_header(alias: str) -> str:
    """First line of an aliased fold text."""
    return f"[fold {alias}]"


def fold_line(sentence_alias: str, text: str, cites: str) -> str:
    """One sentence line of an aliased fold text, with its leading newline."""
    return f"\n- ({sentence_alias}) {text} [{cites}]"


def fold_delta(state: WindowState, victims: Sequence[Segment], summary_text: str, blobs: BlobStore, *, aliased: bool, k: int | None = None) -> int:
    """Tokens a fold frees: victims minus the summary (class "other"), with the estimator in
    force. Aliased (v1) folds count rendered tokens (alias headers included); M1 folds keep
    M1's measure on the blob text (bytes4-v0 unless a tokenizer_set says otherwise). Under
    calib-v3 it is the drop of the whole window measure (render.rendered_window_tokens) when the
    victims leave and the summary enters at index `k` (default: the first victim's), envelopes
    included."""
    counter = estimator_for(state)
    env = envelope_of(counter)
    if aliased and env is not None:
        gone = {seg.segment_id for seg in victims}
        index = {seg.segment_id: i for i, seg in enumerate(state.segments)}
        k = min(index[v] for v in gone) if k is None else k
        items = window_items(state, blobs)
        after = [item for item, seg in zip(items, state.segments) if seg.segment_id not in gone]
        after.insert(k, fold_item(summary_text))
        return sum(window_costs(items, counter, env)) - sum(window_costs(after, counter, env))
    summary = counter.count(summary_text, "other")
    if aliased:
        return sum(rendered_segment_tokens(state, seg, blobs) for seg in victims) - summary
    return sum(segment_tokens(seg, blobs, state) for seg in victims) - summary


def make_fold(
    state: WindowState,
    k: int,
    victim_ids: Sequence[str],
    sentences: Sequence[SummarySentence],
    trigger: str,
    policy: dict | None = None,
    *,
    blobs: BlobStore,
    aliased: bool = False,
) -> FoldRecord:
    """Build (and store the summary blob of) a fold of `victim_ids` inserted at index `k`.

    Raises FoldError unless: victims are non-empty, distinct and all in the window at index
    >= k, with k <= len(window); no victim is pinned; every cite is in the union of
    cite_refs_of the victims plus "inference"; no sentence contains a role-label line; and the
    policy is None for model folds and a dict for backstop folds. Victims are stored in window order.
    No-growth rule, for every trigger and policy: the fold must free tokens, Δ =
    `fold_delta(...)` > 0, else FoldError with reason "fold_would_grow_window".

    With `aliased`, the fold introduces aliases (plan_fold_aliases) and its text cites them."""
    if trigger not in TRIGGERS:
        raise FoldError(f"unknown trigger {trigger!r}")
    if (trigger == "model") != (policy is None):
        raise FoldError("model folds take no policy; backstop folds record theirs")
    victims = list(victim_ids)
    if not victims or len(set(victims)) != len(victims):
        raise FoldError("victims must be non-empty and distinct")
    if not 0 <= k <= len(state.segments):
        raise FoldError(f"k={k} outside the window of {len(state.segments)} segments")
    index = {seg.segment_id: i for i, seg in enumerate(state.segments)}
    for victim in victims:
        if victim not in index:
            raise FoldError(f"victim {victim} is not in the window")
        if index[victim] < k:
            raise FoldError(f"victim {victim} sits before k={k}")
    victims.sort(key=index.__getitem__)
    victim_segments = [state.segments[index[v]] for v in victims]
    if any(seg.pinned for seg in victim_segments):
        raise FoldError("pinned segments cannot be folded", "pinned")
    if splits_tool_block(state, blobs, set(victims), k):
        raise FoldError("a fold must take a tool use and all its results, or none of them", "tool_pair")
    allowed = set().union(*(cite_refs_of(seg, state) for seg in victim_segments)) | {"inference"}
    sentences = tuple(sentences)
    for sentence in sentences:
        outside = set(sentence.cites) - allowed
        if outside:
            raise FoldError(f"cites outside the victims: {sorted(outside)}", "cites_outside_victims", cites=sorted(outside))
        if has_role_label(sentence.text):
            raise FoldError(f"sentence contains a role label: {sentence.text!r}", "role_label")
    fold_id = fold_id_for(trigger, k, victims)
    fold_aliases = plan_fold_aliases(state.aliases, fold_id, sentences) if aliased else ()
    text = render_fold(fold_id, sentences, fold_aliases, state.aliases)
    freed = fold_delta(state, victim_segments, text, blobs, aliased=aliased, k=k)
    if freed <= 0:
        raise FoldError(f"fold_would_grow_window delta={freed}", "fold_would_grow_window", delta=freed)
    summary_hash = blobs.put(text.encode("utf-8"))
    return FoldRecord(
        fold_id=fold_id,
        k=k,
        victims=tuple(victims),
        sentences=sentences,
        trigger=trigger,
        policy=None if policy is None else dict(policy),
        freed_tokens=freed,
        summary_segment_id=new_id("seg", "fold", fold_id),
        summary_hash=summary_hash,
        aliases=fold_aliases,
    )


@dataclass(frozen=True, kw_only=True)
class BackstopPolicy:
    """Legacy backstop "backstop-v0" (M1), kept so M1 ledgers replay and P9 can re-run their folds.

    Fold when the window exceeds hard * budget, down to target * budget. `to_dict` writes `hard`
    and `target` as repr() strings, as M1 recorded them. New sessions use BackstopPolicyV1,
    whose ratios are integer basis points."""

    id: str = "backstop-v0"
    budget_tokens: int
    hard: float = 0.85
    target: float = 0.50
    tail_keep: int = 2
    cold_m: int = 4

    def __post_init__(self) -> None:
        if self.budget_tokens <= 0 or not 0 < self.target <= self.hard or self.tail_keep < 0 or self.cold_m < 0:
            raise ValueError(f"invalid backstop policy {self!r}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "budget_tokens": self.budget_tokens,
            "hard": repr(float(self.hard)),
            "target": repr(float(self.target)),
            "tail_keep": self.tail_keep,
            "cold_m": self.cold_m,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BackstopPolicy | BackstopPolicyV1:
        """Decode any recorded policy, dispatching on its "id" (see `policy_from_dict`)."""
        return policy_from_dict(d)


_V0_KEYS = frozenset({"id", "budget_tokens", "hard", "target", "tail_keep", "cold_m"})


def _decode_v0(d: dict) -> BackstopPolicy:
    """M1 wrote the ratios as repr() strings of floats. Each is parsed exactly with Fraction(str)
    (so "nan", "inf" or junk are rejected) and then held as the float M1 computed with, so a
    re-run of select_backstop reproduces M1's decisions bit for bit."""
    if set(d) != _V0_KEYS:
        raise ValueError(f"backstop-v0 policy keys must be {sorted(_V0_KEYS)}, got {sorted(d)}")
    ratios = {}
    for key in ("hard", "target"):
        if not isinstance(d[key], str):
            raise TypeError(f"backstop-v0 {key} must be a string")
        ratios[key] = float(Fraction(d[key]))
    for key in ("budget_tokens", "tail_keep", "cold_m"):
        _check_int(d[key], key)
    return BackstopPolicy(budget_tokens=d["budget_tokens"], tail_keep=d["tail_keep"], cold_m=d["cold_m"], **ratios)


def _check_int(value: object, what: str, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{what} must be an int >= {minimum}, got {value!r}")


_ESTIMATOR_KEYS = {"fixed": "n", "lindy": "cap", "horizon": "planned_calls"}


@dataclass(frozen=True)
class ReReadEstimator:
    """Expected number of future calls that will read the window, R-hat, including the next one.

    fixed n:               R-hat = n
    lindy cap:             R-hat = clamp(calls_so_far, 1, cap)   (as many future calls as past)
    horizon planned_calls: R-hat = max(1, planned_calls - calls_so_far)"""

    kind: str = "lindy"
    value: int = 64

    def __post_init__(self) -> None:
        if self.kind not in _ESTIMATOR_KEYS:
            raise ValueError(f"unknown estimator kind {self.kind!r}")
        _check_int(self.value, f"estimator {_ESTIMATOR_KEYS[self.kind]}", 0 if self.kind == "horizon" else 1)

    def rhat(self, calls_so_far: int) -> int:
        if self.kind == "fixed":
            return self.value
        if self.kind == "lindy":
            return min(max(calls_so_far, 1), self.value)
        return max(1, self.value - calls_so_far)

    def to_dict(self) -> dict:
        return {"kind": self.kind, _ESTIMATOR_KEYS[self.kind]: self.value}

    @classmethod
    def from_dict(cls, d: dict) -> ReReadEstimator:
        if not isinstance(d, dict) or d.get("kind") not in _ESTIMATOR_KEYS:
            raise ValueError(f"bad estimator {d!r}")
        key = _ESTIMATOR_KEYS[d["kind"]]
        if set(d) != {"kind", key}:
            raise ValueError(f"estimator {d['kind']} takes exactly the keys kind and {key}")
        return cls(d["kind"], d[key])


TTLS = frozenset({"5m", "1h"})
TOKEN_COUNTERS = frozenset({"approx-bytes4", "calib-v1", "calib-v2", "calib-v3"})
RULES = frozenset({"v2b", "v2c"})


@dataclass(frozen=True, kw_only=True)
class BackstopPolicyV1:
    """Payoff-gated backstop "backstop-v1". Every ratio is integer basis points (10000 = 100%),
    so nothing hashed or compared for a decision is a float.

    Zones against budget_tokens: soft < hard < mandatory; a fold aims for target. q0_bp,
    half_life_calls and tau_tokens price the risk of folding evidence that is later re-fetched;
    ttl, model and the estimator price the cache effect (see backstop.py). `token_counter`
    names the token estimator in force in the ledger ("approx-bytes4" is bytes4-v0).

    `rules` picks the selection and summary rules: "v2b" (the default, every ledger before
    BUILD-V2c) or "v2c", the bounded floor (backstop.py, "Bounded floor"). `to_dict` writes it
    only when it is "v2c", so v2b policies, and the folds that recorded them, are unchanged."""

    id: str = "backstop-v1"
    budget_tokens: int
    soft_bp: int = 7000
    hard_bp: int = 8500
    mandatory_bp: int = 9500
    target_bp: int = 5000
    tail_keep: int = 2
    cold_m: int = 4
    q0_bp: int = 2000
    half_life_calls: int = 4
    tau_tokens: int = 150
    ttl: str = "5m"
    model: str = "claude-haiku-5-5"
    estimator: ReReadEstimator = ReReadEstimator()
    token_counter: str = "approx-bytes4"
    rules: str = "v2b"

    _INTS = (
        "budget_tokens", "soft_bp", "hard_bp", "mandatory_bp", "target_bp", "tail_keep", "cold_m",
        "q0_bp", "half_life_calls", "tau_tokens",
    )  # fmt: skip

    def __post_init__(self) -> None:
        if self.id != "backstop-v1":
            raise ValueError(f"BackstopPolicyV1 id must be 'backstop-v1', got {self.id!r}")
        for name in self._INTS:
            _check_int(getattr(self, name), name)
        _check_int(self.budget_tokens, "budget_tokens", 1)
        _check_int(self.half_life_calls, "half_life_calls", 1)
        if not 0 < self.target_bp < self.soft_bp <= self.hard_bp < self.mandatory_bp <= 10000:
            raise ValueError("need 0 < target_bp < soft_bp <= hard_bp < mandatory_bp <= 10000")
        if self.q0_bp > 10000:
            raise ValueError("q0_bp is a probability in basis points, at most 10000")
        if self.ttl not in TTLS:
            raise ValueError(f"ttl must be one of {sorted(TTLS)}")
        if self.token_counter not in TOKEN_COUNTERS:
            raise ValueError(f"token_counter must be one of {sorted(TOKEN_COUNTERS)}")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.estimator, ReReadEstimator):
            raise TypeError("estimator must be a ReReadEstimator")
        if self.rules not in RULES:
            raise ValueError(f"rules must be one of {sorted(RULES)}")

    @property
    def bounded_floor(self) -> bool:
        """The v2c rules (backstop.py, "Bounded floor")."""
        return self.rules == "v2c"

    def to_dict(self) -> dict:
        d = {name: getattr(self, name) for name in self._INTS}
        return {
            "id": self.id,
            **d,
            "ttl": self.ttl,
            "model": self.model,
            "estimator": self.estimator.to_dict(),
            "token_counter": self.token_counter,
            **({"rules": self.rules} if self.rules != "v2b" else {}),
        }

    @classmethod
    def from_dict(cls, d: dict) -> BackstopPolicy | BackstopPolicyV1:
        return policy_from_dict(d)


_V1_KEYS = frozenset({"id", *BackstopPolicyV1._INTS, "ttl", "model", "estimator", "token_counter"})


def _decode_v1(d: dict) -> BackstopPolicyV1:
    if set(d) - {"rules"} != _V1_KEYS:
        raise ValueError(f"backstop-v1 policy keys must be {sorted(_V1_KEYS)} (and optionally rules), got {sorted(d)}")
    if d.get("rules") == "v2b":
        raise ValueError("backstop-v1 policies record rules only when they are not v2b")
    for key in ("ttl", "model", "token_counter", *(("rules",) if "rules" in d else ())):
        if not isinstance(d[key], str):
            raise TypeError(f"backstop-v1 {key} must be a string")
    fields = {k: v for k, v in d.items() if k not in ("id", "estimator")}
    return BackstopPolicyV1(**fields, estimator=ReReadEstimator.from_dict(d["estimator"]))


def policy_from_dict(d: dict) -> BackstopPolicy | BackstopPolicyV1:
    """Decode a recorded policy dict, dispatching on "id": "backstop-v0" gives the legacy class
    (string ratios decoded exactly), "backstop-v1" the basis-point class. Unknown or missing
    keys raise ValueError; wrongly typed values raise ValueError or TypeError."""
    if not isinstance(d, dict):
        raise TypeError("policy must be a dict")
    if d.get("id") == "backstop-v0":
        return _decode_v0(d)
    if d.get("id") == "backstop-v1":
        return _decode_v1(d)
    raise ValueError(f"unknown backstop policy id {d.get('id')!r}")


def _first_line(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line.strip()[:_FIRST_LINE_MAX]
    return ""


def _plain_text(seg: Segment, blobs: BlobStore) -> str:
    """A segment's text; for content blocks, its text blocks then "[tool_use <name>]" per tool
    use. Thinking blocks are never rendered into summaries."""
    if seg.content_type != "blocks":
        return blobs.get(seg.text_hash).decode("utf-8")
    shown = [b for b in content_blocks(seg, blobs) if b["type"] in ("text", "tool_use")]
    parts = [b["text"] if b["type"] == "text" else f"[tool_use {b['name']}]" for b in shown]
    return " ".join(part.replace("\n", " ") for part in parts)


def _escape_role_labels(text: str) -> str:
    """Prefix "> " to every line at which ROLE_LABEL would match, so has_role_label is False."""
    lines = text.split("\n")
    return "\n".join("> " + line if ROLE_LABEL.match("\n".join(lines[i:])) else line for i, line in enumerate(lines))


def extractive_summary_v1(
    state: WindowState, victims: Sequence[str], blobs: BlobStore, *, rules: str = "v2b", digest: bool = False
) -> list[SummarySentence]:
    """The backstop-v1 summary: M1's one deterministic sentence per victim, except that an
    earlier fold is named by its alias ("earlier fold @A (<n> sentences)") when it has one.

    The v2c rules (`rules`, BUILD-V2c): that sentence cites the earlier fold as a whole,
    "fold:<id>", shown as the one alias @A, not each of its n sentences, so a refold line has
    the same size however long the earlier fold was. With `digest` (a v2c decision of the
    mandatory zone records when it chose one, backstop.py "Bounded floor") the run is one
    digest sentence instead (`digest_sentence`), the summary-of-summaries."""
    by_id = {seg.segment_id: seg for seg in state.segments}
    segments = [by_id[v] for v in victims]
    if digest:
        return [digest_sentence(segments)]
    sentences = extractive_summary(state, victims, blobs)
    for n, seg in enumerate(segments):
        alias = state.aliases.alias_of(f"fold:{seg.ref}") if seg.ref_kind == "fold" else None
        if alias is not None:
            text = f"earlier fold {alias} ({len(state.folds[seg.ref].sentences)} sentences)"
            sentences[n] = SummarySentence(text, (f"fold:{seg.ref}",) if rules == "v2c" else sentences[n].cites)
    return sentences


def digest_text(folds: int, messages: int, spans: int) -> str:
    """"earlier context: 2 folds, 1 message, 3 spans" (parts with a count of 0 left out)."""
    parts = [f"{n} {word}{'' if n == 1 else 's'}" for n, word in ((folds, "fold"), (messages, "message"), (spans, "span")) if n]
    return "earlier context: " + ", ".join(parts)


def digest_cite(segment: Segment) -> str | None:
    """What a digest cites for a victim: an earlier fold as a whole ("fold:<id>"), a span by
    its id, nothing for a raw message (it has no span behind it)."""
    return f"fold:{segment.ref}" if segment.ref_kind == "fold" else segment.ref if segment.ref_kind == "span" else None


def digest_sentence(segments: Sequence[Segment]) -> SummarySentence:
    """The one sentence of a v2c digest fold (BUILD-V2c; over earlier folds it is the
    summary-of-summaries): how many folds, messages and spans it
    replaces, citing every earlier fold as a whole and every span, each by its one alias.
    down() of it still reaches every span the run held, so they stay summary_only and
    ctx_refetch of any earlier alias works; a raw message has no span, so it is only counted."""
    folds = sum(seg.ref_kind == "fold" for seg in segments)
    spans = sum(seg.ref_kind == "span" for seg in segments)
    text = digest_text(folds, len(segments) - folds - spans, spans)
    return SummarySentence(text, tuple(c for c in map(digest_cite, segments) if c is not None))


def extractive_summary(state: WindowState, victims: Sequence[str], blobs: BlobStore) -> list[SummarySentence]:
    """One deterministic sentence per victim (window segment ids), role labels escaped:
    span -> "<uri>#<qualname or start-end>: <first non-empty line>" citing the span id;
    fold -> "earlier fold <id> (<n> sentences)" citing all its sentence refs;
    raw  -> "<role> message: <first line>" citing "raw:<segment_id>"."""
    by_id = {seg.segment_id: seg for seg in state.segments}
    sentences = []
    for victim in victims:
        seg = by_id[victim]
        first = _first_line(_plain_text(seg, blobs))
        if seg.ref_kind == "span":
            span = state.spans[seg.ref]
            anchor = span.anchor
            label = anchor.qualname if isinstance(anchor, PyAnchor) else f"{anchor.start}-{anchor.end}"
            uri = state.sources[span.source_id].uri
            text = f"{uri}#{label}: {first}"
            cites: tuple[str, ...] = (span.span_id,)
        elif seg.ref_kind == "fold":
            fold = state.folds[seg.ref]
            text = f"earlier fold {fold.fold_id} ({len(fold.sentences)} sentences)"
            cites = tuple(fold_sentence_refs(fold))
        else:
            text = f"{seg.role} message: {first}"
            cites = (f"raw:{seg.segment_id}",)
        sentences.append(SummarySentence(_escape_role_labels(text), cites))
    return sentences


def segment_append_seqs(events: Sequence[Event]) -> dict[str, int]:
    """Seq at which each segment entered the window (segment_appended, or the fold that made it)."""
    seqs: dict[str, int] = {}
    for event in events:
        if event.type == "segment_appended":
            seqs[event.payload["segment"]["segment_id"]] = event.seq
        elif event.type == "fold":
            seqs[event.payload["fold"]["summary_segment_id"]] = event.seq
    return seqs


def select_backstop(
    state: WindowState, events: Sequence[Event], blobs: BlobStore, policy: BackstopPolicy
) -> tuple[int, list[str]] | None:
    """Pure, deterministic choice of (k, victims), or None.

    None unless window tokens exceed hard * budget. Candidates are unpinned segments outside
    the last `tail_keep`, appended at or before the cold_m-th most recent call (all segments
    count as recent while fewer than cold_m calls exist), and not cited by any claim of the
    last cold_m calls. k is the oldest candidate's index; victims are candidates from k on,
    in window order, added until projected tokens (window - victims + summary) <= target *
    budget or candidates run out."""
    total = window_tokens(state, blobs)
    if total <= policy.hard * policy.budget_tokens:
        return None
    calls = [e for e in events if e.type == "call"]
    if policy.cold_m == 0:
        boundary, recent_calls = float("inf"), set()
    elif len(calls) < policy.cold_m:
        return None
    else:
        boundary = calls[-policy.cold_m].seq
        recent_calls = {e.payload["call_id"] for e in calls[-policy.cold_m :]}
    cited = {
        c["ref"]
        for e in events
        if e.type == "claim" and e.payload["claim"]["call_id"] in recent_calls
        for c in e.payload["claim"]["citations"]
    }
    appended = segment_append_seqs(events)
    segments = state.segments
    candidates = [
        i
        for i, seg in enumerate(segments)
        if not seg.pinned
        and i < len(segments) - policy.tail_keep
        and appended.get(seg.segment_id, 0) <= boundary
        and not refs_of(seg, state) & cited
    ]
    if not candidates:
        return None
    k = candidates[0]
    victims: list[str] = []
    freed = 0
    estimator = estimator_for(state)
    for i in candidates:
        victims.append(segments[i].segment_id)
        freed += segment_tokens(segments[i], blobs, state)
        summary = render_fold_text(fold_id_for("backstop", k, victims), extractive_summary(state, victims, blobs))
        if total - freed + estimator.count(summary, "other") <= policy.target * policy.budget_tokens:
            break
    return k, victims


def payback_calls(L, k, delta, alpha_w, alpha_r, g_over_p=0) -> Fraction:
    """Additional calls, after the call that rewrites the cache, until a fold pays back, exactly:
    R* = ((L - k - delta) * (alpha_w - alpha_r) + g_over_p) / (delta * alpha_r) - 1.

    Arguments may be ints, Fractions, Decimals or floats; each is converted with Fraction() (a
    float exactly as stored), so the result is an exact Fraction."""
    L, k, delta, alpha_w, alpha_r, g_over_p = map(Fraction, (L, k, delta, alpha_w, alpha_r, g_over_p))
    return ((L - k - delta) * (alpha_w - alpha_r) + g_over_p) / (delta * alpha_r) - 1
