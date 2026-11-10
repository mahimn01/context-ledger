"""Session: the mock harness that writes a CLP ledger and keeps every invariant by construction.

Every write goes through `_append`, which first applies the event to the in-memory window
(raising before anything is written if the event is impossible), then appends it to the
ledger. Methods raise SessionError or FoldError instead of writing an event that would
violate an invariant; they never patch the ledger afterwards.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from .alias import UnknownAlias, is_alias
from .anchor.python import python_anchors, resolve_python
from .anchor.text import Resolution, make_text_anchor, normalize_newlines, resolve_text
from .backstop import Analysis, BudgetExceeded, FoldRejected, PayoffModel
from .fold import (
    BackstopPolicy,
    BackstopPolicyV1,
    FoldError,
    extractive_summary,
    extractive_summary_v1,
    fold_delta,
    fold_id_for,
    make_fold,
    plan_fold_aliases,
    render_fold,
    select_backstop,
)
from .ids import canonical_json, new_id, sha256_hex, span_id
from .model import ROLES, Anchor, Citation, Claim, Event, FoldRecord, PyAnchor, Segment, Source, Span
from .pricing import PricingTable
from .render import TokenCounter, content_blocks, empty_text_block, render_request, splits_tool_block
from .store.blobs import BlobStore
from .store.ledger import Ledger
from .tokens import Estimator, estimator_for
from .verify import RefetchRequired, Verdict, verify_claim
from .verify.gate import GateVerdict, answer_gate
from .window import (
    apply_event,
    down,
    exposures,
    fold_sentence,
    kappa,
    manifest_hash,
    parse_fold_ref,
    parse_whole_fold_ref,
    read_map,
    ref_kind_of,
    replay,
    state_at_call,
)


_HEADER_REF = re.compile(r"\[(@[0-9A-HJKMNP-TV-Z]+)(?:\s[^\]]*)?\]|(@[0-9A-HJKMNP-TV-Z]+)\s.*", re.S)


class SessionError(Exception):
    """The requested operation would break a ledger invariant."""


@dataclass(frozen=True)
class CallResult:
    call_id: str
    request: dict
    read_map: dict[str, str]


def unmet_refetch_obligations(events: Sequence[Event], state, citations: Sequence[Citation]) -> list[str]:
    """Fold-sentence cites that lack a `refetch` event, after their fold, for some span in the
    sentence's down set (P6). `state` must know the cited folds."""
    fold_seqs = {e.payload["fold"]["fold_id"]: e.seq for e in events if e.type == "fold"}
    refetches = [(e.seq, e.payload["span_id"]) for e in events if e.type == "refetch"]
    unmet = []
    for citation in citations:
        parsed = parse_fold_ref(citation.ref)
        if parsed is None:
            continue
        fold_seq = fold_seqs.get(parsed[0])
        reach = down(state, citation.ref)
        if fold_seq is None or not any(seq > fold_seq and sid in reach for seq, sid in refetches):
            unmet.append(citation.ref)
    return unmet


class Session:
    """Harness over a ledger and blob store with a token budget and a backstop policy.

    `policy` defaults to BackstopPolicyV1(budget_tokens=budget_tokens); pass a
    BackstopPolicy (backstop-v0) for an M1-style session. A v1 session is "aliased": every
    span appended to the window, every raw segment and every fold gets a short alias, fold
    summaries cite aliases, and the backstop is payoff-gated, so it needs a price table.

    `pricing`, if given, is recorded with a `pricing_set` event unless the same table is
    already in force in the ledger. A v1 session raises ValueError if no table is in force.

    `tokenizer`, if given, is recorded the same way with a `tokenizer_set` event; without one
    the ledger counts tokens with bytes4-v0. A v1 policy's `token_counter` must name the
    estimator in force ("approx-bytes4" for bytes4-v0, "calib-v1", "calib-v2")."""

    gate = None  # the answer gate emit_answer runs when set (the daily harness wraps answer_gate per session); None: answer_gate

    def __init__(
        self,
        ledger: Ledger,
        blobs: BlobStore,
        budget_tokens: int,
        policy: BackstopPolicy | BackstopPolicyV1 | None = None,
        *,
        pricing: PricingTable | None = None,
        tokenizer: Estimator | None = None,
    ) -> None:
        self.policy = policy if policy is not None else BackstopPolicyV1(budget_tokens=budget_tokens)
        if self.policy.budget_tokens != budget_tokens:
            raise ValueError("policy.budget_tokens must equal budget_tokens")
        self.ledger = ledger
        self.blobs = blobs
        self.budget_tokens = budget_tokens
        self.aliased = isinstance(self.policy, BackstopPolicyV1)
        self.last_backstop: Analysis | None = None
        self.last_window: int | None = None
        self.last_folds: list[FoldRecord] = []
        self.events: list[Event] = ledger.events()
        self.state = replay(self.events)
        if pricing is not None:
            self.set_pricing(pricing)
        if tokenizer is not None:
            self.set_tokenizer(tokenizer)
        if self.aliased and self.state.pricing is None:
            raise ValueError("a backstop-v1 session needs a price table: pass pricing=PricingTable.load(...)")
        if self.aliased and self.policy.token_counter != self.counter.name:
            raise ValueError(f"policy token_counter {self.policy.token_counter!r} is not the estimator in force ({self.counter.name!r})")

    @property
    def counter(self) -> TokenCounter | None:
        """The token estimator in force (v1 sessions; None for an M1-style session)."""
        return estimator_for(self.state) if self.aliased else None

    @property
    def pricing(self) -> PricingTable | None:
        """The price table in force at the end of the ledger."""
        return None if self.state.pricing is None else PricingTable.from_dict(self.state.pricing)

    def set_pricing(self, table: PricingTable) -> None:
        """Put `table` in force (a `pricing_set` event) unless it already is."""
        if table.pricing_hash() != self.state.pricing_hash:
            self._append("pricing_set", {"pricing": table.to_dict(), "pricing_hash": table.pricing_hash()})

    def set_tokenizer(self, estimator: Estimator) -> None:
        """Put `estimator` in force (a `tokenizer_set` event) unless it already is."""
        if self.state.tokenizer != estimator.payload():
            self._append("tokenizer_set", estimator.payload())

    def _append(self, type_: str, payload: dict) -> Event:
        canonical_json(payload)  # reject non-canonical payloads before touching state
        apply_event(self.state, Event(len(self.events) + 1, type_, payload, "", ""))
        event = self.ledger.append(type_, payload)
        self.events.append(event)
        return event

    def _text(self, h: str) -> str:
        return self.blobs.get(h).decode("utf-8")

    def _source(self, source_id: str) -> Source:
        source = self.state.sources.get(source_id)
        if source is None:
            raise SessionError(f"unknown source {source_id}")
        return source

    def _ref_known(self, ref: str) -> bool:
        """A citable canonical ref: registered span, fold sentence, whole fold, raw segment or "inference"."""
        kind = ref_kind_of(ref)
        if kind == "span":
            return ref in self.state.spans
        if kind == "fold":
            return fold_sentence(self.state, ref) is not None or parse_whole_fold_ref(ref) in self.state.folds
        if kind == "raw":
            raw_id = ref[len("raw:") :]
            return any(
                e.type == "segment_appended" and e.payload["segment"]["segment_id"] == raw_id and e.payload["segment"]["ref_kind"] == "raw"
                for e in self.events
            )
        return True

    def assign_alias(self, target: str) -> str:
        """The alias of a canonical ref, introducing it (an `alias_assigned` event) if new."""
        existing = self.state.aliases.alias_of(target)
        if existing is not None:
            return existing
        if target == "inference" or not self._ref_known(target):
            raise SessionError(f"cannot alias unknown target {target!r}")
        alias = self.state.aliases.next_alias()
        self._append("alias_assigned", {"alias": alias, "target": target})
        return alias

    # Sources and spans

    def register_source(self, uri: str, text: str, media_type: str = "text/plain", version: str = "v1") -> Source:
        """Register a source; its newline-normalized text is stored as a blob."""
        source_id = new_id("src", uri, version)
        if source_id in self.state.sources:
            raise SessionError(f"source {uri} {version} already registered")
        content_hash = self.blobs.put(normalize_newlines(text).encode("utf-8"))
        source = Source(source_id, uri, version, content_hash, media_type)
        self._append("source_registered", {"source": source.to_dict()})
        return source

    def _register_span(self, source_id: str, content_hash: str, anchor: Anchor, text: str) -> Span:
        sid = span_id(content_hash, anchor.to_dict())
        if sid in self.state.spans:
            return self.state.spans[sid]
        exact = text[anchor.start : anchor.end]
        span = Span(sid, source_id, content_hash, anchor, self.blobs.put(exact.encode("utf-8")), len(exact))
        self._append("span_registered", {"span": span.to_dict()})
        return span

    def register_text_span(self, source: Source, start: int, end: int) -> Span:
        """Register [start, end) of the source's current text (idempotent)."""
        current = self._source(source.source_id)
        text = self._text(current.content_hash)
        return self._register_span(current.source_id, current.content_hash, make_text_anchor(text, start, end), text)

    def register_python_spans(self, source: Source, path: str, commit: str | None = None) -> list[Span]:
        """Register a span for every Python symbol of the source's current text."""
        current = self._source(source.source_id)
        text = self._text(current.content_hash)
        return [self._register_span(current.source_id, current.content_hash, a, text) for a in python_anchors(text, path, commit)]

    # Window

    @staticmethod
    def _check_role(role: str, pinned: bool) -> None:
        if role not in ROLES:
            raise SessionError(f"unknown role {role!r}")
        if role == "system" and not pinned:
            raise SessionError("system segments must be pinned")

    def append_span(
        self, span_id: str, role: str = "tool", pinned: bool = False, parent: str | None = None, *, tool_use_id: str | None = None
    ) -> Segment:
        """Append a span verbatim. With `parent`, the segment is a snippet: the span must lie
        inside the parent span of the same source version. With `tool_use_id`, it is part of
        the tool_result answering that tool use."""
        span = self.state.spans.get(span_id)
        if span is None:
            raise SessionError(f"unknown span {span_id}")
        if parent is not None:
            outer = self.state.spans.get(parent)
            if (
                outer is None
                or parent == span_id
                or outer.source_hash != span.source_hash
                or not outer.anchor.start <= span.anchor.start <= span.anchor.end <= outer.anchor.end
            ):
                raise SessionError(f"span {span_id} is not a sub-span of {parent}")
        self._check_role(role, pinned)
        if self.aliased:
            self.assign_alias(span_id)
        segment_id = new_id("seg", self.ledger.head(), "span", span_id)
        segment = Segment(segment_id, role, "span", span_id, "verbatim", pinned, span.text_hash, parent, tool_use_id=tool_use_id)
        self._append("segment_appended", {"segment": segment.to_dict()})
        return segment

    def append_raw(
        self, role: str, text: str, pinned: bool = False, *, tool_use_id: str | None = None, is_error: bool = False
    ) -> Segment:
        """Append a raw message (system messages must be pinned). With `tool_use_id` it is (part
        of) the tool_result answering that tool use, an error result if `is_error`."""
        self._check_role(role, pinned)
        h = self.blobs.put(text.encode("utf-8"))
        segment = Segment(new_id("seg", self.ledger.head(), "raw", h), role, "raw", h, "verbatim", pinned, h, None, "text", tool_use_id, is_error)
        return self._append_raw_segment(segment)

    def append_blocks(self, blocks: Sequence[dict]) -> Segment:
        """Append an assistant message made of content blocks (text and tool_use); the blob is
        their canonical JSON."""
        blocks = [dict(b) for b in blocks]
        if not blocks or any(b.get("type") not in ("text", "tool_use", "thinking", "redacted_thinking") for b in blocks):
            raise SessionError("assistant content blocks must be a non-empty list of text, tool_use and thinking blocks")
        h = self.blobs.put(canonical_json(blocks))
        return self._append_raw_segment(Segment(new_id("seg", self.ledger.head(), "raw", h), "assistant", "raw", h, "verbatim", False, h, None, "blocks"))

    def _append_raw_segment(self, segment: Segment) -> Segment:
        self._append("segment_appended", {"segment": segment.to_dict()})
        if self.aliased:
            self.assign_alias(f"raw:{segment.segment_id}")
        return segment

    def first_candidate(self) -> str | None:
        """Segment id of the first window position the backstop may fold (v1 sessions), or None.
        Everything before it is the frozen prefix (cache breakpoint BP2)."""
        if not self.aliased:
            return None
        model = self.payoff_model()
        return next((seg.segment_id for seg, why in zip(self.state.segments, model.why) if why is None), None)

    def payoff_model(self, measured: int | None = None) -> PayoffModel:
        """The backstop-v1 payoff model of the current window (v1 sessions only), its zones on
        the `measured` window if one is given."""
        return PayoffModel(self.state, self.events, self.blobs, self.policy, self.pricing, self.counter, measured)

    def gated_fold(self, victim_ids: Sequence[str], sentences) -> FoldRecord:
        """A model fold through the backstop-v1 gate (the ctx_fold path).

        The victims must be one contiguous run of foldable positions (A.4), the summary must
        free tokens, and outside the mandatory zone the fold must pay off (J > 0). Raises
        FoldRejected (infeasible_run, fold_would_grow_window, payback_not_met) or FoldError
        (cites outside the victims, role labels); nothing is recorded then."""
        if not self.aliased:
            raise SessionError("gated folds need a backstop-v1 session")
        index = {seg.segment_id: i for i, seg in enumerate(self.state.segments)}
        missing = [v for v in victim_ids if v not in index]
        if missing or not victim_ids or len(set(victim_ids)) != len(victim_ids):
            raise FoldRejected("infeasible_run", why="not_in_window", targets=len(victim_ids))
        positions = sorted(index[v] for v in victim_ids)
        i, j = positions[0], positions[-1] + 1
        if positions != list(range(i, j)):
            raise FoldRejected("infeasible_run", why="not_contiguous", position=next(p for p in range(i, j) if p not in positions))
        model = self.payoff_model()
        blocked = model.infeasible(i, j)
        if blocked is not None:
            raise FoldRejected("infeasible_run", position=blocked[0], why=blocked[1])
        sentences = tuple(sentences)
        victims = [seg.segment_id for seg in self.state.segments[i:j]]
        fold_id = fold_id_for("model", i, victims)
        text = render_fold(fold_id, sentences, plan_fold_aliases(self.state.aliases, fold_id, sentences), self.state.aliases)
        delta = fold_delta(self.state, self.state.segments[i:j], text, self.blobs, aliased=True, k=i)
        if delta <= 0:
            raise FoldRejected("fold_would_grow_window", delta=delta)
        run = model.evaluate(i, j, delta)
        if model.zone != "mandatory" and run.J <= 0:
            raise FoldRejected("payback_not_met", J=run.J, rstar=run.rstar, rhat=model.rhat)
        return self.fold(victims, sentences, k=i)

    def fold(self, victim_ids: Sequence[str], sentences, k: int | None = None) -> FoldRecord:
        """Model fold, ungated; `k` defaults to the window index of the first victim. Raises FoldError."""
        if k is None:
            index = {seg.segment_id: i for i, seg in enumerate(self.state.segments)}
            k = min((index[v] for v in victim_ids if v in index), default=0)
        record = make_fold(self.state, k, victim_ids, sentences, "model", None, blobs=self.blobs, aliased=self.aliased)
        self._append("fold", {"fold": record.to_dict()})
        return record

    def maybe_backstop(self) -> FoldRecord | None:
        """Run the backstop-v0 policy; fold and return the record if it selects victims and the
        fold frees tokens (a selection whose summary would grow the window is not applied)."""
        if self.aliased:
            return None
        selection = select_backstop(self.state, self.events, self.blobs, self.policy)
        if selection is None:
            return None
        k, victims = selection
        sentences = extractive_summary(self.state, victims, self.blobs)
        try:
            record = make_fold(self.state, k, victims, sentences, "backstop", self.policy.to_dict(), blobs=self.blobs)
        except FoldError as exc:
            if exc.reason == "fold_would_grow_window":
                return None
            raise
        self._append("fold", {"fold": record.to_dict()})
        return record

    def evict(self, segment_ids: Sequence[str], reason: str) -> None:
        """Remove unpinned segments from the window."""
        ids = list(segment_ids)
        window = {seg.segment_id: seg for seg in self.state.segments}
        if not ids or len(set(ids)) != len(ids) or any(i not in window for i in ids):
            raise SessionError("evict needs distinct segment ids that are in the window")
        if any(window[i].pinned for i in ids):
            raise SessionError("pinned segments cannot be evicted")
        if splits_tool_block(self.state, self.blobs, set(ids)):
            raise SessionError("an evict must take a tool use and all its results, or none of them")
        self._append("evict", {"segment_ids": ids, "reason": reason})

    def update_source(self, source_id: str, new_text: str) -> None:
        """Record new content for a source; spans of the old content become stale. No-op if unchanged."""
        source = self._source(source_id)
        content_hash = self.blobs.put(normalize_newlines(new_text).encode("utf-8"))
        if content_hash != source.content_hash:
            self._append("source_updated", {"source_id": source_id, "content_hash": content_hash})

    # Calls

    def _request(self) -> dict:
        return render_request(self.state, self.blobs)

    def prepare_call(self, measure: Callable[[], int | None] | None = None) -> FoldRecord | None:
        """Run the backstop before a call; return the first fold it made, if any.

        backstop-v1 (BUILD-V2b): while the window is in the discretionary or mandatory zone,
        apply the fold `select_backstop_v1` chooses, each recorded as its own fold event, and
        judge the window again, until it is at or below hard or no fold is chosen (in the
        discretionary zone: none pays off). The folds are kept in `last_folds` and the first
        analysis (the one behind the fold returned) in `last_backstop`. If the window is still
        mandatory and no run is feasible, raises BudgetExceeded; the folds already made stay
        recorded. `measure`, if given,
        returns the window of the request the call would send as count_tokens counts it minus
        the tool definitions (None if it could not count): the zones use it, measured afresh
        before each decision (backstop.py, "Measured window"). `last_window` is the W measure
        of the window the call is sent with."""
        if not self.aliased:
            return self.maybe_backstop()
        self.last_folds = []
        while True:
            measured = measure() if measure is not None else None
            analysis = self.payoff_model(measured).analysis()
            if not self.last_folds:
                self.last_backstop = analysis
            self.last_window = analysis.window
            decision = analysis.decision
            if decision is None:
                if analysis.zone == "mandatory":
                    raise BudgetExceeded(analysis.window, self.budget_tokens, "no_feasible_run")
                return self.last_folds[0] if self.last_folds else None
            sentences = extractive_summary_v1(self.state, decision.victims, self.blobs, rules=self.policy.rules, digest=decision.digest)
            policy = decision.record(self.policy, self.state.pricing_hash)
            record = make_fold(self.state, decision.i, decision.victims, sentences, "backstop", policy, blobs=self.blobs, aliased=True)
            if record.freed_tokens != decision.delta:
                raise SessionError(f"backstop predicted delta {decision.delta}, the fold frees {record.freed_tokens}")
            self._append("fold", {"fold": record.to_dict()})
            self.last_folds.append(record)

    def call(self) -> CallResult:
        """Run the backstop, render the window (`render_request`) and record the call."""
        self.prepare_call()
        return self.record_call(self._request())

    def record_call(self, request: dict) -> CallResult:
        """Check that every segment's blob re-hashes and its (JSON-escaped) verbatim text occurs
        byte-for-byte in the serialized `request`, then record the call with its manifest
        hash, exposures and read map. Runs no backstop. An empty text block of a message is
        not looked for: the request builder never sends one (BUILD-V3FIX); nor, once the ledger
        records thinking bindings, a thinking block: the builder leaves one out when what
        preceded it was rewritten (BUILD-V4FIX)."""
        serialized = canonical_json(request).decode("utf-8")
        unbound_ok = self.state.thinking_bound is not None
        for seg in self.state.segments:
            if seg.content_type == "blocks":  # each field of each block, wherever cache_control lands
                blocks = [b for b in content_blocks(seg, self.blobs) if not empty_text_block(b) and not (unbound_ok and b.get("type") in ("thinking", "redacted_thinking"))]
                fragments = [canonical_json({k: v}).decode("utf-8")[1:-1] for b in blocks for k, v in b.items()]
            else:
                text = self._text(seg.text_hash)
                dropped = seg.role != "system" and seg.tool_use_id is None and empty_text_block({"type": "text", "text": text})
                fragments = [] if dropped else [canonical_json(text).decode("utf-8")[1:-1]]
            if not all(f in serialized for f in fragments):
                raise SessionError(f"segment {seg.segment_id} is missing from the request")
        call_id = new_id("call", self.ledger.head())
        current = read_map(self.state)
        self._append(
            "call",
            {"call_id": call_id, "manifest_hash": manifest_hash(self.state), "exposures": exposures(self.state), "read_map": current},
        )
        return CallResult(call_id, request, current)

    def record_model_call(
        self,
        call_id: str,
        *,
        model: str,
        request: dict,
        build: dict,
        breakpoints: Sequence[dict],
        usage: dict,
        usage_raw: dict,
        stop_reason: str,
        response_id: str,
        response_hash: str,
        cost_pusd: int,
        extra: dict | None = None,
    ) -> Event:
        """Record a `model_call` right after its `call` (P16): the request hash, how it was built
        (so P15 can rebuild it), the usage and its exact cost under the price table in force.
        `extra` adds fields (M3: the predicted prompt and cache-read tokens)."""
        if not self.state.calls or self.state.calls[-1] != call_id:
            raise SessionError(f"model_call {call_id} must follow its own call event")
        if self.state.pricing_hash is None:
            raise SessionError("a model_call needs a price table in force")
        payload = {
            "call_id": call_id,
            "model": model,
            "request_hash": sha256_hex(canonical_json(request)),
            "build": dict(build),
            "breakpoints": [dict(b) for b in breakpoints],
            "usage": dict(usage),
            "usage_raw": usage_raw,
            "stop_reason": stop_reason,
            "response_id": response_id,
            "response_hash": response_hash,
            "pricing_hash": self.state.pricing_hash,
            "cost_pusd": cost_pusd,
            **(extra or {}),
        }
        return self._append("model_call", payload)

    def refetch(self, span_id: str, *, tool_use_id: str | None = None) -> Resolution:
        """Resolve the span's anchor against its source's current text. On OK or MOVED, register
        the (possibly new) span and append it verbatim (inside the tool_result of `tool_use_id`,
        if given). Always records a `refetch` event."""
        span = self.state.spans.get(span_id)
        if span is None:
            raise SessionError(f"unknown span {span_id}")
        source = self._source(span.source_id)
        text = self._text(source.content_hash)
        if isinstance(span.anchor, PyAnchor):
            resolution = resolve_python(span.anchor, text, old_text=self._text(span.text_hash))
        else:
            resolution = resolve_text(span.anchor, text)
        new_span_id = None
        if resolution.status in ("OK", "MOVED"):
            fresh = self._register_span(source.source_id, source.content_hash, resolution.anchor, text)
            self.append_span(fresh.span_id, tool_use_id=tool_use_id)
            new_span_id = fresh.span_id
        self._append("refetch", {"span_id": span_id, "status": resolution.status, "new_span_id": new_span_id})
        return resolution

    # Claims

    def _latest_call(self, call_id: str | None) -> str:
        if call_id is None:
            if not self.state.calls:
                raise SessionError("no call has been made yet")
            return self.state.calls[-1]
        if call_id not in self.state.calls:
            raise SessionError(f"unknown call {call_id}")
        return call_id

    def _resolve(self, citation: Citation) -> tuple[Citation, bool]:
        """(Citation with a canonical ref and the alias the model wrote, whether it resolved).
        An unknown alias keeps the alias as its ref; an unknown canonical ref is kept as is.
        A ref that is a whole header, "@A <anything>" or "[@A ...]", resolves to alias @A when
        @A is a known alias (the header text is dropped); otherwise it is an unknown ref."""
        header = _HEADER_REF.fullmatch(citation.ref.strip()) if not is_alias(citation.ref) and not self._ref_known(citation.ref) else None
        alias = next((g for g in header.groups() if g), None) if header else None
        if alias is not None and self.state.aliases.has_alias(alias):
            citation = Citation(alias, citation.quote)
        if is_alias(citation.ref):
            try:
                return Citation(self.state.aliases.target_of(citation.ref), citation.quote, citation.ref), True
            except UnknownAlias:
                return Citation(citation.ref, citation.quote, citation.ref), False
        if citation.alias is not None and self.state.aliases.alias_of(citation.ref) != citation.alias:
            raise SessionError(f"alias {citation.alias} does not name {citation.ref}")
        return citation, self._ref_known(citation.ref)

    def unknown_refs(self, claim: Claim) -> list[str]:
        """The cited refs of `claim` that name nothing in the ledger (the unknown_ref cites)."""
        return [c.ref for c in claim.citations if c.ref != "inference" and not self._ref_known(c.ref)]

    def _claims(self) -> dict[str, Claim]:
        return {e.payload["claim"]["claim_id"]: Claim.from_dict(e.payload["claim"]) for e in self.events if e.type == "claim"}

    def _superseded(self) -> set[str]:
        return {c.supersedes for c in self._claims().values() if c.supersedes is not None}

    def _check_supersedes(self, supersedes: str | None) -> None:
        if supersedes is None:
            return
        old = self._claims().get(supersedes)
        if old is None or old.status not in ("needs_refetch", "ungrounded") or supersedes in self._superseded():
            raise SessionError(f"claim {supersedes!r} is not an unsuperseded needs_refetch or ungrounded claim")

    def claim(
        self,
        text: str,
        citations: Sequence[Citation],
        call_id: str | None = None,
        *,
        turn_id: str | None = None,
        supersedes: str | None = None,
    ) -> tuple[Claim, Verdict]:
        """Verify a claim against the window at `call_id` (default the latest call) and record it.

        A claim is an observation of what the model said, so it is always recorded; nothing here
        refuses one. Citations may name a ref by alias ("@3"); it is resolved to its canonical
        ref and the alias is kept on the citation. Status:
        - "ungrounded", reason "unknown_ref", if a cite does not resolve;
        - "inference" if every cite is "inference";
        - "grounded" if verification passes and every cited fold sentence was re-fetched after
          its fold (P6);
        - "needs_refetch" if the cited spans are held only as summary_only, dropped or stale, or
          P6 obligations are unmet: the verdict is a RefetchRequired (returned, not raised)
          whose obligations are the span ids to re-fetch;
        - else "ungrounded".
        `supersedes` names an earlier unsuperseded needs_refetch or ungrounded claim."""
        call_id = self._latest_call(call_id)
        self._check_supersedes(supersedes)
        resolved = [self._resolve(c) for c in citations]
        citations = tuple(c for c, _ in resolved)
        claim_id = new_id("claim", self.ledger.head(), call_id, text)
        draft = Claim(claim_id, call_id, text, citations, "ungrounded", None, turn_id, supersedes)
        at_call = state_at_call(self.events, call_id)
        if not all(ok for _, ok in resolved):
            verdict, status = Verdict(0, False, "flag", "unknown_ref"), "ungrounded"
        elif citations and all(c.ref == "inference" for c in citations):
            verdict, status = verify_claim(draft, at_call, self.blobs), "inference"
        else:
            verdict = verify_claim(draft, at_call, self.blobs)
            unmet = unmet_refetch_obligations(self.events, self.state, citations) if verdict.passed else []
            obligations = tuple(sorted(set().union(*(down(self.state, ref) for ref in unmet))))
            if obligations:
                verdict = RefetchRequired(verdict.tier, False, "refetch", "refetch_obligation", obligations, claim_id, obligations)
            elif unmet:  # a cited fold sentence with no span behind it: nothing to re-fetch
                verdict = Verdict(verdict.tier, False, "flag", "refetch_obligation")
            elif not verdict.passed and verdict.tier == 0 and verdict.action == "refetch" and verdict.support:
                states = {kappa(at_call, span_id).value for span_id in verdict.support}
                reason = next(r for r in ("stale", "dropped", "summary_only") if r in states)
                verdict = RefetchRequired(0, False, "refetch", reason, verdict.support, claim_id, verdict.support)
            status = "grounded" if verdict.passed else "needs_refetch" if isinstance(verdict, RefetchRequired) else "ungrounded"
        claim = replace(draft, status=status, verdict=verdict.to_dict())
        self._append("claim", {"claim": claim.to_dict()})
        return claim, verdict

    def abstain(
        self,
        text: str,
        call_id: str | None = None,
        *,
        turn_id: str | None = None,
        supersedes: str | None = None,
        reason: str = "model_abstained",
    ) -> Claim:
        """Record that the model declined to answer (or that the harness abstained for it)."""
        if call_id is not None or self.state.calls:
            call_id = self._latest_call(call_id)
        self._check_supersedes(supersedes)
        verdict = Verdict(0, False, "abstain", reason)
        claim_id = new_id("claim", self.ledger.head(), call_id or "", text)
        claim = Claim(claim_id, call_id, text, (), "abstain", verdict.to_dict(), turn_id, supersedes)
        self._append("claim", {"claim": claim.to_dict()})
        return claim

    def flag(self, text: str, reason: str, call_id: str | None = None, *, turn_id: str | None = None, supersedes: str | None = None) -> Claim:
        """Record an ungrounded claim the harness could not verify at all (e.g. an answer that
        did not parse), with a "flag" verdict naming `reason`."""
        call_id = self._latest_call(call_id)
        self._check_supersedes(supersedes)
        verdict = Verdict(0, False, "flag", reason)
        claim_id = new_id("claim", self.ledger.head(), call_id, text)
        claim = Claim(claim_id, call_id, text, (), "ungrounded", verdict.to_dict(), turn_id, supersedes)
        self._append("claim", {"claim": claim.to_dict()})
        return claim

    def turn_claims(self, turn_id: str) -> list[Claim]:
        """Every claim recorded for `turn_id`, in order."""
        return [c for c in self._claims().values() if c.turn_id == turn_id]

    def unresolved_claims(self, turn_id: str) -> list[Claim]:
        """Claims of `turn_id` with status needs_refetch or ungrounded that nothing supersedes, in order."""
        superseded = self._superseded()
        return [
            c
            for c in self._claims().values()
            if c.turn_id == turn_id and c.status in ("needs_refetch", "ungrounded") and c.claim_id not in superseded
        ]

    @staticmethod
    def _chain_end(claims: dict[str, Claim], claim_id: str) -> Claim:
        """The last claim of the chain superseding `claim_id` (itself if nothing does). Only
        needs_refetch and ungrounded claims can be superseded, so the end is a passing claim
        (grounded or inference: the failure is CLEARED), an abstain claim (it was withdrawn) or
        an open failed claim."""
        superseded_by = {c.supersedes: c.claim_id for c in claims.values() if c.supersedes is not None}
        end = claims[claim_id]
        while end.claim_id in superseded_by:
            end = claims[superseded_by[end.claim_id]]
        return end

    def _failure_ends(self, claims: dict[str, Claim], turn_id: str) -> list[tuple[Claim, Claim]]:
        """(failed claim, the end of its chain) for every needs_refetch or ungrounded claim of the turn."""
        return [(c, self._chain_end(claims, c.claim_id)) for c in claims.values() if c.turn_id == turn_id and c.status in ("needs_refetch", "ungrounded")]

    def cleared_failures(self, turn_id: str) -> list[Claim]:
        """The needs_refetch and ungrounded claims of `turn_id` that a passing claim cleared, in order."""
        return [c for c, end in self._failure_ends(self._claims(), turn_id) if end.status in ("grounded", "inference")]

    def emit_answer(self, turn_id: str, claim_ids: Sequence[str], *, abstained: bool = False, question: str | None = None, answer: object = None) -> Event:
        """Let an answer leave the system: record `answer_emitted {turn_id, claim_ids, abstained}`.

        Raises, recording nothing, RefetchObligationError (with the span ids to re-fetch) if a
        listed claim, or any unsuperseded claim of the turn, is needs_refetch; and
        UngroundedClaimError if a listed claim is ungrounded, or (unless the turn abstains) a
        needs_refetch or ungrounded claim of the turn is not cleared: still open, or withdrawn
        by an abstain claim (`_chain_end`; BUILD-V3: only a superseding claim that itself passes
        clears a failed claim). Listed claims must belong to the turn.

        With `question` (the question text; L since BUILD-V3) a non-abstaining answer must also
        pass the answer gate (verify.gate.answer_gate on `answer` and the listed claims), else
        GateError; the gate's verdict is then recorded in the event as "gate"."""
        claims = self._claims()
        listed = []
        for claim_id in claim_ids:
            claim = claims.get(claim_id)
            if claim is None or claim.turn_id != turn_id:
                raise SessionError(f"claim {claim_id!r} is not a claim of turn {turn_id!r}")
            listed.append(claim)
        pending = {c.claim_id: c for c in [*listed, *self.unresolved_claims(turn_id)]}
        refetch = [c for c in pending.values() if c.status == "needs_refetch"]
        if refetch:
            obligations = sorted({o for c in refetch for o in c.verdict.get("obligations", [])})
            raise RefetchObligationError([c.claim_id for c in refetch], obligations)
        ungrounded = [c for c in listed if c.status == "ungrounded"]
        if not abstained:
            ungrounded += [c for c in pending.values() if c.status == "ungrounded" and c not in ungrounded]
            ungrounded += [c for c, end in self._failure_ends(claims, turn_id) if end.status == "abstain"]
        if ungrounded:
            raise UngroundedClaimError([c.claim_id for c in ungrounded])
        payload: dict = {"turn_id": turn_id, "claim_ids": list(claim_ids), "abstained": abstained}
        if question is not None and not abstained:
            verdict = (self.gate or answer_gate)(question, answer, listed, self.state, self.blobs)
            if not verdict.passed:
                raise GateError([c.claim_id for c in listed], verdict)
            payload["gate"] = verdict.to_dict()
        return self._append("answer_emitted", payload)


class EgressError(SessionError):
    """An answer may not leave the system yet. `claim_ids` are the claims blocking it."""

    def __init__(self, message: str, claim_ids: Sequence[str]) -> None:
        super().__init__(message)
        self.claim_ids = list(claim_ids)


class RefetchObligationError(EgressError):
    """Claims rest on summary-only, dropped or stale spans; `obligations` must be re-fetched."""

    def __init__(self, claim_ids: Sequence[str], obligations: Sequence[str]) -> None:
        super().__init__(f"claims {list(claim_ids)} need a re-fetch of {list(obligations)}", claim_ids)
        self.obligations = list(obligations)


class UngroundedClaimError(EgressError):
    """Claims failed verification and were neither cleared by a passing claim nor covered by an abstention."""

    def __init__(self, claim_ids: Sequence[str]) -> None:
        super().__init__(f"claims {list(claim_ids)} are ungrounded", claim_ids)


class GateError(EgressError):
    """The answer gate (verify.gate) refused a non-abstaining answer: it is not anchored to
    what the question asks, or a claim cites a note updated later. `verdict` says why."""

    def __init__(self, claim_ids: Sequence[str], verdict: GateVerdict) -> None:
        shown = f" ({', '.join(verdict.detail)})" if verdict.detail else ""
        super().__init__(f"the answer gate refused the answer: {verdict.reason}{shown}", claim_ids)
        self.verdict = verdict


__all__ = [
    "BudgetExceeded",
    "CallResult",
    "EgressError",
    "FoldError",
    "FoldRejected",
    "GateError",
    "RefetchObligationError",
    "Session",
    "SessionError",
    "UngroundedClaimError",
    "unmet_refetch_obligations",
]
