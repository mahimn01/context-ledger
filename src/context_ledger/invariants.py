"""Invariant checker: replays a ledger and reports every violated property.

`check` never raises. Each property check is guarded, so malformed data is reported as a
violation of the property being checked; a replay failure is a P2 violation at that seq
and stops the replay-dependent checks (P11 still covers the whole chain).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .alias import alias_for, is_alias
from .backstop import select_backstop_v1
from .fold import (
    BackstopPolicy,
    BackstopPolicyV1,
    cite_refs_of,
    extractive_summary,
    extractive_summary_v1,
    fold_delta,
    has_role_label,
    policy_from_dict,
    render_fold,
    select_backstop,
)
import json

from .ids import canonical_json, sha256_hex, span_id
from .model import Claim, Event, FoldRecord, Segment, Span, TextAnchor
from .pricing import BREAKDOWN_FIELDS, USAGE_FIELDS, PricingError, PricingTable
from .render import rendered_window_tokens
from .store.blobs import BlobStore
from .store.ledger import verify_chain
from .tokens import estimator_for
from .verify import REFETCH_REASONS, verify_claim
from .window import (
    WindowState,
    apply_event,
    down,
    exposures,
    fold_sentence,
    manifest_hash,
    parse_fold_ref,
    parse_whole_fold_ref,
    read_map,
    ref_kind_of,
    window_tokens,
)


@dataclass(frozen=True)
class Violation:
    prop: str
    seq: int | None
    detail: str


class _Checker:
    def __init__(self, events: Sequence[Event], blobs: BlobStore) -> None:
        self.events = list(events)
        self.blobs = blobs
        self.out: list[Violation] = []
        self.state = WindowState()
        self.pinned: dict[str, str] = {}  # pinned segment id -> text_hash, as appended
        self.raw_ids: set[str] = set()
        self.fold_seqs: dict[str, int] = {}
        self.refetches: list[tuple[int, str]] = []
        self.call_states: dict[str, WindowState] = {}
        self.alias_count = 0
        self.alias_target: dict[str, str] = {}  # first introduction of each alias
        self.alias_of: dict[str, str] = {}  # first alias of each target
        self.tokens_before_fold: int | None = None
        self.claims: dict[str, Claim] = {}
        self.superseded: set[str] = set()
        self.prev_request: dict | None = None  # the previous model call's rebuilt request, for P15(c)
        self.awaiting_model_call: str | None = None  # the latest call, until its model_call

    def fail(self, prop: str, seq: int | None, detail: str) -> None:
        self.out.append(Violation(prop, seq, detail))

    def guard(self, prop: str, seq: int | None, fn: Callable[..., None], *args: object) -> None:
        try:
            fn(seq, *args)
        except Exception as exc:
            self.fail(prop, seq, f"check could not run: {exc!r}")

    def text(self, h: str) -> str:
        return self.blobs.get(h).decode("utf-8")

    def run(self) -> list[Violation]:
        self.guard("P11", None, self.p11_chain)
        for position, event in enumerate(self.events):
            seq = event.seq if isinstance(getattr(event, "seq", None), int) else position + 1
            if not self.step(position, event, seq):
                break
        return self.out

    def step(self, position: int, event: Event, seq: int) -> bool:
        kind = getattr(event, "type", None)
        payload = getattr(event, "payload", None)
        if kind == "alias_assigned":
            self.guard("P13", seq, lambda q: self.p13_introduce(q, payload["alias"], payload["target"], None))
        elif kind == "fold":
            self.guard("P13", seq, self.p13_fold, payload)
            self.guard("P3", seq, self.p3_fold_before, payload)
            self.guard("P7", seq, lambda q: self.p7_victims(q, FoldRecord.from_dict(payload["fold"]).victims))
            self.guard("P8", seq, self.p8_sentences, payload)
            self.guard("P9", seq, self.p9_backstop, payload, position)
            self.guard("P10", seq, self.p10_acyclic, payload)
            self.guard("P12", seq, self.p12_before, payload)
        elif kind == "evict":
            self.guard("P7", seq, lambda q: self.p7_victims(q, list(payload["segment_ids"])))
        elif kind == "claim":
            self.guard("P5", seq, self.p5_claim, payload)
            self.guard("P6", seq, self.p6_refetch, payload)
            self.guard("P13", seq, self.p13_claim, payload)
            self.guard("P14", seq, self.p14_claim, payload)
        elif kind == "answer_emitted":
            self.guard("P14", seq, self.p14_answer, payload)
        elif kind == "model_call":
            self.guard("P15", seq, self.p15_model_call, payload)
            self.guard("P16", seq, self.p16_model_call, payload)
        try:
            apply_event(self.state, event)
        except Exception as exc:
            self.fail("P2", seq, f"replay failed: {exc}")
            return False
        if kind in ("source_registered", "source_updated"):
            h = payload["source"]["content_hash"] if kind == "source_registered" else payload["content_hash"]
            self.guard("P1", seq, self.p1_blob, h)
        elif kind == "span_registered":
            self.guard("P1", seq, self.p1_span, Span.from_dict(payload["span"]))
        elif kind == "segment_appended":
            segment = Segment.from_dict(payload["segment"])
            self.guard("P1", seq, self.p1_segment, segment)
            self.guard("P8", seq, self.p8_system_pinned, segment)
            if segment.pinned:
                self.pinned[segment.segment_id] = segment.text_hash
            if segment.ref_kind == "raw":
                self.raw_ids.add(segment.segment_id)
        elif kind == "fold":
            fold = FoldRecord.from_dict(payload["fold"])
            self.fold_seqs[fold.fold_id] = seq
            self.guard("P1", seq, self.p1_blob, fold.summary_hash)
            self.guard("P3", seq, self.p3_summary, fold)
            self.guard("P8", seq, self.p8_fold_role, fold)
            self.guard("P12", seq, self.p12_after, fold)
        elif kind == "refetch" and isinstance(payload, dict) and isinstance(payload.get("span_id"), str):
            self.refetches.append((seq, payload["span_id"]))
        elif kind == "call":
            self.guard("P2", seq, self.p2_call, payload)
            self.guard("P4", seq, self.p4_read_map, payload)
            self.guard("P7", seq, self.p7_pinned_present)
            self.guard("P13", seq, self.p13_call)
            self.call_states[payload["call_id"]] = self.state.copy()
            self.awaiting_model_call = payload["call_id"]
        elif kind == "pricing_set":
            self.guard("P16", seq, self.p16_pricing, payload)
        return True

    # P1 content addressing

    def p1_blob(self, seq: int, h: str) -> None:
        try:
            self.blobs.get(h)
        except Exception as exc:
            self.fail("P1", seq, f"blob {h} unavailable or corrupt: {exc!r}")

    def p1_span(self, seq: int, span: Span) -> None:
        source_text, span_text = self.text(span.source_hash), self.text(span.text_hash)
        anchor = span.anchor
        if source_text[anchor.start : anchor.end] != span_text:
            self.fail("P1", seq, f"span {span.span_id} text differs from its source slice")
        if span.n_chars != len(span_text) or (isinstance(anchor, TextAnchor) and anchor.exact != span_text):
            self.fail("P1", seq, f"span {span.span_id} length or exact text is inconsistent")
        if span.span_id != span_id(span.source_hash, anchor.to_dict()):
            self.fail("P1", seq, f"span id {span.span_id} is not the hash of its source and anchor")

    def p1_segment(self, seq: int, segment: Segment) -> None:
        self.p1_blob(seq, segment.text_hash)
        if segment.ref_kind == "span" and self.state.spans[segment.ref].text_hash != segment.text_hash:
            self.fail("P1", seq, f"segment {segment.segment_id} text_hash differs from its span")

    # P2 conservation, P4 status soundness

    def p2_call(self, seq: int, payload: dict) -> None:
        if payload["manifest_hash"] != manifest_hash(self.state):
            self.fail("P2", seq, f"call {payload['call_id']} manifest hash does not match the replayed window")
        if payload["exposures"] != exposures(self.state):
            self.fail("P2", seq, f"call {payload['call_id']} exposures do not match the replayed window")

    def p4_read_map(self, seq: int, payload: dict) -> None:
        if payload["read_map"] != read_map(self.state):
            self.fail("P4", seq, f"call {payload['call_id']} read_map does not match the replayed knowledge states")

    # P3 fold linkage

    def p3_fold_before(self, seq: int, payload: dict) -> None:
        fold = FoldRecord.from_dict(payload["fold"])
        index = {seg.segment_id: i for i, seg in enumerate(self.state.segments)}
        if not 0 <= fold.k <= len(self.state.segments):
            self.fail("P3", seq, f"fold {fold.fold_id} k={fold.k} outside the window")
        missing = [v for v in fold.victims if v not in index or index[v] < fold.k]
        if missing:
            self.fail("P3", seq, f"fold {fold.fold_id} victims not in the window at index >= k: {missing}")
        allowed = {"inference"}.union(*(cite_refs_of(self.state.segments[index[v]], self.state) for v in fold.victims if v in index))
        outside = sorted({c for s in fold.sentences for c in s.cites} - allowed)
        if outside:
            self.fail("P3", seq, f"fold {fold.fold_id} cites outside its victims: {outside}")

    def p3_summary(self, seq: int, fold: FoldRecord) -> None:
        if self.text(fold.summary_hash) != render_fold(fold.fold_id, fold.sentences, fold.aliases, self.state.aliases):
            self.fail("P3", seq, f"fold {fold.fold_id} summary blob does not match its sentences")

    # P5 provenance completeness, P6 refetch obligation

    def p5_claim(self, seq: int, payload: dict) -> None:
        claim = Claim.from_dict(payload["claim"])
        unresolved = [c.ref for c in claim.citations if not self.resolves(c.ref)]
        if (claim.verdict or {}).get("reason") == "unknown_ref":
            if claim.status != "ungrounded" or not unresolved:
                self.fail("P5", seq, f"claim {claim.claim_id} is labelled unknown_ref but every cite resolves or it is not ungrounded")
        elif unresolved:
            self.fail("P5", seq, f"claim {claim.claim_id} cites {unresolved[0]!r}, which does not resolve")
        if claim.status != "grounded":
            return
        at_call = self.call_states.get(claim.call_id)
        if at_call is None:
            self.fail("P5", seq, f"grounded claim {claim.claim_id} names unknown call {claim.call_id}")
            return
        if not set().union(*(down(at_call, c.ref) for c in claim.citations)):
            self.fail("P5", seq, f"grounded claim {claim.claim_id} reduces to no span")
        elif not verify_claim(claim, at_call, self.blobs).passed:
            self.fail("P5", seq, f"grounded claim {claim.claim_id} does not re-verify at its call")

    def resolves(self, ref: str) -> bool:
        kind = ref_kind_of(ref)
        if kind == "span":
            return ref in self.state.spans
        if kind == "raw":
            return ref[len("raw:") :] in self.raw_ids
        if kind == "fold":
            return fold_sentence(self.state, ref) is not None or parse_whole_fold_ref(ref) in self.state.folds
        return True

    def p6_refetch(self, seq: int, payload: dict) -> None:
        claim = Claim.from_dict(payload["claim"])
        if claim.status != "grounded":
            return
        for citation in claim.citations:
            parsed = parse_fold_ref(citation.ref)
            if parsed is None:
                continue
            fold_seq = self.fold_seqs.get(parsed[0])
            reach = down(self.state, citation.ref)
            if fold_seq is None or not any(s > fold_seq and sid in reach for s, sid in self.refetches):
                self.fail("P6", seq, f"grounded claim {claim.claim_id} cites {citation.ref} without a refetch after the fold")

    # P7 pinning

    def p7_victims(self, seq: int, victims: Sequence[str]) -> None:
        window = {seg.segment_id: seg for seg in self.state.segments}
        pinned = [v for v in victims if v in self.pinned or (v in window and window[v].pinned)]
        if pinned:
            self.fail("P7", seq, f"pinned segments named as victims: {pinned}")

    def p7_pinned_present(self, seq: int) -> None:
        window = {seg.segment_id: seg.text_hash for seg in self.state.segments}
        for segment_id, text_hash in self.pinned.items():
            if window.get(segment_id) != text_hash:
                self.fail("P7", seq, f"pinned segment {segment_id} missing from the call's window or changed")

    # P8 typed roles

    def p8_sentences(self, seq: int, payload: dict) -> None:
        for sentence in FoldRecord.from_dict(payload["fold"]).sentences:
            if has_role_label(sentence.text):
                self.fail("P8", seq, f"fold sentence contains a role label: {sentence.text!r}")

    def p8_fold_role(self, seq: int, fold: FoldRecord) -> None:
        if any(s.segment_id == fold.summary_segment_id and s.role != "user" for s in self.state.segments):
            self.fail("P8", seq, f"fold segment {fold.summary_segment_id} is not a user segment")

    def p8_system_pinned(self, seq: int, segment: Segment) -> None:
        if segment.role == "system" and not segment.pinned:
            self.fail("P8", seq, f"system segment {segment.segment_id} is not pinned")

    # P9 backstop determinism

    def p9_backstop(self, seq: int, payload: dict, position: int) -> None:
        fold = FoldRecord.from_dict(payload["fold"])
        if fold.trigger != "backstop":
            return
        if fold.policy.get("id") == "backstop-v1":
            self.p9_v1(seq, fold, position)
            return
        policy = BackstopPolicy.from_dict(fold.policy)
        selection = select_backstop(self.state, self.events[:position], self.blobs, policy)
        if selection != (fold.k, list(fold.victims)):
            self.fail("P9", seq, f"backstop fold {fold.fold_id} does not match a re-run of the policy: {selection}")
        elif extractive_summary(self.state, list(fold.victims), self.blobs) != list(fold.sentences):
            self.fail("P9", seq, f"backstop fold {fold.fold_id} sentences do not match a re-run of the summary")

    # P12 fold shrinkage

    def window_measure(self, aliased: bool) -> int:
        return rendered_window_tokens(self.state, self.blobs) if aliased else window_tokens(self.state, self.blobs)

    def p12_before(self, seq: int, payload: dict) -> None:
        """freed_tokens > 0 and equal to the recomputed delta (rendered tokens for aliased folds,
        M1's measure otherwise)."""
        self.tokens_before_fold = None
        fold = FoldRecord.from_dict(payload["fold"])
        aliased = bool(fold.aliases)
        window = {seg.segment_id: seg for seg in self.state.segments}
        if fold.freed_tokens <= 0:
            self.fail("P12", seq, f"fold {fold.fold_id} frees {fold.freed_tokens} tokens")
        if all(v in window for v in fold.victims):
            delta = fold_delta(self.state, [window[v] for v in fold.victims], self.text(fold.summary_hash), self.blobs, aliased=aliased, k=fold.k)
            if delta != fold.freed_tokens:
                self.fail("P12", seq, f"fold {fold.fold_id} records freed_tokens={fold.freed_tokens}, recomputed {delta}")
        self.tokens_before_fold = self.window_measure(aliased)

    def p12_after(self, seq: int, fold: FoldRecord) -> None:
        """The window's tokens strictly decrease across the fold."""
        if self.tokens_before_fold is not None and not self.window_measure(bool(fold.aliases)) < self.tokens_before_fold:
            self.fail("P12", seq, f"fold {fold.fold_id} does not shrink the window")

    # P14 egress soundness

    def p14_claim(self, seq: int, payload: dict) -> None:
        """Supersession names an earlier unsuperseded needs_refetch or ungrounded claim, and a
        needs_refetch claim carries its refetch verdict and sorted obligations."""
        claim = Claim.from_dict(payload["claim"])
        if claim.supersedes is not None:
            old = self.claims.get(claim.supersedes)
            if old is None or old.status not in ("needs_refetch", "ungrounded") or claim.supersedes in self.superseded:
                self.fail("P14", seq, f"claim {claim.claim_id} supersedes {claim.supersedes}, which is not an open needs_refetch or ungrounded claim")
            self.superseded.add(claim.supersedes)
        if claim.status == "needs_refetch":
            verdict = claim.verdict or {}
            obligations = verdict.get("obligations")
            if verdict.get("action") != "refetch" or verdict.get("reason") not in REFETCH_REASONS or not obligations or obligations != sorted(set(obligations)):
                self.fail("P14", seq, f"needs_refetch claim {claim.claim_id} lacks a refetch verdict with sorted obligations")
        self.claims[claim.claim_id] = claim

    def p14_answer(self, seq: int, payload: dict) -> None:
        """Every listed claim is grounded, inference or abstain and belongs to the turn; unless
        the turn abstained, no needs_refetch or ungrounded claim of the turn is unsuperseded."""
        turn_id, claim_ids, abstained = payload["turn_id"], payload["claim_ids"], payload["abstained"]
        if not isinstance(turn_id, str) or not isinstance(claim_ids, list) or not isinstance(abstained, bool):
            raise TypeError("answer_emitted needs str turn_id, list claim_ids and bool abstained")
        for claim_id in claim_ids:
            claim = self.claims.get(claim_id)
            if claim is None or claim.status not in ("grounded", "inference", "abstain") or claim.turn_id != turn_id:
                self.fail("P14", seq, f"answer of turn {turn_id} lists claim {claim_id}, which is not a verified claim of that turn")
        if not abstained:
            for claim in self.claims.values():
                if claim.turn_id == turn_id and claim.status in ("needs_refetch", "ungrounded") and claim.claim_id not in self.superseded:
                    self.fail("P14", seq, f"answer of turn {turn_id} leaves {claim.status} claim {claim.claim_id} unsuperseded")

    # P15 request validity and reconstruction

    def p15_model_call(self, seq: int, payload: dict) -> None:
        """(a) rebuilding the request from the replayed window at its call, with the recorded
        build params, gives the recorded request_hash and breakpoints; (b) it is a valid
        Messages request with at most four breakpoints; (c) its prefix through the previous
        call's BP2 is byte-identical to that call's, unless the window before BP2 changed."""
        from .anthropic.request import InvalidWindow, build_request, linear_paths, linearize, strip_cache_control

        self.prev_request, prev = None, self.prev_request
        call_id, build = payload["call_id"], payload["build"]
        at = self.call_states.get(call_id)
        if at is None or at.pricing is None:
            self.fail("P15", seq, f"model_call {call_id} has no recorded call or no price table to rebuild with")
            return
        if build["prev_last"] != self.state.model_last:
            self.fail("P15", seq, f"model_call {call_id} records prev_last {build['prev_last']}, the ledger gives {self.state.model_last}")
        tools = json.loads(self.text(build["tools_hash"]))
        if sorted(t["name"] for t in tools) != build["tool_names"]:
            self.fail("P15", seq, f"model_call {call_id} tool names do not match its tool definitions")
        try:
            request, meta = build_request(
                at, self.blobs, model=payload["model"], max_tokens=build["max_tokens"], tools=tools, ttl=build["ttl"],
                pricing=PricingTable.from_dict(at.pricing), frozen=build["frozen"], prev_last=build["prev_last"],
                thinking=build.get("thinking"),
            )  # fmt: skip
        except InvalidWindow as exc:
            self.fail("P15", seq, f"model_call {call_id}: the window is not a valid request: {exc}")
            return
        if sha256_hex(canonical_json(request)) != payload["request_hash"]:
            self.fail("P15", seq, f"model_call {call_id} request_hash does not match the rebuilt request")
        if len(payload["breakpoints"]) > 4 or [dict(b) for b in meta.breakpoints] != payload["breakpoints"]:
            self.fail("P15", seq, f"model_call {call_id} breakpoints do not match the rebuilt request (at most 4)")
        blocks = [canonical_json(strip_cache_control(b)) for _, b in linearize(request)]
        ids = [seg.segment_id for seg in at.segments]
        bp2 = next((b for b in meta.breakpoints if b["bp"] == "BP2"), None)
        if prev is not None and prev["bp2"] is not None and prev["tools"] == build["tools_hash"] and prev["system"] == request["system"]:
            covered = prev["covered"]
            if ids[: len(covered)] == covered and blocks[: prev["bp2"] + 1] != prev["blocks"][: prev["bp2"] + 1]:
                self.fail("P15", seq, f"model_call {call_id} changed the cached prefix before BP2 without a change to the window there")
        if bp2 is not None:
            index = {path: n for n, path in enumerate(linear_paths(request))}
            last = max(p for p, own in enumerate(meta.paths) if own and min(index[x] for x in own) <= bp2["block"])
            self.prev_request = {"bp2": bp2["block"], "blocks": blocks, "covered": ids[: last + 1], "tools": build["tools_hash"], "system": request["system"]}

    # P16 cost replay

    def p16_pricing(self, seq: int, payload: dict) -> None:
        table = PricingTable.from_dict(payload["pricing"])
        if table.pricing_hash() != payload["pricing_hash"]:
            self.fail("P16", seq, "pricing_set hash does not match its table")

    def p16_model_call(self, seq: int, payload: dict) -> None:
        """A price table is in force and named by pricing_hash, the model is in it, every usage
        field is an int >= 0, cost_pusd is exactly its price, and the model_call follows its own
        call with no other call in between."""
        call_id = payload["call_id"]
        if call_id != self.awaiting_model_call:
            self.fail("P16", seq, f"model_call {call_id} does not directly follow its call event")
        self.awaiting_model_call = None
        usage = payload["usage"]
        keys = (*USAGE_FIELDS, *BREAKDOWN_FIELDS)
        if not isinstance(usage, dict) or set(usage) != set(keys) or any(
            not isinstance(usage[k], int) or isinstance(usage[k], bool) or usage[k] < 0 for k in keys
        ):
            self.fail("P16", seq, f"model_call {call_id} usage is not six ints >= 0")
            return
        if self.state.pricing is None:
            self.fail("P16", seq, f"model_call {call_id} has no pricing_set before it")
            return
        table = PricingTable.from_dict(self.state.pricing)
        if payload["pricing_hash"] != self.state.pricing_hash or table.pricing_hash() != self.state.pricing_hash:
            self.fail("P16", seq, f"model_call {call_id} pricing_hash is not the table in force")
            return
        try:
            cost = table.cost_pusd(payload["model"], usage, payload["build"]["ttl"])
        except PricingError as exc:
            self.fail("P16", seq, f"model_call {call_id} cannot be priced: {exc}")
            return
        if cost != payload["cost_pusd"]:
            self.fail("P16", seq, f"model_call {call_id} records cost {payload['cost_pusd']} pUSD, the table gives {cost}")

    # P13 alias integrity

    def p13_introduce(self, seq: int, alias: str, target: str, fold: FoldRecord | None) -> None:
        """(a) dense counter, (b) alias introduced once, (c) target aliased once, (d) target exists."""
        expected = alias_for(self.alias_count)
        self.alias_count += 1
        if alias != expected:
            self.fail("P13", seq, f"alias {alias} introduced where the counter gives {expected}")
        if alias in self.alias_target:
            self.fail("P13", seq, f"alias {alias} introduced twice")
        if target in self.alias_of:
            self.fail("P13", seq, f"target {target} already has alias {self.alias_of[target]}")
        if not self.target_exists(target, fold):
            self.fail("P13", seq, f"alias {alias} names unknown target {target!r}")
        self.alias_target.setdefault(alias, target)
        self.alias_of.setdefault(target, alias)

    def target_exists(self, target: str, fold: FoldRecord | None) -> bool:
        kind = ref_kind_of(target)
        if kind == "span":
            return target in self.state.spans
        if kind == "raw":
            return target[len("raw:") :] in self.raw_ids
        if kind == "fold":
            if fold is not None and (target == f"fold:{fold.fold_id}" or parse_fold_ref(target) in {(fold.fold_id, i) for i in range(len(fold.sentences))}):
                return True
            return self.resolves(target)
        return False

    def p13_fold(self, seq: int, payload: dict) -> None:
        fold = FoldRecord.from_dict(payload["fold"])
        for alias, target in fold.aliases:
            self.p13_introduce(seq, alias, target, fold)
        if fold.aliases:
            shown = {t for _, t in fold.aliases} | set(self.alias_of)
            needed = [f"fold:{fold.fold_id}"] + [f"fold:{fold.fold_id}#{i}" for i in range(len(fold.sentences))]
            needed += [c for sentence in fold.sentences for c in sentence.cites if c != "inference"]
            missing = sorted(set(needed) - shown)
            if missing:
                self.fail("P13", seq, f"fold {fold.fold_id} text shows refs without an alias: {missing}")

    def p13_call(self, seq: int) -> None:
        """(e) every alias rendered at this call (span headers, fold texts) resolves to its target."""
        for segment in self.state.segments:
            if segment.ref_kind == "span" and self.alias_count:
                alias = self.state.aliases.alias_of(segment.ref)
                if alias is None or self.alias_target.get(alias) != segment.ref:
                    self.fail("P13", seq, f"span segment {segment.segment_id} has no alias resolving to {segment.ref}")
            elif segment.ref_kind == "fold" and segment.ref in self.state.folds:
                for alias, target in self.state.folds[segment.ref].aliases:
                    if self.alias_target.get(alias) != target:
                        self.fail("P13", seq, f"fold {segment.ref} shows {alias}, which does not resolve to {target}")

    def p13_claim(self, seq: int, payload: dict) -> None:
        """(f) every citation alias resolves to that citation's ref at claim time."""
        claim = Claim.from_dict(payload["claim"])
        for citation in claim.citations:
            if citation.alias is None:
                continue
            if citation.alias in self.alias_target:
                ok = self.alias_target[citation.alias] == citation.ref
            else:
                ok = (claim.verdict or {}).get("reason") == "unknown_ref" and citation.ref == citation.alias and is_alias(citation.alias)
            if not ok:
                self.fail("P13", seq, f"claim {claim.claim_id} cites {citation.alias}, which does not name {citation.ref}")

    def p9_v1(self, seq: int, fold: FoldRecord, position: int) -> None:
        """Re-run select_backstop_v1 just before the fold with the recorded params, the price
        table in force and the recorded measured window, if any (BUILD-V2b: a count_tokens
        measurement made outside the ledger, so it is taken as recorded): the decision record,
        k, the victims and the sentences must match. Several backstop folds may precede one
        call (BUILD-V2b); each is re-run on the window just before it."""
        policy = policy_from_dict(fold.policy["params"])
        if not isinstance(policy, BackstopPolicyV1) or self.state.pricing is None:
            self.fail("P9", seq, f"backstop-v1 fold {fold.fold_id} has no v1 params or no price table in force")
            return
        table = PricingTable.from_dict(self.state.pricing)
        if table.pricing_hash() != self.state.pricing_hash or fold.policy.get("pricing_hash") != self.state.pricing_hash:
            self.fail("P9", seq, f"backstop-v1 fold {fold.fold_id} was priced with a table that is not in force")
            return
        counter = estimator_for(self.state)
        if policy.token_counter != counter.name:
            self.fail("P9", seq, f"backstop-v1 fold {fold.fold_id} names token counter {policy.token_counter}, the ledger has {counter.name} in force")
            return
        decision = select_backstop_v1(self.state, self.events[:position], self.blobs, policy, table, counter, fold.policy.get("measured"))
        if decision is None or decision.record(policy, self.state.pricing_hash) != fold.policy:
            self.fail("P9", seq, f"backstop-v1 fold {fold.fold_id} decision record does not match a re-run of the policy")
        elif (decision.i, decision.victims) != (fold.k, fold.victims):
            self.fail("P9", seq, f"backstop-v1 fold {fold.fold_id} victims do not match a re-run of the policy")
        elif extractive_summary_v1(self.state, list(fold.victims), self.blobs, rules=policy.rules, digest=fold.policy.get("digest", False)) != list(fold.sentences):
            self.fail("P9", seq, f"backstop-v1 fold {fold.fold_id} sentences do not match a re-run of the summary")

    # P10 acyclicity, P11 chain integrity

    def p10_acyclic(self, seq: int, payload: dict) -> None:
        fold = FoldRecord.from_dict(payload["fold"])
        for sentence in fold.sentences:
            for cite in sentence.cites:
                if ref_kind_of(cite) != "fold":
                    continue
                parsed = parse_fold_ref(cite)  # a sentence of an earlier fold, or (v2c) the whole of one
                if (parsed[0] if parsed else parse_whole_fold_ref(cite)) not in self.fold_seqs:
                    self.fail("P10", seq, f"fold {fold.fold_id} cites {cite}, which is not an earlier fold")

    def p11_chain(self, seq: int | None) -> None:
        for bad in verify_chain(self.events):
            self.fail("P11", bad, "hash chain does not verify")


def check(events: Sequence[Event], blobs: BlobStore) -> list[Violation]:
    """Return every violated property in `events` against `blobs` (empty if the ledger is sound)."""
    try:
        return _Checker(events, blobs).run()
    except Exception as exc:  # defensive: check() must never raise
        return [Violation("P2", None, f"checker failed: {exc!r}")]
