"""LedgerClient and AsyncLedgerClient: Messages API calls recorded in the ledger.

The wrapper takes a client the caller already built (an `anthropic.Anthropic` /
`AsyncAnthropic`, or any object with `.messages.create(**params)`). It never constructs an
SDK client and never reads credentials or the environment.

Both classes share every line of logic: a turn is a generator that yields each request and
receives each response; the sync class answers it with `client.messages.create(...)`, the
async class with `await client.messages.create(...)`. The ledger and the tool handlers stay
synchronous, so the same scripted responses give byte-identical ledgers.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

from ..backstop import BudgetExceeded
from ..ids import canonical_json
from ..model import Claim
from ..prompts import PROMPT_SHA256, PROMPT_VERSION, AnswerFormatError, parse_answer
from ..render import span_label
from ..session import GateError, RefetchObligationError, Session, UngroundedClaimError
from ..tools import CTX_TOOLS, ToolOutcome, dispatch as ctx_dispatch, refetch_candidates
from ..verify.gate import asked_target
from .cache import CachePredictor
from .request import build_request, thinking_bindings
from .usage import USAGE_KEYS, content_blocks, handling, normalize_usage, plain, response_dict

Dispatch = Callable[..., ToolOutcome]
log = logging.getLogger("context_ledger.client")
FORCED_MESSAGE = "In this condition you must give a best-guess answer, with claims and quotes."  # BUILD-V3, the forced condition


@dataclass(frozen=True)
class StepResult:
    call_id: str
    stop_reason: str | None  # as the API returned it
    handling: str  # how the loop treated it
    content: tuple[dict, ...]
    usage: dict
    cost_pusd: int


@dataclass(frozen=True)
class TurnResult:
    turn_id: str
    answer: Any  # the ParsedAnswer that was emitted, or None if the turn abstained
    claims: tuple[Claim, ...]  # every claim recorded during the turn, in order
    emitted: tuple[str, ...]  # the claim ids of the answer that left the system
    abstained: bool
    reason: str  # "answered", "abstained" or why the harness abstained
    steps: tuple[StepResult, ...]
    usage: dict
    cost_pusd: int
    refetch_routes: int  # times a claim on non-verbatim text was routed back to ctx_refetch
    reminders: int = 0  # refetch-before-abstaining reminders sent (0 or 1)
    refetches: int = 0  # ctx_refetch calls the model made in the turn
    gates: tuple[dict, ...] = ()  # BUILD-V3: every answer-gate verdict of the turn, in order (GateVerdict.to_dict())
    cleared_failures: tuple[Claim, ...] = ()  # BUILD-V3: failed claims a passing superseding claim cleared
    fallback: Any = None  # BUILD-V3, forced: (ParsedAnswer, claims) of the last blocked candidate when the turn ends blocked
    forced_violation: bool = False  # BUILD-V3, forced: the turn abstained with no candidate after the model wrote UNANSWERABLE
    bound_hit: bool = False  # the round bound ended the turn


@dataclass(frozen=True)
class _Pending:
    call_id: str
    request: dict
    build: dict
    breakpoints: tuple[dict, ...]
    window: int | None = None  # the measured W the backstop held the request to (tool_tokens set)


class _LedgerClientCore:
    """Options added in M3, all off by default so M2 ledgers are unchanged: `thinking` (sent
    and recorded in the build params), `predict` (record the builder's predicted cache read
    and the client's counted prompt, `client.last_prompt_count`, beside the API's numbers),
    `prompt` (the (version, sha256) of the system prompt in use), `parse` (the final-answer
    parser), `on_answer` (called with (turn_id, parsed answer) for every parsed attempt) and
    `format_hint` (the sentence a reformat request ends with). Added in BUILD-V2, off by default:
    `remind_refetch`, refetch before abstaining. When the model's answer abstains, or the
    harness is about to abstain as ungrounded, and a round is left, the ledger is checked once
    per turn for spans held only behind summary lines that name an identifier of the question
    (tools.refetch_candidates); if there are any, one reminder listing them is sent and the
    turn goes on. Otherwise, or after the one reminder, the abstention stands. Added in
    BUILD-V2b, off by default: `tool_tokens`, the arm's tool-definition tokens. When given, the
    backstop judges the window on the request about to be sent as `client.messages.count_tokens`
    counts it, minus `tool_tokens` (the definition of W), not on the estimator; with `predict`
    the model_call records that measure as predicted["window"]. If counting fails (a warning is
    logged) that decision falls back to the estimator.

    Added in BUILD-V3, off by default: `gate`, the answer gate (verify.gate). Every
    non-abstaining answer is emitted with the turn's question text, so emit_answer runs the
    gate; a GateError gets a fix message naming the asked target (or the later note) and the
    turn goes on until an answer passes, the model abstains or the round bound is hit (then the
    harness abstains with gate_blocked). And `forced` (the forced-answer condition): when the
    model writes UNANSWERABLE, one message asks for a best guess; a claim that fails Tier 1 gets
    a fix message every time, never the abstention after one; and if the turn ends with no
    answer emitted while a candidate answer is blocked, the harness abstains with forced_blocked
    and the turn result carries that last candidate as `fallback`. With no candidate after the
    model wrote UNANSWERABLE, the result says forced_violation. Messages then never offer to
    abstain."""

    def __init__(
        self,
        session: Session,
        client: Any,
        *,
        model: str,
        max_tokens: int,
        tools: list[dict] | None = None,
        ttl: str | None = None,
        dispatch: Dispatch | None = None,
        thinking: dict | None = None,
        predict: bool = False,
        prompt: tuple[str, str] | None = None,
        parse: Callable[[str], Any] = parse_answer,
        on_answer: Callable[[str, Any], None] | None = None,
        format_hint: str = "End your turn with exactly one <answer>{...}</answer> block.",
        remind_refetch: bool = False,
        tool_tokens: int | None = None,
        gate: bool = False,
        forced: bool = False,
    ) -> None:
        if not session.aliased or session.pricing is None:
            raise ValueError("LedgerClient needs a backstop-v1 session with a price table in force")
        session.pricing.price(model)  # the model must be priced
        self.session, self.client, self.model, self.max_tokens = session, client, model, max_tokens
        self.tools = sorted((dict(t) for t in (CTX_TOOLS if tools is None else tools)), key=lambda t: t["name"])
        self.ttl = ttl or session.policy.ttl
        self.dispatch = dispatch or ctx_dispatch
        self.tools_hash = session.blobs.put(canonical_json(self.tools))
        self.thinking, self.predict, self.parse, self.on_answer, self.format_hint = thinking, predict, parse, on_answer, format_hint
        self.prompt = prompt or (PROMPT_VERSION, PROMPT_SHA256)
        self.remind_refetch, self.tool_tokens = remind_refetch, tool_tokens
        self.gate, self.forced = gate, forced
        self._cache = CachePredictor(session.counter, session.pricing.price(model).min_cache_tokens)

    # One model call, split around the network round trip

    def _build(self) -> tuple[dict, dict, Any]:
        """(request, build params, BuildMeta) of the window as it stands."""
        s = self.session
        build = {
            "max_tokens": self.max_tokens,
            "ttl": self.ttl,
            "tool_names": [t["name"] for t in self.tools],
            "tools_hash": self.tools_hash,
            "frozen": s.first_candidate(),
            "prev_last": s.state.model_last,
            "prompt": {"version": self.prompt[0], "sha256": self.prompt[1]},
        }
        if self.thinking is not None:
            build["thinking"] = self.thinking
        request, meta = build_request(
            s.state, s.blobs, model=self.model, max_tokens=self.max_tokens, tools=self.tools, ttl=self.ttl,
            pricing=s.pricing, frozen=build["frozen"], prev_last=build["prev_last"], thinking=self.thinking,
        )  # fmt: skip
        return request, build, meta

    def _measure(self) -> int | None:
        """The W measure of the request the window renders now: count_tokens on the same body
        the guard counts (the request without max_tokens) minus the tool definitions; None,
        with a warning, if counting fails."""
        request, _, _ = self._build()
        try:
            result = self.client.messages.count_tokens(**{k: v for k, v in request.items() if k != "max_tokens"})
            value = result["input_tokens"] if isinstance(result, dict) else result.input_tokens
        except Exception as exc:  # the decision falls back to the estimator, as the guard's bound does
            log.warning("count_tokens failed, the backstop uses the estimator for this decision: %s", type(exc).__name__)
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            log.warning("count_tokens returned %r, the backstop uses the estimator for this decision", value)
            return None
        return value - self.tool_tokens

    def _begin(self) -> _Pending:
        """Backstop (may raise BudgetExceeded), build the request, record the `call` event."""
        s = self.session
        if self.tool_tokens is None:
            s.prepare_call()
        else:
            s.prepare_call(self._measure)
        request, build, meta = self._build()
        call = s.record_call(request)
        return _Pending(call.call_id, request, build, meta.breakpoints, s.last_window if self.tool_tokens is not None else None)

    def _predicted(self, pending: _Pending) -> dict:
        """The predicted cache read (anthropic.cache: the longest byte-identical prefix an earlier
        call of this conversation cached, within the TTL, capped at the prompt), the client's
        counted prompt, if it counts, and the measured window the backstop held the request to,
        if it measured."""
        count = getattr(self.client, "last_prompt_count", None)
        count = count if isinstance(count, int) else None
        predicted = {"cache_read": self._cache.predict(pending.request, count), "prompt": count}
        return {"predicted": predicted if pending.window is None else {**predicted, "window": pending.window}}

    def _finish(self, pending: _Pending, response: Any) -> StepResult:
        """Record the assistant message and the `model_call` event with usage and exact cost."""
        s = self.session
        raw = response_dict(response)
        content = content_blocks(raw.get("content") or [])
        usage = normalize_usage(raw.get("usage"), self.ttl)
        cost = s.pricing.cost_pusd(self.model, usage, self.ttl)
        response_hash = s.blobs.put(canonical_json(content))
        if len(content) == 1 and content[0]["type"] == "text":
            s.append_raw("assistant", content[0]["text"])
        elif content:
            s.append_blocks(content)
        stop_reason = raw.get("stop_reason")
        extra = self._predicted(pending) if self.predict else None
        bound = thinking_bindings(pending.request, content)  # BUILD-V4FIX: {} unless the response has a thinking block
        if bound:
            extra = {**(extra or {}), "thinking_bound": bound}
        s.record_model_call(
            pending.call_id, model=self.model, request=pending.request, build=pending.build, breakpoints=pending.breakpoints,
            usage=usage, usage_raw=plain(raw.get("usage") or {}), stop_reason=stop_reason, response_id=str(raw.get("id", "")),
            response_hash=response_hash, cost_pusd=cost, extra=extra,
        )  # fmt: skip
        return StepResult(pending.call_id, stop_reason, handling(stop_reason), tuple(content), usage, cost)

    def _answer_cut_tool_uses(self, step: StepResult) -> None:
        """BUILD-V4FIX (harness patch v4.1-cut-tool-use): a response that ends the turn (max_tokens
        or refusal) while it holds tool_use blocks gets an error tool_result for each, so the next
        request still answers every tool use. In the v4 grid two Haiku responses cut at max_tokens
        held 12 ctx_refetch calls each; nothing answered them, every later request of the episode
        failed InvalidWindow ("a tool_use is not answered by the next message") and the run
        stopped. A turn that ends any other way is unchanged."""
        for block in step.content:
            if block["type"] == "tool_use":
                self.session.append_raw("tool", f"Not run: your response stopped ({step.handling}) before this tool call could run.", tool_use_id=block["id"], is_error=True)

    # The tool-use loop of one turn, as a generator of requests

    def _turn(self, user_text: str, turn_id: str, max_rounds: int) -> Generator[dict, Any, TurnResult]:
        s = self.session
        start = len(s.events)
        s.append_raw("user", user_text)
        steps: list[StepResult] = []
        asked_reformat = asked_fix = checked = asked_forced = said_unanswerable = bound_hit = False
        routes = reminders = 0
        outcome: tuple[Any, list[str], bool, str] | None = None
        stop: str | None = None  # why the harness abstains, when it does before the round bound
        candidate: tuple[Any, list[Claim]] | None = None  # the latest answer not emitted, and its claims
        blocked_by: str | None = None  # what blocked it: "refetch", "ungrounded" or "gate"
        gates: list[dict] = []

        def reminder() -> str | None:
            """The refetch-before-abstaining reminder, checked once per turn while a round is left."""
            nonlocal checked
            if not self.remind_refetch or checked or round_no >= max_rounds - 1:
                return None
            checked = True
            return self._reminder(user_text)

        for round_no in range(max_rounds):
            try:
                pending = self._begin()
            except BudgetExceeded:
                stop = "budget_exceeded"
                break
            step = self._finish(pending, (yield pending.request))
            steps.append(step)
            if step.handling == "tool_use":
                for block in step.content:
                    if block["type"] == "tool_use":
                        result = self.dispatch(s, block["name"], block["input"], turn_id=turn_id, tool_use_id=block["id"])
                        s.append_raw("tool", result.text, tool_use_id=block["id"], is_error=result.is_error)
                continue
            if step.handling == "pause_turn":
                continue
            if step.handling in ("max_tokens", "refusal"):
                self._answer_cut_tool_uses(step)
                stop = step.handling
                break
            text = "\n".join(b["text"] for b in step.content if b["type"] == "text")
            try:
                parsed = self.parse(text)
            except AnswerFormatError as exc:
                s.flag(text or "(empty answer)", "unparseable_answer", turn_id=turn_id)
                if asked_reformat:
                    stop = "unparseable_answer"
                    break
                asked_reformat = True
                s.append_raw("user", f"Verification: your answer could not be parsed ({exc}). {self.format_hint}")
                continue
            if self.on_answer is not None:
                self.on_answer(turn_id, parsed)
            said_unanswerable = parsed.abstain
            if parsed.abstain:
                if self.forced and not asked_forced:
                    asked_forced = True
                    s.abstain("abstained before the forced-answer message", turn_id=turn_id, reason="model_abstained")
                    s.append_raw("user", FORCED_MESSAGE)
                    continue
                message = reminder()
                if message is None:
                    stop = "model_abstained"
                    break
                s.abstain("abstained before a refetch reminder", turn_id=turn_id, reason="model_abstained")
                s.append_raw("user", message)
                reminders += 1
                continue
            # An answer with no claims has nothing verified behind it: it is ungrounded (no_claims).
            claims = self._record_answer(parsed, turn_id) if parsed.claims else [s.flag(text or "(empty answer)", "no_claims", turn_id=turn_id)]
            candidate = (parsed, claims)
            try:
                event = s.emit_answer(turn_id, [c.claim_id for c in claims], **({"question": user_text, "answer": parsed.answer} if self.gate else {}))
                gates += [event.payload["gate"]] if "gate" in event.payload else []
                outcome = (parsed, [c.claim_id for c in claims], False, "answered")
                break
            except RefetchObligationError as exc:
                blocked_by = "refetch"
                routes += 1
                s.append_raw("user", self._refetch_message(turn_id, claims, exc))
            except GateError as exc:
                blocked_by = "gate"
                gates.append(exc.verdict.to_dict())
                s.append_raw("user", self._gate_message(user_text, exc.verdict))
            except UngroundedClaimError as exc:
                blocked_by = "ungrounded"
                if asked_fix and not self.forced:
                    message = reminder()
                    if message is None:
                        stop = "ungrounded"
                        break
                    s.append_raw("user", message)
                    reminders += 1
                    continue
                asked_fix = True
                s.append_raw("user", self._fix_message(turn_id, claims, exc))
        fallback = None
        if outcome is None:
            if stop is None:
                bound_hit = True
                stop = "gate_blocked" if blocked_by == "gate" else "rounds_exhausted"
            if self.forced and candidate is not None:
                fallback, stop = candidate, "forced_blocked"
            outcome = self._abstain(turn_id, stop)
        answer, emitted, abstained, reason = outcome
        turn_claims = [Claim.from_dict(e.payload["claim"]) for e in s.events[start:] if e.type == "claim"]
        total = {key: sum(step.usage[key] for step in steps) for key in USAGE_KEYS}
        return TurnResult(
            turn_id, answer, tuple(turn_claims), tuple(emitted), abstained, reason, tuple(steps), total,
            sum(step.cost_pusd for step in steps), routes, reminders,
            sum(b["type"] == "tool_use" and b["name"] == "ctx_refetch" for step in steps for b in step.content),
            tuple(gates), tuple(s.cleared_failures(turn_id)), fallback,
            self.forced and abstained and candidate is None and said_unanswerable, bound_hit,
        )  # fmt: skip

    def _record_answer(self, parsed, turn_id: str) -> list[Claim]:
        """Record each claim; the k-th supersedes the k-th open claim of the turn. Open claims
        left over stay open (BUILD-V3: only a superseding claim that passes clears a failed
        claim, so they block until a later answer clears them or the turn abstains)."""
        s = self.session
        open_claims = s.unresolved_claims(turn_id)
        claims = []
        for k, item in enumerate(parsed.claims):
            supersedes = open_claims[k].claim_id if k < len(open_claims) else None
            claims.append(s.claim(item.text, item.citations, turn_id=turn_id, supersedes=supersedes)[0])
        return claims

    def _abstain(self, turn_id: str, reason: str) -> tuple[None, list[str], bool, str]:
        """Abstain over every open claim of the turn (or once if none) and emit that answer."""
        s = self.session
        cover = [s.abstain(f"abstained: {reason}", turn_id=turn_id, supersedes=c.claim_id, reason=reason) for c in s.unresolved_claims(turn_id)]
        cover = cover or [s.abstain(f"abstained: {reason}", turn_id=turn_id, reason=reason)]
        ids = [c.claim_id for c in cover]
        s.emit_answer(turn_id, ids, abstained=True)
        return None, ids, True, reason

    def _reminder(self, question: str) -> str | None:
        """The reminder naming the folded spans that may answer `question`, or None if none."""
        s = self.session
        spans = refetch_candidates(s, question)
        if not spans:
            return None
        shown = ", ".join(f"{s.assign_alias(span_id)} ({span_label(s.state, span_id)})" for span_id in spans)
        tail = "." if self.forced else ", or abstain again if they do not answer it."
        return f"Before abstaining: you no longer hold these verbatim, and they may answer the question: {shown}. Call ctx_refetch on them and answer again{tail}"

    def _refetch_message(self, turn_id: str, claims: list[Claim], exc: RefetchObligationError) -> str:
        """One line per claim of this answer that rests on text no longer held verbatim. When
        none does, the obligation is an earlier claim's: this answer superseded other open
        claims, so an earlier needs_refetch claim is still open. BUILD-V3FIX: that case gets one
        line per such claim, naming it and the refs it relies on; it used to get an empty
        message, which the API refused (an empty text block) until the run stopped."""
        alias = self.session.assign_alias
        refs = lambda c: " ".join(alias(o) for o in c.verdict.get("obligations", []))  # noqa: E731
        lines = [
            f"Verification: claim {n} relies on {refs(c)} which you no longer hold verbatim "
            f"({c.verdict['reason']}). Call ctx_refetch on them and answer again."
            for n, c in enumerate(claims, start=1)
            if c.claim_id in exc.claim_ids
        ] or [f'Verification: an earlier claim of this turn still fails ({c.verdict["reason"]}): "{c.text}". It relies on {refs(c)}; call '
              f"ctx_refetch on them if you no longer hold them verbatim. Only a claim that passes clears it: answer again with one passing "
              f"claim in its place{self._or_abstain()}." for c in self.session.turn_claims(turn_id) if c.claim_id in exc.claim_ids]  # fmt: skip
        return "\n".join(lines)

    def _gate_message(self, question: str, verdict) -> str:
        """Why the answer gate refused the answer, in plain words, and what to do: a stale note
        names the later note to use (ctx_refetch brings it back if it is folded); an answer not
        anchored names the asked target and says the cited text is about something else."""
        s = self.session
        alt = self._or_abstain()
        shown = lambda ref: f"{s.assign_alias(ref)} ({span_label(s.state, ref)})" if ref in s.state.spans else f"`{ref}`"  # noqa: E731
        if verdict.reason == "stale_superseded":
            return "\n".join(
                f"Verification: your claim cites {shown(old)}, but a later note {shown(new)} updates it, so {shown(old)} is out of date. "
                f"Answer from the latest note (call ctx_refetch on {s.assign_alias(new) if new in s.state.spans else new} if you no longer hold it "
                f"verbatim) and answer again{alt}."
                for old, new in verdict.stale
            )  # fmt: skip
        target = asked_target(question)  # not None: the gate passes a question with no target
        asked = f"the {target.attribute} of service `{target.name}`" if target.kind == "session" else f"`{target.name}`"
        look = "(ctx_search finds spans you have seen, ctx_refetch brings folded ones back, corpus_search and corpus_read find the rest)"
        if verdict.reason == "target_not_witnessed" and verdict.detail and verdict.detail[0] != target.name:
            missing = _and([f"`{d}`" for d in verdict.detail])
            return (f"Verification: the question asks about {missing} of {asked}, but no verbatim quote you cite about {asked} names {missing}. "
                    f"Quote the line of that text that does, and answer again{alt}.")  # fmt: skip
        if verdict.reason == "target_not_witnessed":
            return (f"Verification: the question asks about {asked}, but none of your cited verbatim quotes is about it: the text you cite "
                    f"is about something else. Find the text about {asked} {look}, cite it with a verbatim quote, and answer again{alt}.")  # fmt: skip
        values = _and([f"`{d}`" for d in verdict.detail]) or "your answer"
        return (f"Verification: your answer {values} is not in a verbatim quote of the text about {asked} that you cite. Quote the line "
                f"of that text that gives the value {look}, and answer again{alt}.")  # fmt: skip

    def _or_abstain(self) -> str:
        """The offer to abstain that ends a fix line: none in the forced condition."""
        return "" if self.forced else "; or abstain"

    def _fix_message(self, turn_id: str, claims: list[Claim], exc: UngroundedClaimError) -> str:
        """One plain-words line per failed claim of this answer: why it was not accepted (with
        the offending tokens of value_not_covered / identifier_not_anchored, the unknown ref
        and an alias to cite instead, a quote that is not verbatim, no claims at all) and what
        to do; then one line per earlier claim of the turn that still fails (BUILD-V3: it stays
        open until a passing claim replaces it)."""
        lines = [self._fix_line(n, c) for n, c in enumerate(claims, start=1) if c.claim_id in exc.claim_ids]
        mine = {c.claim_id for c in claims}
        earlier = [c for c in self.session.turn_claims(turn_id) if c.claim_id in exc.claim_ids and c.claim_id not in mine]
        lines += [f'Verification: an earlier claim of this turn still fails ({(c.verdict or {}).get("reason") or c.status}): "{c.text}". '
                  f"Only a claim that passes clears it: answer again with one passing claim in its place{self._or_abstain()}." for c in earlier]  # fmt: skip
        return "\n".join(lines) or f"Verification: an earlier claim of this turn failed. Answer again{'' if self.forced else ' or abstain'}."

    def _fix_line(self, n: int, claim: Claim) -> str:
        verdict = claim.verdict or {}
        reason, detail = verdict.get("reason"), verdict.get("detail") or []
        head = f"Verification: claim {n} was not accepted:"
        if reason == "no_claims":
            return ("Verification: your answer has no claims, so nothing in it is verified. Give at least one claim that cites the "
                    f"text it rests on by its @alias with a quote copied verbatim from it, and answer again{self._or_abstain()}.")  # fmt: skip
        if reason in ("value_not_covered", "identifier_not_anchored") and detail:
            verb, them = ("does", "it") if len(detail) == 1 else ("do", "them")
            where = "in your quotes" if reason == "value_not_covered" else "in your quotes, and no cited span is named so"
            return (f"{head} {_and([f'`{t}`' for t in detail])} {verb} not appear {where}. Quote a verbatim line that contains "
                    f"{them}, or remove {them} from the claim text, and answer again{self._or_abstain()}.")  # fmt: skip
        if reason == "unknown_ref":
            unknown = self.session.unknown_refs(claim)
            refs = _and([f"`{r}`" for r in unknown]) or "a ref"
            example = self._example_alias(unknown[0]) if unknown else None
            hint = f"e.g. {example}" if example else "e.g. [@3 Lib/x.py#f] gives @3"
            return f"{head} ref {refs} is not a ref shown to you; cite the @alias from the header, {hint}. Answer again{self._or_abstain()}."
        if reason == "no_valid_quote":
            return (f"{head} the quote is not verbatim in the cited text (or shorter than 20 characters). Copy the quote exactly "
                    f"from the cited text and answer again{self._or_abstain()}.")  # fmt: skip
        why = {"no_citations": "it cites nothing", "no_span_support": "it cites no verbatim text"}.get(reason, f"it failed ({reason})")
        return f"{head} {why}. Cite the verbatim text it rests on by its @alias with an exact quote, and answer again{self._or_abstain()}."

    def _example_alias(self, ref: str) -> str | None:
        """The alias of a verbatim span the unknown `ref` most likely meant: one whose header
        label it contains, else one of the file it names covering its line range, else one of
        that file; None if no span is held verbatim."""
        state = self.session.state
        held = [seg.ref for seg in reversed(state.segments) if seg.ref_kind == "span" and state.aliases.alias_of(seg.ref)]
        lines = re.fullmatch(r"(?P<path>.+?):(?P<a>\d+)-(?P<b>\d+)", ref.strip())

        def overlaps(span_id: str) -> bool:
            span = state.spans[span_id]
            text = self.session.blobs.get(span.source_hash).decode("utf-8")
            first, last = text.count("\n", 0, span.anchor.start) + 1, text.count("\n", 0, max(span.anchor.start, span.anchor.end - 1)) + 1
            return first <= int(lines["b"]) and int(lines["a"]) <= last

        uri = lambda span_id: state.sources[state.spans[span_id].source_id].uri  # noqa: E731
        for test in (
            lambda s: span_label(state, s) in ref,
            lambda s: lines is not None and uri(s) == lines["path"] and overlaps(s),
            lambda s: uri(s) in ref,
        ):
            found = next((s for s in held if test(s)), None)
            if found is not None:
                return state.aliases.alias_of(found)
        return None


def _and(items: list[str]) -> str:
    """"a", "a and b", "a, b and c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


class LedgerClient(_LedgerClientCore):
    """Synchronous wrapper: `client.messages.create(**params)` returns the response."""

    def step(self) -> StepResult:
        """One model call: backstop, build, record the call, create, record the response."""
        pending = self._begin()
        return self._finish(pending, self.client.messages.create(**pending.request))

    def run_turn(self, user_text: str, *, turn_id: str, max_rounds: int = 8) -> TurnResult:
        """Append the user's question and loop over model calls until an answer is emitted:
        tool uses are dispatched to the ctx handlers; a claim on non-verbatim text is routed
        to ctx_refetch; max_tokens, refusal, an exceeded budget or exhausted rounds abstain."""
        turn = self._turn(user_text, turn_id, max_rounds)
        try:
            request = next(turn)
            while True:
                request = turn.send(self.client.messages.create(**request))
        except StopIteration as done:
            return done.value


class AsyncLedgerClient(_LedgerClientCore):
    """Asynchronous wrapper: `await client.messages.create(**params)`; otherwise identical."""

    async def step(self) -> StepResult:
        pending = self._begin()
        return self._finish(pending, await self.client.messages.create(**pending.request))

    async def run_turn(self, user_text: str, *, turn_id: str, max_rounds: int = 8) -> TurnResult:
        turn = self._turn(user_text, turn_id, max_rounds)
        try:
            request = next(turn)
            while True:
                request = turn.send(await self.client.messages.create(**request))
        except StopIteration as done:
            return done.value


__all__ = ["AsyncLedgerClient", "LedgerClient", "StepResult", "TurnResult"]
