"""LedgerClientV4: L4's turn (mechanism "l-v4", BUILD-V4 items 3-6). The v3 LedgerClient and its
turn are unchanged; this class adds a turn of its own with three differences.

1. The question first. The question text is the turn's first user segment, so the backstop's
   current-question rule protects it (v3 diagnostics: L appended a question's note before the
   question, the note became the protected "current question", and the question itself was
   folded away in a third to two fifths of the turns that carried a note). Then `preamble()`
   appends the turn's evidence as user segments: the note, and the harness's pre-resolution
   (the asked target's exact span and its address, or the address-miss certificate).
2. Repair. When the answer gate refuses an answer (target_not_witnessed, value_not_with_target,
   stale_superseded, ...), `repair(verdict)` puts the anchored span of the asked target, or the
   latest note of its chain, into the window and returns the message that says so with the
   refusal reason; the turn goes on. At most MAX_REPAIRS repairs per question, each counted
   with its reason; later refusals, and one after the last round, get v3's gate message. Every other egress failure is handled
   as in v3 (refetch routes, fix messages, the refetch-before-abstaining reminder).
3. Forced. As in v3 an UNANSWERABLE gets one message asking for a best guess. Before the last
   round, if no answer has passed yet and a candidate is blocked, the harness says it is the last
   round, and, unless the previous round's refusal was already repaired and the model has not
   answered it yet, repairs once more if a repair is left ("final"; "final_overflow" if the
   backstop then cannot make room for the call): the fallback goes through repair and the gate
   like any answer. If the turn still ends blocked, the result carries the last candidate as
   `fallback` with reason unverified_guess (the ledger records an abstention: nothing unverified
   leaves the egress); with no candidate, reason forced_noanswer, whatever ended the turn.
   `underlying` keeps v3's reason (rounds_exhausted, model_abstained, budget_exceeded, ...), so
   L4's rows stay comparable with L's.

With `gate=False` (L4-nogate, BUILD-V4B decision 3) emit_answer is not given the question, so the
answer gate never runs and no GateError, hence no gate-triggered repair, can happen; Tier 0/1 and
everything else above are the same (a forced last-round repair after a Tier 0/1 block included).
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

from ..backstop import BudgetExceeded
from ..model import Claim
from ..prompts import AnswerFormatError
from ..session import GateError, RefetchObligationError, UngroundedClaimError
from .client import FORCED_MESSAGE, LedgerClient, TurnResult
from .usage import USAGE_KEYS

MAX_REPAIRS = 2
LAST_ROUND = "This is your last round: answer now from the text shown, with claims and verbatim quotes."


@dataclass(frozen=True)
class TurnResultV4(TurnResult):
    repairs: int = 0
    repair_reasons: tuple[str, ...] = ()
    unverified_guess: bool = False
    underlying: str = "answered"  # the reason in v3's terms, before the forced relabelling


class LedgerClientV4(LedgerClient):
    """L4's client (module docstring). `run_turn_v4` is its only new entry point."""

    def run_turn_v4(self, question: str, *, turn_id: str, max_rounds: int, preamble: Callable[[], None], repair: Callable[[Any], str]) -> TurnResultV4:
        turn = self._turn_v4(question, turn_id, max_rounds, preamble, repair)
        try:
            request = next(turn)
            while True:
                request = turn.send(self.client.messages.create(**request))
        except StopIteration as done:
            return done.value

    def _turn_v4(self, question: str, turn_id: str, max_rounds: int, preamble: Callable[[], None], repair: Callable[[Any], str]) -> Generator[dict, Any, TurnResultV4]:
        s = self.session
        start = len(s.events)
        s.append_raw("user", question)
        preamble()
        steps = []
        asked_reformat = asked_fix = checked = asked_forced = said_unanswerable = bound_hit = False
        routes = reminders = 0
        repairs: list[str] = []
        outcome: tuple[Any, list[str], bool, str] | None = None
        stop: str | None = None
        candidate: tuple[Any, list[Claim]] | None = None
        blocked_by: str | None = None
        last_verdict = None
        just_repaired = False  # a repair was sent and the model has not answered it yet
        final_round: int | None = None
        gates: list[dict] = []

        def reminder() -> str | None:
            nonlocal checked
            if not self.remind_refetch or checked or round_no >= max_rounds - 1:
                return None
            checked = True
            return self._reminder(question)

        for round_no in range(max_rounds):
            if self.forced and round_no == max_rounds - 1 and round_no > 0 and candidate is not None:
                if just_repaired:
                    s.append_raw("user", LAST_ROUND)
                elif len(repairs) < MAX_REPAIRS:
                    repairs.append("final")
                    final_round = round_no
                    s.append_raw("user", f"{repair(last_verdict)} {LAST_ROUND}")
            try:
                pending = self._begin()
            except BudgetExceeded:
                stop = "budget_exceeded"
                if final_round == round_no:
                    repairs[-1] = "final_overflow"
                break
            step = self._finish(pending, (yield pending.request))
            steps.append(step)
            just_repaired = False
            if step.handling == "tool_use":
                for block in step.content:
                    if block["type"] == "tool_use":
                        result = self.dispatch(s, block["name"], block["input"], turn_id=turn_id, tool_use_id=block["id"])
                        s.append_raw("tool", result.text, tool_use_id=block["id"], is_error=result.is_error)
                continue
            if step.handling == "pause_turn":
                continue
            if step.handling in ("max_tokens", "refusal"):
                self._answer_cut_tool_uses(step)  # BUILD-V4FIX, as in v3's turn
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
            claims = self._record_answer(parsed, turn_id) if parsed.claims else [s.flag(text or "(empty answer)", "no_claims", turn_id=turn_id)]
            candidate = (parsed, claims)
            try:
                event = s.emit_answer(turn_id, [c.claim_id for c in claims], **({"question": question, "answer": parsed.answer} if self.gate else {}))
                gates += [event.payload["gate"]] if "gate" in event.payload else []
                outcome = (parsed, [c.claim_id for c in claims], False, "answered")
                break
            except RefetchObligationError as exc:
                blocked_by, last_verdict = "refetch", None
                routes += 1
                s.append_raw("user", self._refetch_message(turn_id, claims, exc))
            except GateError as exc:
                blocked_by, last_verdict = "gate", exc.verdict
                gates.append(exc.verdict.to_dict())
                if len(repairs) < MAX_REPAIRS and round_no < max_rounds - 1:  # a repair needs a round to be answered in
                    repairs.append(exc.verdict.reason)
                    s.append_raw("user", repair(exc.verdict))
                    just_repaired = True
                else:
                    s.append_raw("user", self._gate_message(question, exc.verdict))
            except UngroundedClaimError as exc:
                blocked_by, last_verdict = "ungrounded", None
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
        underlying = "answered"
        if outcome is None:
            if stop is None:
                bound_hit = True
                stop = "gate_blocked" if blocked_by == "gate" else "rounds_exhausted"
            underlying = stop
            if self.forced and candidate is not None:
                fallback, stop = candidate, "unverified_guess"
            elif self.forced:
                stop = "forced_noanswer"
            outcome = self._abstain(turn_id, stop)
        answer, emitted, abstained, reason = outcome
        turn_claims = [Claim.from_dict(e.payload["claim"]) for e in s.events[start:] if e.type == "claim"]
        total = {key: sum(step.usage[key] for step in steps) for key in USAGE_KEYS}
        return TurnResultV4(
            turn_id, answer, tuple(turn_claims), tuple(emitted), abstained, reason, tuple(steps), total,
            sum(step.cost_pusd for step in steps), routes, reminders,
            sum(b["type"] == "tool_use" and b["name"] == "ctx_refetch" for step in steps for b in step.content),
            tuple(gates), tuple(s.cleared_failures(turn_id)), fallback,
            self.forced and abstained and candidate is None and said_unanswerable, bound_hit,
            repairs=len(repairs), repair_reasons=tuple(repairs), unverified_guess=fallback is not None, underlying=underlying,
        )  # fmt: skip


__all__ = ["LAST_ROUND", "MAX_REPAIRS", "LedgerClientV4", "TurnResultV4"]
