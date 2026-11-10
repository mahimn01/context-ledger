"""The two v4 arms (BUILD-V4 items 3-7). B3, B4 and L (l-v3) are in arms.py, unchanged.

L4  (mechanism "l-v4") L with the address index: one conversation per episode through
    LedgerClientV4, the address tools (lookup, read, outline, search at most twice per question)
    and the four ctx tools, the same backstop, W, read cap and bounds (10 calls per question, 90
    per episode) as L. Per question: the question text is the turn's first user segment (so the
    backstop protects it), then its note as a verbatim span, then PRE-RESOLUTION: the harness
    parses the asked target from the question text with the gate's own parser
    (verify.gate.asked_target) and looks it up in the index; an exact address is read into the
    window as verbatim spans and its address named; a miss gives the address-miss certificate
    (no exact address, the nearest DIFFERENT identifiers); a session target gives the latest
    note of its (service, attribute) chain, refetched if it was folded. Every answer passes the
    gate; a gate refusal is REPAIRED (the anchored span or latest note put in the window, the
    reason named) at most twice per question. The harness's own reads (pre-resolution and repair)
    take the target's anchored span whole when it fits W // 2 (address_tools.preresolve_cap), so
    its anchored path witnesses the target; otherwise its first lines under the read cap; forced, the fallback goes through repair and the
    gate too and is emitted only as unverified_guess (anthropic/client_v4.py).
L4-nogate (mechanism "l-v4-nogate", BUILD-V4B decision 3) L4 exactly, with the answer gate off and
    therefore no gate-triggered repair: Tier 0/1, pre-resolution, tools, prompts (L4's) and the
    forced handling are L4's (forced, a candidate Tier 0/1 still blocks at the end is emitted as
    unverified_guess). Rows say gate "off" (L4's "on").
B5  The control: B3 plus the same pre-resolution and the lookup, outline and read tools, with
    NO gate and NO repair. A fresh conversation per question, as B3: the first user message
    holds the BM25 context (chunks and the episode's earlier notes, as B3 retrieves them), the
    pre-resolved text (a read block "[path:start-end]": whole when it fits W // 2 and the first
    request still fits W - max_tokens, else its first lines under the read cap; or the latest
    note "[note:<pos>]") with its address or the certificate, and the question with its note.
    The prompt stays within W - max_tokens: before each call the lowest-ranked chunk is dropped,
    then the oldest tool result is elided to a one-line stub naming its address, while it is
    over; if it is still over, B5 abstains with budget_exceeded. At most 10 calls per question
    and 90 per episode, as B4, L and L4. The answer is parsed as B3's; one reformat request as
    B4. Transcripts: runs/<run_id>/B5/episode-NN.jsonl.

Pre-resolution reads only the question TEXT and user-visible note text (`pre_resolve` takes a
string, never the question dict), so no gold or metadata field can reach an arm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..anthropic.cache import CachePredictor
from ..anthropic.client_v4 import LedgerClientV4
from ..anthropic.request import cache_control
from ..anthropic.usage import content_blocks, handling
from ..envelope import CONTEXT_OPEN, prompt_tokens
from ..fold import BackstopPolicyV1
from ..index import Address, AddressIndex, NearMiss, NoteTable
from ..live.guarded import GuardedClient
from ..live.redact import api_error_record
from ..live.spend import BudgetRefused, GuardInvariantBroken
from ..render import span_label
from ..session import Session
from ..store import Ledger, SqliteBlobStore
from ..tokens import envelope_of, source_class
from ..tools import CTX_TOOLS
from ..verify.gate import Target, asked_target
from ..window import kappa
from .address_tools import ADDRESS_TOOLS, B5_TOOLS, LedgerAddressTools, PlainAddressTools, preresolve_cap
from .arms import SEARCH_DEPTH, ArmConfig, CallLog, QuestionResult, RunStopped, _create, _episodes, _estimate, _finish_answer, note_piece, with_note
from .corpus import CorpusIndex, lines_of
from .formats import format_read
from .prompts import prompt_sha256
from .reconcile import fold_reconciliation
from .tools import cited_lines, read_cap

L4_MECHANISM = "l-v4"  # l-v3's gate and Tier 1, plus the address index, pre-resolution, repair and the question-first turn (BUILD-V4)
L4_NOGATE_MECHANISM = "l-v4-nogate"  # L4 with the answer gate off, so no gate-triggered repair (BUILD-V4B decision 3)
B5_MECHANISM = "b5-v4"
NEAR_SHOWN = 4


@dataclass
class QuestionResultV4(QuestionResult):
    """A v4 row: QuestionResult's fields plus what the v4 mechanisms did."""

    preresolved: dict | None = None  # the pre-resolution: status, target, addresses, near misses / note chain
    repairs: int | None = None  # L4: gate refusals repaired (at most 2, the forced last-round repair included)
    repair_reasons: list | None = None  # L4: the reason of each repair, in order ("final" for the forced last-round one)
    unverified_guess: bool = False  # L4, forced: the answer is the last candidate, blocked after repair (scored as an answer)
    tool_counts: dict | None = None  # calls per tool name this question (lookup, read, outline, search, ctx_*)
    mechanism: str | None = None  # "l-v4", "l-v4-nogate" or "b5-v4"
    gate: str | None = None  # L4 "on", L4-nogate "off"
    stop_reason: str | None = None  # L4: the reason in v3's terms (bound_hit, model_abstained, budget_exceeded, ...) under unverified_guess and forced_noanswer


@dataclass(frozen=True)
class Resolution:
    """What pre-resolution found for a question: status "exact" or "miss" (code), "note" or
    "no_note" (session), or "no_target"."""

    status: str
    target: Target | None
    addresses: tuple[Address, ...] = ()
    near: tuple[NearMiss, ...] = ()
    chain: tuple[str, ...] = ()
    different: tuple[tuple[str, str, str], ...] = ()
    visited: int = 0

    def to_dict(self) -> dict:
        d: dict = {"status": self.status, "target": self.target.label() if self.target else None, "visited": self.visited}
        if self.addresses:
            d["addresses"] = [a.ref for a in self.addresses]
        if self.near:
            d["near"] = [n.address.key for n in self.near]
        if self.chain:
            d["chain"] = list(self.chain)
        return d


def pre_resolve(question_text: str, address: AddressIndex, notes: NoteTable) -> Resolution:
    """Pre-resolution from the question TEXT alone (and the notes the user showed): the asked
    target, parsed as the gate parses it, looked up exactly."""
    target = asked_target(question_text)
    if target is None:
        return Resolution("no_target", None)
    if target.kind == "session":
        chain = tuple(notes.chain(target.name, target.attribute or ""))
        return Resolution("note" if chain else "no_note", target, chain=chain, different=tuple(notes.different(target.name, target.attribute or "")))
    found = address.lookup(target.name, NEAR_SHOWN)
    return Resolution("exact" if found.exact else "miss", target, found.exact, found.near, visited=found.visited)


def asked_label(target: Target) -> str:
    return f"the {target.attribute} of service {target.name}" if target.kind == "session" else f"`{target.name}`"


def miss_text(res: Resolution) -> str:
    """The address-miss certificate."""
    if res.status == "no_note":
        head = f"Index: no note so far in this conversation states the {res.target.attribute} of service {res.target.name}."
        others = [f"the {a} of service {s} ({ref})" for s, a, ref in res.different]
        return head + (f" Notes on DIFFERENT pairs (not the one asked): {'; '.join(others)}." if others else "")
    head = f"Index: no exact address for `{res.target.name}` in this corpus."
    others = [f"`{n.address.key}` ({n.why}) {n.address.ref}" for n in res.near]
    return head + (f" Nearest DIFFERENT identifiers (none of them is the one asked): {'; '.join(others)}." if others else " No similar name either.")


# L4


class L4Episode:
    """One L4 conversation: a Session over runs/<run_id>/L4/episode-NN.sqlite and LedgerClientV4.
    A subclass may switch the answer gate off (L4NoGateEpisode); it keeps L4's prompts and tools."""

    ARM, GATE, MECHANISM = "L4", True, L4_MECHANISM

    def __init__(self, index: CorpusIndex, client: GuardedClient, cfg: ArmConfig, log: CallLog, episode: int) -> None:
        if not isinstance(client, GuardedClient):
            raise TypeError("every arm receives a GuardedClient")
        if cfg.address is None:
            raise ValueError("L4 needs the address index (ArmConfig.address)")
        self.cfg, self.log, self.episode, self.client = cfg, log, episode, client
        directory = cfg.run_dir / self.ARM
        directory.mkdir(parents=True, exist_ok=True)
        self.db = directory / f"episode-{episode:02d}.sqlite"
        self.ledger, self.blobs = Ledger(self.db), SqliteBlobStore(self.db)
        policy = BackstopPolicyV1(budget_tokens=cfg.window, model=cfg.model, token_counter=cfg.estimator.name, rules="v2c")
        self.session = Session(self.ledger, self.blobs, cfg.window, policy, pricing=cfg.pricing, tokenizer=cfg.estimator)
        self.session.append_raw("system", cfg.system("L4"), pinned=True)
        self.attempts: list[tuple[str, Any]] = []
        self.tools = LedgerAddressTools(index, read_cap(cfg.window), address=cfg.address)
        self.notes = NoteTable()
        self.note_spans: dict[str, str] = {}  # "note:<pos>" -> its span id
        self.wrapper = LedgerClientV4(
            self.session, client, model=cfg.model, max_tokens=cfg.max_tokens, tools=CTX_TOOLS + ADDRESS_TOOLS,
            dispatch=self.tools, thinking=cfg.thinking, predict=True,
            prompt=(cfg.prompt_version, prompt_sha256("L4", cfg.prompt_version)), parse=cfg.prompt.parse, format_hint=cfg.prompt.reformat,
            on_answer=lambda turn_id, parsed: self.attempts.append((turn_id, parsed)), remind_refetch=True,
            tool_tokens=(cfg.tool_definitions or {}).get(self.ARM), gate=self.GATE, forced=cfg.condition == "forced",
        )  # fmt: skip
        self.calls = 0

    # the asked target and its pre-resolution (the daily harness widens both to note names)

    def target(self, text: str) -> Target | None:
        return asked_target(text)

    def resolve(self, text: str) -> Resolution:
        return pre_resolve(text, self.cfg.address, self.notes)

    # evidence into the window

    def _show(self, span_id: str) -> str:
        s = self.session
        return f"{s.assign_alias(span_id)} ({span_label(s.state, span_id)})"

    def _hold(self, span_id: str) -> str:
        """Make `span_id` verbatim in the window (append it, or refetch it if it was folded or
        dropped) and return its alias; a verbatim span is left as it is."""
        s = self.session
        state = kappa(s.state, span_id).value
        if state == "verbatim":
            return s.assign_alias(span_id)
        if state == "unseen":
            s.append_span(span_id, role="user")
            return s.assign_alias(span_id)
        s.refetch(span_id)
        fresh = s.events[-1].payload["new_span_id"]
        if fresh is None:  # STALE or DANGLING (never for an immutable corpus or a note): show the stored text again
            s.append_span(span_id, role="user")
            fresh = span_id
        return s.assign_alias(fresh)

    def _hold_address(self, a: Address) -> list[str]:
        """The asked target's anchored span held verbatim, whole, when it fits preresolve_cap (W //
        2); else the spans covering the address's first lines as read covers them (the read cap).
        Their aliases."""
        from .tools import clamp_read

        s = self.session
        own = self.tools.symbol_span(s, a.path, a.start, a.end)
        if own is not None and self.tools.span_tokens(s, own) <= preresolve_cap(self.cfg.window):
            return [self._hold(own)]
        start, end, _ = clamp_read(self.tools.index, a.path, a.start, a.end, lambda t, cls: s.counter.count(t, cls), self.tools.cap)
        span_ids, _ = self.tools._cover(s, a.path, start, end)
        return [self._hold(span_id) for span_id in span_ids]

    def _evidence(self, res: Resolution) -> str:
        """Put the resolution's evidence in the window and say where it is."""
        if res.status == "exact":
            first, rest = res.addresses[0], res.addresses[1:]
            aliases = self._hold_address(first)
            shown = f", shown in this conversation as {', '.join(aliases)}" if aliases else ""
            text = f"Index: `{res.target.name}` is at {first.ref}{shown}."
            if rest:
                text += f" It is also defined at: {'; '.join(a.ref for a in rest)}."
            return text
        if res.status == "note":
            latest = res.chain[-1]
            alias = self._hold(self.note_spans[latest])
            older = [r for r in res.chain[:-1]]
            text = f"Index: the latest note stating the {res.target.attribute} of service {res.target.name} is {alias} ({latest})"
            return text + (f"; it updates {', '.join(older)}." if older else ".")
        return miss_text(res)

    def ask(self, q: dict) -> QuestionResultV4:
        result = QuestionResultV4(self.ARM, q["qid"], q["episode"], condition=self.cfg.condition, mechanism=self.MECHANISM, gate="on" if self.GATE else "off")
        text = q["text"]
        target = self.target(text)
        result.gate_target = target.label() if target is not None else None
        rounds = min(self.cfg.max_calls_question, self.cfg.max_calls_episode - self.calls)
        if rounds <= 0:
            result.bound_hit, result.reason = True, "bound_hit"
            return result
        start = len(self.session.events)
        note = q.get("note")
        resolved: list[Resolution] = []

        def preamble() -> None:
            if note is not None:  # the note: a verbatim span of the user turn, after the question
                ref = f"note:{q['pos']}"
                source = self.session.register_source(ref, note["text"])
                span = self.session.register_text_span(source, 0, len(note["text"]))
                self.session.append_span(span.span_id, role="user")
                self.note_spans[ref] = span.span_id
                self.notes.add(ref, note["text"])
            res = self.resolve(text)
            resolved.append(res)
            if res.status != "no_target":
                self.session.append_raw("user", self._evidence(res))

        def repair(verdict) -> str:
            return self._repair(text, verdict)

        try:
            turn = self.wrapper.run_turn_v4(text, turn_id=q["qid"], max_rounds=rounds, preamble=preamble, repair=repair)
        except (BudgetRefused, GuardInvariantBroken) as exc:
            self._log_calls(q, start)
            raise RunStopped("budget_refused" if isinstance(exc, BudgetRefused) else "guard_halted", str(exc)) from None
        except Exception as exc:
            made = self._log_calls(q, start)
            self.calls += len(made)
            result.error, result.reason = api_error_record(exc), "api_error"
            result.preresolved = resolved[0].to_dict() if resolved else None
            try:  # close the turn, so the backstop's current question moves on to the next one
                self.wrapper._abstain(q["qid"], "api_error")
            except Exception:
                pass  # the error is recorded in the row; a turn that cannot be closed stays open
            return result
        self.calls += len(turn.steps)
        rows = self._log_calls(q, start)
        result.calls, result.cost_pusd = len(turn.steps), sum(r["cost_pusd"] for r in rows)
        result.abstained, result.reason = turn.abstained, ("bound_hit" if turn.reason == "rounds_exhausted" else turn.reason)
        result.bound_hit = turn.bound_hit
        state = self.session.state

        def cite(claims) -> list[dict]:
            return [{"ref": cited_lines(state, self.blobs, c.ref) or c.ref, "quote": c.quote, "alias": c.alias, "span": c.ref} for claim in claims for c in claim.citations]

        if not turn.abstained and turn.answer is not None:
            result.answer = turn.answer.answer
            result.citations = cite([c for c in turn.claims if c.claim_id in set(turn.emitted)])
        elif turn.fallback is not None:  # forced: the last blocked candidate, after repair, as an unverified guess
            parsed, claims = turn.fallback
            result.answer, result.abstained, result.citations = parsed.answer, False, cite(claims)
            result.forced_fallback = result.unverified_guess = True
        result.forced_violation = turn.forced_violation
        result.claim_statuses = [c.status for c in turn.claims]
        result.refetch_reminders, result.refetches = turn.reminders, turn.refetches
        result.verdict_reasons = [(c.verdict or {}).get("reason") for c in turn.claims]
        result.gate_reasons = [g["reason"] for g in turn.gates]
        result.gate_reason = result.gate_reasons[-1] if result.gate_reasons else None
        result.cleared_failures = [{"status": c.status, "reason": (c.verdict or {}).get("reason"), "detail": (c.verdict or {}).get("detail", [])}
                                   for c in turn.cleared_failures]  # fmt: skip
        final = turn.answer
        result.blocked_answers = [p.answer for t, p in self.attempts if t == q["qid"] and p is not final and not p.abstain]
        result.folds = sum(e.type == "fold" for e in self.session.events[start:])
        result.repairs, result.repair_reasons = turn.repairs, list(turn.repair_reasons)
        result.stop_reason = "bound_hit" if turn.underlying == "rounds_exhausted" else turn.underlying
        result.preresolved = resolved[0].to_dict() if resolved else None
        result.tool_counts = self.tools.turn_counts(q["qid"])
        return result

    def _repair(self, question: str, verdict) -> str:
        """Put the evidence for the asked target in the window and say why the answer was refused
        (client_v4 item 2); `verdict` None is the forced last-round repair."""
        alt = "" if self.cfg.condition == "forced" else " If the text shown does not answer the question, you may abstain."
        if verdict is not None and verdict.reason == "stale_superseded" and verdict.stale:
            parts = []
            for old, new in verdict.stale:
                held = self._hold(new) if new in self.session.state.spans else new
                parts.append(f"your claim cites {self._show(old) if old in self.session.state.spans else old}, but the later note {held} updates it")
            return f"Repair (stale_superseded): {'; '.join(parts)}. The latest note is shown in this conversation: answer again from it.{alt}"
        target = self.target(question)
        if target is None:
            return f"Repair: your answer was not accepted. Answer again with claims that cite verbatim text.{alt}"
        res = self.resolve(question)
        reason = verdict.reason if verdict is not None else "final"
        why = {
            "target_not_witnessed": f"none of your cited verbatim quotes is about {asked_label(target)}",
            "value_not_with_target": f"your answer is not in a verbatim quote of the text about {asked_label(target)}",
        }.get(reason, "your answer did not pass verification")
        if verdict is not None and reason == "target_not_witnessed" and verdict.detail and verdict.detail[0] != target.name:
            why = f"no verbatim quote you cite about {asked_label(target)} names {', '.join(f'`{d}`' for d in verdict.detail)}"
        if reason == "value_not_with_target" and verdict is not None and verdict.detail:
            why += f" ({', '.join(f'`{d}`' for d in verdict.detail)})"
        evidence = self._evidence(res)
        if res.status in ("exact", "note"):
            return f"Repair ({reason}): {why}. {evidence} Cite that text with a verbatim quote of the line that gives the answer, and answer again.{alt}"
        return f"Repair ({reason}): {why}. {evidence}{alt}"

    def _log_calls(self, q: dict, start: int) -> list[dict]:
        rows = []
        for event in self.session.events[start:]:
            if event.type == "model_call":
                p = event.payload
                rows.append(self.log.add(arm=self.ARM, episode=self.episode, qid=q["qid"], kind="answer", model=p["model"], usage=p["usage"],
                                         cost_pusd=p["cost_pusd"], predicted_prompt=p["predicted"]["prompt"],
                                         predicted_cache_read=p["predicted"]["cache_read"], stop_reason=p["stop_reason"], call_id=p["call_id"],
                                         window=p["predicted"].get("window")))  # fmt: skip
        return rows

    def count(self, body: dict) -> int | None:
        try:
            result = self.client.messages.count_tokens(**body)
        except Exception:
            return None
        value = result["input_tokens"] if isinstance(result, dict) else getattr(result, "input_tokens", None)
        return value if isinstance(value, int) else None

    def close(self) -> None:
        try:
            rows = fold_reconciliation(self.session.events, self.blobs, self.count, self.episode)
            with open(self.cfg.run_dir / self.ARM / "folds.jsonl", "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
        finally:
            self.ledger.close()
            self.blobs.close()


class L4NoGateEpisode(L4Episode):
    """L4-nogate (BUILD-V4B decision 3): L4 with the answer gate off (client_v4, gate=False), so
    no gate-triggered repair; runs/<run_id>/L4-nogate/episode-NN.sqlite."""

    ARM, GATE, MECHANISM = "L4-nogate", False, L4_NOGATE_MECHANISM


def run_l4(questions: list[dict], index: CorpusIndex, client: GuardedClient, cfg: ArmConfig, log: CallLog, episode_class: type = L4Episode) -> list[QuestionResult]:
    episodes: dict[int, L4Episode] = {}

    def make(e: int):
        episodes[e] = episode_class(index, client, cfg, log, e)
        return episodes[e].ask

    try:
        return _episodes(episode_class.ARM, questions, make)
    finally:
        for ep in episodes.values():
            ep.close()


# B5


@dataclass
class B5Episode:
    """B5's state across an episode: the earlier questions' notes (B3's retrieval documents) and
    the note table; each question is a fresh conversation."""

    index: CorpusIndex
    client: GuardedClient
    cfg: ArmConfig
    log: CallLog
    episode: int
    notes: list[str] = field(default_factory=list)
    table: NoteTable = field(default_factory=NoteTable)
    note_texts: dict = field(default_factory=dict)  # "note:<pos>" -> its piece "[note:<pos>]\n<text>"
    calls: int = 0  # model calls of the episode, bounded by max_calls_episode as B4's and L's

    def __post_init__(self) -> None:
        if not isinstance(self.client, GuardedClient):
            raise TypeError("every arm receives a GuardedClient")
        if self.cfg.address is None:
            raise ValueError("B5 needs the address index (ArmConfig.address)")
        self.system = self.cfg.system("B5")
        self.tools = sorted((json.loads(json.dumps(t)) for t in B5_TOOLS), key=lambda t: t["name"])
        self.plain = PlainAddressTools(self.index, self.cfg.address, lambda t, cls: self.cfg.estimator.count(t, cls), read_cap(self.cfg.window))
        self.cache = CachePredictor(self.cfg.estimator, self.cfg.min_cache_tokens)
        self.env = envelope_of(self.cfg.estimator)
        directory = self.cfg.run_dir / "B5"
        directory.mkdir(parents=True, exist_ok=True)
        self.transcript = directory / f"episode-{self.episode:02d}.jsonl"

    def _log(self, **event) -> None:
        with open(self.transcript, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")

    def retrieve(self, text: str) -> list[tuple[float, int, str, str]]:
        """B3's retrieval: the corpus chunks and the episode's earlier notes, best bm25 first."""
        index = self.index
        found = [(score, n, format_read(hit.path, hit.start, hit.end, lines_of(index.text(hit.path), hit.start, hit.end)), source_class(hit.path))
                 for n, (score, hit) in enumerate(index.scored_search(text, SEARCH_DEPTH))]  # fmt: skip
        scored = [(index.bm25(text, note), len(found) + n, note, "other") for n, note in enumerate(self.notes)]
        return sorted(found + [s for s in scored if s[0] is not None])

    def evidence(self, res: Resolution, question: str) -> str:
        """The pre-resolved text: an exact address read whole when it fits preresolve_cap and the
        first request (system, it and the question) fits W - max_tokens, else its first lines under
        the read cap; the latest note; or the certificate."""
        if res.status == "exact":
            first, rest = res.addresses[0], res.addresses[1:]
            more = f" It is also defined at: {'; '.join(a.ref for a in rest)}." if rest else ""
            index_line = f"\n\nIndex: `{res.target.name}` is at {first.ref}, shown above.{more}"
            whole = "\n".join(self.index.text(first.path).split("\n")[first.start - 1 : first.end])
            if self.cfg.estimator.count(whole, source_class(first.path)) <= preresolve_cap(self.cfg.window):
                text = self.plain.read(first.path, first.start, first.end, cap=preresolve_cap(self.cfg.window)) + index_line
                if self.tokens(self.request([], text, question, [])) <= self.cfg.window - self.cfg.max_tokens:
                    return text
            return self.plain.read(first.path, first.start, first.end) + index_line
        if res.status == "note":
            latest = res.chain[-1]
            older = f"; it updates {', '.join(res.chain[:-1])}" if len(res.chain) > 1 else ""
            return f"{self.note_texts[latest]}\n\nIndex: the latest note stating the {res.target.attribute} of service {res.target.name} is {latest}, shown above{older}."
        return miss_text(res)

    def request(self, chunks: list[str], evidence: str | None, question: str, rest: list[dict]) -> dict:
        content = []
        if chunks:
            content.append({"type": "text", "text": CONTEXT_OPEN + "\n\n".join(chunks) + "\n</context>"})
        if evidence:
            content.append({"type": "text", "text": evidence})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}, *json.loads(json.dumps(rest))]
        blocks = [b for m in messages for b in m["content"]]
        blocks[-1]["cache_control"] = cache_control("5m")
        return {"model": self.cfg.model, "max_tokens": self.cfg.max_tokens, "thinking": self.cfg.thinking, "tools": self.tools,
                "system": [{"type": "text", "text": self.system, "cache_control": cache_control("5m")}], "messages": messages}  # fmt: skip

    def tokens(self, request: dict) -> int:
        """The W measure: system and messages, never the tools (calib-v3 envelopes, or the M4 sum)."""
        if self.env is not None:
            return prompt_tokens({"system": request["system"], "messages": request["messages"]}, self.cfg.estimator, self.env)
        return _estimate(self.cfg, self.system) + sum(_estimate(self.cfg, json.dumps(b)) for m in request["messages"] for b in m["content"])

    def ask(self, q: dict) -> QuestionResultV4:
        result = QuestionResultV4("B5", q["qid"], q["episode"], condition=self.cfg.condition, mechanism=B5_MECHANISM)
        if (piece := note_piece(q)) is not None:
            self.table.add(f"note:{q['pos']}", q["note"]["text"])
            self.note_texts[f"note:{q['pos']}"] = piece
        res = pre_resolve(q["text"], self.cfg.address, self.table)
        result.preresolved = res.to_dict()
        question = with_note(q, f"Question: {q['text']}")
        evidence = None if res.status == "no_target" else self.evidence(res, question)
        budget = self.cfg.window - self.cfg.max_tokens
        chunks: list[str] = []
        for _, _, body, _ in self.retrieve(q["text"]):
            if self.tokens(self.request([*chunks, body], evidence, question, [])) <= budget:
                chunks.append(body)
        rest: list[dict] = []  # the conversation after the first user message
        elided: set[str] = set()  # tool_use ids whose results were elided
        counts: dict[str, int] = {}
        calls, start_rows, asked_reformat = 0, len(self.log.rows), False
        self._log(type="question", qid=q["qid"], preresolved=result.preresolved, chunks=len(chunks))
        while True:
            if calls >= self.cfg.max_calls_question or self.calls >= self.cfg.max_calls_episode:
                result.bound_hit, result.reason = True, "bound_hit"
                break
            request = self.fit(chunks, evidence, question, rest, elided)
            if request is None:  # over W - max_tokens with nothing left to drop: abstain, as L does
                result.reason = "budget_exceeded"
                break
            raw = _create(self.client, request, self.log, self.cache, self.cfg, arm="B5", episode=self.episode, qid=q["qid"], kind="answer")
            calls += 1
            self.calls += 1
            content = content_blocks(raw.get("content") or [])
            rest.append({"role": "assistant", "content": content or [{"type": "text", "text": "(empty)"}]})
            self._log(type="response", qid=q["qid"], content=content, stop_reason=raw.get("stop_reason"))
            if handling(raw.get("stop_reason")) == "tool_use":
                results = []
                for block in content:
                    if block["type"] == "tool_use":
                        counts[block["name"]] = counts.get(block["name"], 0) + 1
                        text, is_error = self.plain(block["name"], block["input"])
                        results.append({"type": "tool_result", "tool_use_id": block["id"], "content": [{"type": "text", "text": text}], **({"is_error": True} if is_error else {})})
                        self._log(type="tool", qid=q["qid"], name=block["name"], input=block["input"], result=text, is_error=is_error)
                rest.append({"role": "user", "content": results})
                continue
            if _finish_answer(result, raw, self.cfg):
                break
            if asked_reformat:
                break
            asked_reformat = True
            rest.append({"role": "user", "content": [{"type": "text", "text": f"Verification: your answer could not be parsed. {self.cfg.prompt.reformat}"}]})
        result.calls = calls
        result.cost_pusd = sum(r["cost_pusd"] for r in self.log.rows[start_rows:])
        result.tool_counts = dict(sorted(counts.items()))
        self._log(type="answer", qid=q["qid"], answer=result.answer, abstained=result.abstained, reason=result.reason)
        if piece is not None:
            self.notes.append(piece)
        return result

    def fit(self, chunks: list[str], evidence: str | None, question: str, rest: list[dict], elided: set[str]) -> dict | None:
        """The request, within W - max_tokens: drop the lowest-ranked chunk, then elide the oldest
        tool result (never the latest round's) to a stub, while it is over; None if it is still
        over with nothing left to drop."""
        budget = self.cfg.window - self.cfg.max_tokens
        while True:
            request = self.request(chunks, evidence, question, rest)
            if self.tokens(request) <= budget:
                return request
            if chunks:
                chunks.pop()
                continue
            older = [b for m in rest[:-1] if m["role"] == "user" for b in m["content"] if b.get("type") == "tool_result" and b["tool_use_id"] not in elided]
            if not older:
                return None
            older[0]["content"] = [{"type": "text", "text": "[elided to keep the window: call the tool again, or read the address, if you need it]"}]
            elided.add(older[0]["tool_use_id"])


def run_b5(questions: list[dict], index: CorpusIndex, client: GuardedClient, cfg: ArmConfig, log: CallLog) -> list[QuestionResult]:
    return _episodes("B5", questions, lambda e: B5Episode(index, client, cfg, log, e).ask)


def run_l4_nogate(questions: list[dict], index: CorpusIndex, client: GuardedClient, cfg: ArmConfig, log: CallLog) -> list[QuestionResult]:
    return run_l4(questions, index, client, cfg, log, L4NoGateEpisode)


ARMS_V4 = {"B5": run_b5, "L4": run_l4, "L4-nogate": run_l4_nogate}

__all__ = ["ARMS_V4", "B5_MECHANISM", "L4_MECHANISM", "L4_NOGATE_MECHANISM", "B5Episode", "L4Episode", "L4NoGateEpisode", "QuestionResultV4", "Resolution", "miss_text",
           "pre_resolve", "run_b5", "run_l4", "run_l4_nogate"]
