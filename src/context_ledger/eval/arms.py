"""The three pilot arms (D). All share the model, the window W, max_tokens 1,024 per call, the
answer format and the system prompt framing, and each receives a GuardedClient, never a raw one.

B3  RAG. One shot per question, a fresh conversation, no tools: BM25 top-k chunks of the rho
    index for the question text, filling the window to W minus the reserve (system, question
    and max_tokens). The system prompt and the retrieved context carry cache breakpoints, but
    the context changes per question, so few cache hits are expected.
B4  Agent with LLM compaction. One conversation per episode with corpus_search and corpus_read.
    When the estimated window (system plus conversation) reaches 85% of W, a separate
    summarization call (same model, same guard, a fixed versioned prompt) replaces all history
    except the system prompt and the current question with its summary. Its cost counts toward
    B4. The episode is logged as a plain JSONL transcript under runs/<run_id>/B4/.
L   Ledger hybrid. One conversation per episode through the M2 LedgerClient and Session, with
    the corpus tools (corpus_read registers sources and spans) and the four ctx tools, the
    payoff-gated backstop at hard 85% and target 50% of W, and the estimator in force. Since
    BUILD-V2 (mechanism "l-v2", recorded in run.json) L reminds the model once per question to
    refetch folded spans naming the question's identifiers before an abstention stands. Since
    BUILD-V2b (mechanism "l-v2b") the backstop judges W on the request about to be sent as
    count_tokens counts it minus L's tool definitions: calib-v3 under-counts L's alias headers,
    fold lines and small span blocks by up to half, so L was held to about 1.7 W (p50). And it
    folds as often as it takes, before a call, to bring the window to hard (or abstains with
    budget_exceeded if it cannot get out of the mandatory zone). Since BUILD-V2c (mechanism
    "l-v2c", backstop rules "v2c") the floor is bounded: what an earlier turn left, backstop
    folds included, is always foldable, a refold names the earlier fold by its one alias, and
    in the mandatory zone a run may fold into one digest line (backstop.py, "Bounded floor"),
    so before a question's first call L abstains with budget_exceeded only when the system
    prompt and the question alone are over hard. Since BUILD-V3 (mechanism "l-v3") every answer
    L would emit passes the answer gate (verify/gate.py) with the question's text: a cited quote
    must witness the asked target and hold the answer's value, and a cited note must not have
    been updated by a later note; and a failed claim is cleared only by a superseding claim
    that passes.

Conditions (BUILD-V3). "allowed" (prompt clp-v3-allowed) lets every arm answer UNANSWERABLE;
"forced" (prompt clp-v3-forced) tells every arm alike always to give a best guess. B3 and B4
make their calls as in allowed; an UNANSWERABLE answer is kept, scored as an abstention by the
unchanged grader and flagged forced_violation. L still verifies every claim and gates every
answer; an UNANSWERABLE gets one message asking for a best guess, and if the turn ends (the
round bound) while its latest candidate answer is blocked, the row's answer is that candidate,
flagged forced_fallback (reason forced_blocked; the ledger records an abstention). With no
candidate after an UNANSWERABLE the row is an abstention flagged forced_violation. Every row
records its condition.

Session facts (BUILD-V2c, questions v2s). A question may carry a note, part of its user
message for every arm (`with_note`): "[note:<pos>]" then the note text, before the question; a
session answer cites note:<pos>. B4 keeps notes in its conversation, where compaction may
summarize or drop them. B3 retrieves the episode's earlier notes as documents next to the
corpus chunks, scored by the same FTS5 BM25 (CorpusIndex.bm25). L registers each note as a
source "note:<pos>" and carries it as a verbatim span in the user turn (header
"[@A note:<pos>#0-n]"), as corpus_read carries a file: folding, ctx_search, ctx_refetch and
Tier 1 then treat it like any span.

Reads: corpus_read returns at most 200 lines or min(2,000, W // 4) estimated tokens of content,
in B4 and L alike (BUILD-V2b; recorded in run.json as read_cap_tokens), so one read is a quarter
of the window at most; in L each span counts with its alias header (BUILD-V2c, eval/tools.py).

Bounds: at most 10 model calls per question (hitting it counts as an abstention flagged
bound_hit) and 90 per episode. A SpendGuard refusal mid-episode stops the run cleanly: results
so far are kept and the run is marked incomplete. W counts the system prompt and the
conversation; tool definitions (fixed per arm) are not counted. Under calib-v3 (BUILD-V2) W is
the whole prompt as count_tokens counts it minus the arm's tool definitions, envelopes included,
measured the same way in every arm (envelope.py): B3 fills chunks while the prompt stays within
W - max_tokens, B4 compacts at 85% of it and L's backstop acts at hard 85% / target 50% of it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..anthropic.client import LedgerClient
from ..anthropic.cache import CachePredictor
from ..anthropic.request import PILOT_THINKING, cache_control
from ..anthropic.usage import content_blocks, handling, normalize_usage, response_dict
from ..envelope import CONTEXT_OPEN, prompt_tokens
from ..fold import BackstopPolicyV1
from ..ids import canonical_json, sha256_hex
from ..live.guarded import GuardedClient
from ..live.redact import api_error_record
from ..live.spend import BudgetRefused, GuardInvariantBroken
from ..pricing import PricingTable
from ..prompts import AnswerFormatError
from ..session import Session
from ..store import Ledger, SqliteBlobStore
from ..tokens import envelope_of, source_class
from ..tools import CTX_TOOLS
from ..verify.gate import asked_target
from .corpus import CorpusIndex, lines_of
from .formats import format_read
from .prompts import COMPACTION_PROMPT, PILOT_PROMPT_VERSION, PROMPTS, PromptVersion, prompt_sha256, system_prompt
from .reconcile import fold_reconciliation
from .tools import CORPUS_TOOLS, LedgerCorpusTools, PlainCorpusTools, cited_lines, read_cap

MAX_TOKENS = 1024
MAX_CALLS_QUESTION = 10
MAX_CALLS_EPISODE = 90
COMPACT_BP = 8500
SEARCH_DEPTH = 200
SUMMARY_PREFIX = "Summary of the earlier conversation:\n"
MAX_API_ERRORS = 3
L_MECHANISM = "l-v3"  # BUILD-V2 (v2 Tier 1 witnesses, fix messages, refetch before abstaining) + BUILD-V2b (W measured with count_tokens, enforced) + BUILD-V2c (bounded floor) + BUILD-V3 (answer gate)
# BUILD-V3FIX, in run.json and every L results row (absent = unpatched): L never sends an empty text block (the
# request builder drops them; an earlier claim's refetch obligation gets a message, not an empty one), and the
# fake server refuses one with the API's 400. L's mechanism is l-v3 otherwise unchanged.
HARNESS_PATCH = "v3.1-empty-text-cache-control"
# BUILD-V4FIX, recorded (after HARNESS_PATCH, joined by "+") in run.json and every L, L4 and L4-nogate row of a run with a
# v4 arm (B5, L4 or L4-nogate): a thinking block is sent back only while what preceded it is unchanged (request.py), and a
# response cut at max_tokens (or refused) while holding tool uses gets an error result for each (client.py). The code is
# shared, so it is in force in a run of B3, B4 and L alone too, which still writes v3's configuration exactly (BUILD-V4):
# with thinking disabled (Haiku) neither fix can fire there unless a response is cut while holding a tool use, which no
# v3 ledger shows (the v4 grid had two, both in Haiku forced rho 10000).
HARNESS_PATCH_V4FIX = "v4.1-thinking-binding+cut-tool-use"
CONDITIONS = ("allowed", "forced")


class RunStopped(Exception):
    """The run cannot go on (the guard refused or halted, or the API keeps failing)."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


@dataclass
class ArmConfig:
    model: str
    window: int
    pricing: PricingTable
    estimator: Any
    run_dir: Path
    max_tokens: int = MAX_TOKENS
    max_calls_question: int = MAX_CALLS_QUESTION
    max_calls_episode: int = MAX_CALLS_EPISODE
    prompt_version: str = PILOT_PROMPT_VERSION
    tool_definitions: dict[str, int] | None = None  # each arm's tool-definition tokens; L's measure needs L's
    condition: str = "allowed"  # BUILD-V3: "allowed" or "forced"
    address: Any = None  # BUILD-V4: the address index (index.AddressIndex) at the run's rho; B5 and L4 only

    @property
    def prompt(self) -> PromptVersion:
        return PROMPTS[self.prompt_version]

    def system(self, arm: str) -> str:
        return system_prompt(arm, self.prompt_version)

    @property
    def thinking(self) -> dict:
        return PILOT_THINKING[self.model]

    @property
    def min_cache_tokens(self) -> int:
        return self.pricing.price(self.model).min_cache_tokens


@dataclass
class QuestionResult:
    """What one arm did on one question; scored later by score.py."""

    arm: str
    qid: str
    episode: int
    answer: Any = None
    abstained: bool = True
    reason: str = "not_run"
    bound_hit: bool = False
    calls: int = 0
    cost_pusd: int = 0
    citations: list = field(default_factory=list)
    claim_statuses: list = field(default_factory=list)
    blocked_answers: list = field(default_factory=list)
    compactions: int = 0
    folds: int = 0
    error: dict | None = None
    refetch_reminders: int | None = None  # L only (BUILD-V2): 0 or 1 refetch-before-abstaining reminder
    refetches: int | None = None  # L only: ctx_refetch calls this turn
    verdict_reasons: list | None = None  # L only: the verdict reason of every claim of the turn, in order
    condition: str = "allowed"  # BUILD-V3: the condition the row ran under
    forced_violation: bool = False  # BUILD-V3, forced: the row abstained because the model wrote UNANSWERABLE
    forced_fallback: bool = False  # BUILD-V3, forced, L only: the answer is L's last blocked candidate
    gate_reason: str | None = None  # L only (BUILD-V3): the reason of the turn's last answer-gate verdict, None if the gate never ran
    gate_target: str | None = None  # L only: the target the gate parses from the question text, None if there is none
    gate_reasons: list | None = None  # L only: every answer-gate verdict reason of the turn, in order
    cleared_failures: list | None = None  # L only: failed claims of the turn a passing superseding claim cleared

    def to_dict(self) -> dict:
        return asdict(self)


def _citations(claims) -> list[dict]:
    return [{"ref": c.ref, "quote": c.quote} for claim in claims for c in claim.citations]


class CallLog:
    """Every model call of the run, appended to runs/<run_id>/calls.jsonl: usage, exact cost
    and the predicted prompt and cache-read tokens beside the API's numbers, and the key profile
    that paid (when given)."""

    def __init__(self, path: Path, key_profile: str | None = None) -> None:
        self.path = path
        self.rows: list[dict] = []
        self.extra = {} if key_profile is None else {"key_profile": key_profile}

    def add(self, **row) -> dict:
        row.update(self.extra)
        canonical_json(row)
        self.rows.append(row)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        return row


def _create(client: GuardedClient, request: dict, log: CallLog, cache: CachePredictor | None, cfg: ArmConfig, **tags) -> dict:
    """One guarded call, logged with the predicted cache read (capped at the counted prompt).
    Guard refusals stop the run; API errors are re-raised."""
    try:
        response = client.messages.create(**request)
    except (BudgetRefused, GuardInvariantBroken) as exc:
        raise RunStopped("budget_refused" if isinstance(exc, BudgetRefused) else "guard_halted", str(exc)) from None
    predicted_read = cache.predict(request, client.last_prompt_count) if cache is not None else 0
    raw = response_dict(response)
    usage = normalize_usage(raw.get("usage"), "5m")
    cost = cfg.pricing.cost_pusd(cfg.model, usage)
    log.add(**tags, model=cfg.model, usage=usage, cost_pusd=cost, predicted_prompt=client.last_prompt_count,
            predicted_cache_read=predicted_read, stop_reason=raw.get("stop_reason"))  # fmt: skip
    return raw


def _estimate(cfg: ArmConfig, text: str, cls: str = "other") -> int:
    return cfg.estimator.count(text, cls)


def note_piece(q: dict) -> str | None:
    """A question's note as B3 and B4 show it: "[note:<pos>]", then the note text; None if
    the question has no note."""
    note = q.get("note")
    return None if note is None else f"[note:{q['pos']}]\n{note['text']}"


def with_note(q: dict, question: str) -> str:
    """The user message of `q`: its note piece, if any, then `question`."""
    piece = note_piece(q)
    return question if piece is None else f"{piece}\n\n{question}"


# B3: RAG


def run_b3(questions: list[dict], index: CorpusIndex, client: GuardedClient, cfg: ArmConfig, log: CallLog) -> list[QuestionResult]:
    if not isinstance(client, GuardedClient):
        raise TypeError("every arm receives a GuardedClient")
    system = cfg.system("B3")
    cache = CachePredictor(cfg.estimator, cfg.min_cache_tokens)  # the system prompt's prefix is shared by every question

    env = envelope_of(cfg.estimator)

    def build(chunks: list[str], question: str) -> dict:
        content = []
        if chunks:
            content.append({"type": "text", "text": CONTEXT_OPEN + "\n\n".join(chunks) + "\n</context>", "cache_control": cache_control("5m")})
        content.append({"type": "text", "text": question})
        return {
            "model": cfg.model, "max_tokens": cfg.max_tokens, "thinking": cfg.thinking,
            "system": [{"type": "text", "text": system, "cache_control": cache_control("5m")}],
            "messages": [{"role": "user", "content": content}],
        }  # fmt: skip

    def ask(q: dict, notes: list[str]) -> QuestionResult:
        result = QuestionResult("B3", q["qid"], q["episode"], condition=cfg.condition)
        question = with_note(q, f"Question: {q['text']}")
        room = cfg.window - _estimate(cfg, system) - _estimate(cfg, question) - cfg.max_tokens
        chunks, used = [], 0
        for _, _, body, cls in retrieve(q["text"], notes):
            if env is not None:  # calib-v3: the whole prompt, envelopes included, within W - max_tokens
                if prompt_tokens(build([*chunks, body], question), cfg.estimator, env) <= cfg.window - cfg.max_tokens:
                    chunks.append(body)
                continue
            tokens = _estimate(cfg, body, cls)
            if used + tokens <= room:
                chunks.append(body)
                used += tokens
        request = build(chunks, question)
        raw = _create(client, request, log, cache, cfg, arm="B3", episode=q["episode"], qid=q["qid"], kind="answer")
        result.calls, result.cost_pusd = 1, log.rows[-1]["cost_pusd"]
        _finish_answer(result, raw, cfg)
        return result

    def retrieve(text: str, notes: list[str]) -> list[tuple[float, int, str, str]]:
        """The corpus chunks and the episode's earlier notes for `text`, best bm25 first (a
        chunk before a note on a tie), as (score, order, body, estimator class)."""
        found = [(score, n, format_read(hit.path, hit.start, hit.end, lines_of(index.text(hit.path), hit.start, hit.end)), source_class(hit.path))
                 for n, (score, hit) in enumerate(index.scored_search(text, SEARCH_DEPTH))]  # fmt: skip
        scored = [(index.bm25(text, note), len(found) + n, note, "other") for n, note in enumerate(notes)]
        return sorted(found + [s for s in scored if s[0] is not None])

    def episode(e: int):
        notes: list[str] = []  # the notes of the episode's earlier questions: B3's prior user turns

        def ask_in(q: dict) -> QuestionResult:
            result = ask(q, notes)
            if (piece := note_piece(q)) is not None:
                notes.append(piece)
            return result

        return ask_in

    return _episodes("B3", questions, episode)


def _finish_answer(result: QuestionResult, raw: dict, cfg: ArmConfig) -> bool:
    """Parse a final response into the result; False if it did not parse."""
    if handling(raw.get("stop_reason")) in ("max_tokens", "refusal"):
        result.reason = handling(raw.get("stop_reason"))
        return True
    text = "\n".join(b["text"] for b in content_blocks(raw.get("content") or []) if b["type"] == "text")
    try:
        parsed = cfg.prompt.parse(text)
    except AnswerFormatError:
        result.reason = "unparseable_answer"
        return False
    result.abstained = parsed.abstain
    result.reason = "model_abstained" if parsed.abstain else "answered"
    result.forced_violation = parsed.abstain and cfg.condition == "forced"
    result.answer = None if parsed.abstain else parsed.answer
    result.citations = [] if parsed.abstain else _citations(parsed.claims)
    return True


# B4: agent with LLM compaction


def _plain(block: dict) -> str:
    if block["type"] == "text":
        return block["text"]
    if block["type"] == "tool_use":
        return f"[tool_use {block['name']} {json.dumps(block['input'], sort_keys=True)}]"
    if block["type"] == "tool_result":
        return "\n".join(_plain(b) for b in block["content"])
    return ""  # thinking blocks are never summarized


class B4Episode:
    """One B4 conversation."""

    def __init__(self, index: CorpusIndex, client: GuardedClient, cfg: ArmConfig, log: CallLog, episode: int) -> None:
        if not isinstance(client, GuardedClient):
            raise TypeError("every arm receives a GuardedClient")
        self.index, self.client, self.cfg, self.log, self.episode = index, client, cfg, log, episode
        self.tools = PlainCorpusTools(index, lambda t, cls: cfg.estimator.count(t, cls), read_cap(cfg.window))
        self.system = cfg.system("B4")
        self.messages: list[dict] = []
        self.calls = 0
        self.cache = CachePredictor(cfg.estimator, cfg.min_cache_tokens)
        self.prev_last = 0  # blocks in the previous request's messages (its last block gets a breakpoint)
        directory = cfg.run_dir / "B4"
        directory.mkdir(parents=True, exist_ok=True)
        self.transcript = directory / f"episode-{episode:02d}.jsonl"

    def _log(self, **event) -> None:
        with open(self.transcript, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")

    def window_tokens(self) -> int:
        """The W measure of the conversation (system and messages, never the tools): under
        calib-v3 envelope.prompt_tokens, envelopes included; else the M4 text-only sum."""
        env = envelope_of(self.cfg.estimator)
        if env is not None:
            return prompt_tokens({"system": [{"type": "text", "text": self.system}], "messages": self.messages}, self.cfg.estimator, env)
        tokens = _estimate(self.cfg, self.system)
        for message in self.messages:
            for block in message["content"]:
                tokens += _estimate(self.cfg, _plain(block), "json" if block["type"] == "tool_result" else "other")
        return tokens

    def request(self) -> dict:
        messages = json.loads(json.dumps(self.messages))
        blocks = [b for m in messages for b in m["content"]]
        if blocks:
            marks = {len(blocks) - 1, self.prev_last - 1} - {-1}
            for n in marks:
                if n < len(blocks):
                    blocks[n]["cache_control"] = cache_control("5m")
        self.prev_last = len(blocks)
        tools = sorted((json.loads(json.dumps(t)) for t in CORPUS_TOOLS), key=lambda t: t["name"])
        return {"model": self.cfg.model, "max_tokens": self.cfg.max_tokens, "thinking": self.cfg.thinking, "tools": tools,
                "system": [{"type": "text", "text": self.system, "cache_control": cache_control("5m")}], "messages": messages}  # fmt: skip

    def compact(self, q: dict, question_block: dict) -> None:
        history = self.messages[:-1] if self.messages and self.messages[-1]["content"] == [question_block] else self.messages
        transcript = "\n".join(f"{m['role']}: {_plain(b)}" for m in history for b in m["content"] if _plain(b))
        request = {"model": self.cfg.model, "max_tokens": self.cfg.max_tokens, "thinking": self.cfg.thinking,
                   "system": [{"type": "text", "text": COMPACTION_PROMPT}],
                   "messages": [{"role": "user", "content": [{"type": "text", "text": transcript or "(nothing yet)"}]}]}  # fmt: skip
        raw = _create(self.client, request, self.log, None, self.cfg, arm="B4", episode=self.episode, qid=q["qid"], kind="summary")
        summary = "\n".join(b["text"] for b in content_blocks(raw.get("content") or []) if b["type"] == "text")
        self.messages = [{"role": "user", "content": [{"type": "text", "text": SUMMARY_PREFIX + summary}, question_block]}]
        self.prev_last = 0  # the tools and system prefix stays cached; the predictor knows by its bytes
        self._log(type="compaction", qid=q["qid"], summary=summary, sha256=sha256_hex(transcript.encode("utf-8")))

    def ask(self, q: dict) -> QuestionResult:
        result = QuestionResult("B4", q["qid"], q["episode"], condition=self.cfg.condition)
        question_block = {"type": "text", "text": with_note(q, q["text"])}
        self.messages.append({"role": "user", "content": [question_block]})
        self._log(type="question", qid=q["qid"], text=question_block["text"])
        calls = cost = 0
        asked_reformat = False
        start_rows = len(self.log.rows)
        while True:
            if calls >= self.cfg.max_calls_question or self.calls >= self.cfg.max_calls_episode:
                result.bound_hit, result.reason = True, "bound_hit"
                break
            if self.window_tokens() * 10000 >= COMPACT_BP * self.cfg.window and len(self.messages) > 1:
                self.compact(q, question_block)
                calls, self.calls, result.compactions = calls + 1, self.calls + 1, result.compactions + 1
                continue
            raw = _create(self.client, self.request(), self.log, self.cache, self.cfg, arm="B4", episode=self.episode, qid=q["qid"], kind="answer")
            calls, self.calls = calls + 1, self.calls + 1
            content = content_blocks(raw.get("content") or [])
            self.messages.append({"role": "assistant", "content": content or [{"type": "text", "text": "(empty)"}]})
            self._log(type="response", qid=q["qid"], content=content, stop_reason=raw.get("stop_reason"))
            if handling(raw.get("stop_reason")) == "tool_use":
                results = []
                for block in content:
                    if block["type"] == "tool_use":
                        text, is_error = self.tools(block["name"], block["input"])
                        results.append({"type": "tool_result", "tool_use_id": block["id"], "content": [{"type": "text", "text": text}], **({"is_error": True} if is_error else {})})
                        self._log(type="tool", qid=q["qid"], name=block["name"], input=block["input"], result=text, is_error=is_error)
                self.messages.append({"role": "user", "content": results})
                continue
            if _finish_answer(result, raw, self.cfg):
                break
            if asked_reformat:
                break
            asked_reformat = True
            self.messages.append({"role": "user", "content": [{"type": "text", "text": f"Verification: your answer could not be parsed. {self.cfg.prompt.reformat}"}]})
        result.calls = calls
        result.cost_pusd = sum(r["cost_pusd"] for r in self.log.rows[start_rows:])
        self._log(type="answer", qid=q["qid"], answer=result.answer, abstained=result.abstained, reason=result.reason)
        return result


def run_b4(questions: list[dict], index: CorpusIndex, client: GuardedClient, cfg: ArmConfig, log: CallLog) -> list[QuestionResult]:
    return _episodes("B4", questions, lambda e: B4Episode(index, client, cfg, log, e).ask)


# L: ledger hybrid


class LEpisode:
    """One L conversation: a Session over runs/<run_id>/L/episode-NN.sqlite and a LedgerClient."""

    def __init__(self, index: CorpusIndex, client: GuardedClient, cfg: ArmConfig, log: CallLog, episode: int) -> None:
        if not isinstance(client, GuardedClient):
            raise TypeError("every arm receives a GuardedClient")
        self.cfg, self.log, self.episode, self.client = cfg, log, episode, client
        directory = cfg.run_dir / "L"
        directory.mkdir(parents=True, exist_ok=True)
        self.db = directory / f"episode-{episode:02d}.sqlite"
        self.ledger, self.blobs = Ledger(self.db), SqliteBlobStore(self.db)
        policy = BackstopPolicyV1(budget_tokens=cfg.window, model=cfg.model, token_counter=cfg.estimator.name, rules="v2c")
        self.session = Session(self.ledger, self.blobs, cfg.window, policy, pricing=cfg.pricing, tokenizer=cfg.estimator)
        self.session.append_raw("system", cfg.system("L"), pinned=True)
        self.attempts: list[tuple[str, Any]] = []
        self.wrapper = LedgerClient(
            self.session, client, model=cfg.model, max_tokens=cfg.max_tokens, tools=CTX_TOOLS + CORPUS_TOOLS,
            dispatch=LedgerCorpusTools(index, read_cap(cfg.window)), thinking=cfg.thinking, predict=True,
            prompt=(cfg.prompt_version, prompt_sha256("L", cfg.prompt_version)), parse=cfg.prompt.parse, format_hint=cfg.prompt.reformat,
            on_answer=lambda turn_id, parsed: self.attempts.append((turn_id, parsed)), remind_refetch=True,
            tool_tokens=(cfg.tool_definitions or {}).get("L"), gate=True, forced=cfg.condition == "forced",
        )  # fmt: skip
        self.calls = 0

    def ask(self, q: dict) -> QuestionResult:
        result = QuestionResult("L", q["qid"], q["episode"], condition=self.cfg.condition)
        target = asked_target(q["text"])  # the question text only, as the gate reads it
        result.gate_target = target.label() if target is not None else None
        rounds = min(self.cfg.max_calls_question, self.cfg.max_calls_episode - self.calls)
        if rounds <= 0:
            result.bound_hit, result.reason = True, "bound_hit"
            return result
        start = len(self.session.events)
        if q.get("note") is not None:  # the note as a verbatim span of the user turn (module docstring)
            source = self.session.register_source(f"note:{q['pos']}", q["note"]["text"])
            span = self.session.register_text_span(source, 0, len(q["note"]["text"]))
            self.session.append_span(span.span_id, role="user")
        try:
            turn = self.wrapper.run_turn(q["text"], turn_id=q["qid"], max_rounds=rounds)
        except (BudgetRefused, GuardInvariantBroken) as exc:
            self._log_calls(q, start)
            raise RunStopped("budget_refused" if isinstance(exc, BudgetRefused) else "guard_halted", str(exc)) from None
        except Exception as exc:
            self._log_calls(q, start)
            result.error, result.reason = api_error_record(exc), "api_error"
            return result
        self.calls += len(turn.steps)
        rows = self._log_calls(q, start)
        result.calls, result.cost_pusd = len(turn.steps), sum(r["cost_pusd"] for r in rows)
        result.abstained, result.reason = turn.abstained, ("bound_hit" if turn.reason == "rounds_exhausted" else turn.reason)
        result.bound_hit = turn.bound_hit
        state = self.session.state  # span refs exported as path:start-end, so one checker grades every arm

        def cite(claims) -> list[dict]:
            return [{"ref": cited_lines(state, self.blobs, c.ref) or c.ref, "quote": c.quote, "alias": c.alias, "span": c.ref} for claim in claims for c in claim.citations]

        if not turn.abstained and turn.answer is not None:
            result.answer = turn.answer.answer
            result.citations = cite([c for c in turn.claims if c.claim_id in set(turn.emitted)])
        elif turn.fallback is not None:  # forced: the last blocked candidate stands in as the answer
            parsed, claims = turn.fallback
            result.answer, result.abstained, result.forced_fallback, result.citations = parsed.answer, False, True, cite(claims)
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
        return result

    def _log_calls(self, q: dict, start: int) -> list[dict]:
        rows = []
        for event in self.session.events[start:]:
            if event.type == "model_call":
                p = event.payload
                rows.append(self.log.add(arm="L", episode=self.episode, qid=q["qid"], kind="answer", model=p["model"], usage=p["usage"],
                                         cost_pusd=p["cost_pusd"], predicted_prompt=p["predicted"]["prompt"],
                                         predicted_cache_read=p["predicted"]["cache_read"], stop_reason=p["stop_reason"], call_id=p["call_id"],
                                         window=p["predicted"].get("window")))  # fmt: skip
        return rows

    def count(self, body: dict) -> int | None:
        """count_tokens through the guarded client (free); None if it fails."""
        try:
            result = self.client.messages.count_tokens(**body)
        except Exception:
            return None
        value = result["input_tokens"] if isinstance(result, dict) else getattr(result, "input_tokens", None)
        return value if isinstance(value, int) else None

    def close(self) -> None:
        """Write the episode's fold reconciliation (predicted vs actual), then close the ledger."""
        try:
            rows = fold_reconciliation(self.session.events, self.blobs, self.count, self.episode)
            with open(self.cfg.run_dir / "L" / "folds.jsonl", "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
        finally:
            self.ledger.close()
            self.blobs.close()


def run_l(questions: list[dict], index: CorpusIndex, client: GuardedClient, cfg: ArmConfig, log: CallLog) -> list[QuestionResult]:
    episodes: dict[int, LEpisode] = {}

    def make(e: int):
        episodes[e] = LEpisode(index, client, cfg, log, e)
        return episodes[e].ask

    try:
        return _episodes("L", questions, make)
    finally:
        for ep in episodes.values():
            ep.close()


def _episodes(arm: str, questions: list[dict], make: Callable[[int], Callable[[dict], QuestionResult]]) -> list[QuestionResult]:
    """Run each episode's questions in order through one conversation per episode."""
    out: list[QuestionResult] = []
    asked: dict[int, Callable[[dict], QuestionResult]] = {}
    errors = 0
    for q in questions:
        if q["episode"] not in asked:
            asked[q["episode"]] = make(q["episode"])
        try:
            result = asked[q["episode"]](q)
        except RunStopped as stop:
            stop.partial = out  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            result = QuestionResult(arm, q["qid"], q["episode"], reason="api_error", error=api_error_record(exc))
        errors += result.error is not None
        out.append(result)
        if errors >= MAX_API_ERRORS:
            stop = RunStopped("api_errors", "too many API errors")
            stop.partial = out  # type: ignore[attr-defined]
            raise stop
    return out


ARMS = {"B3": run_b3, "B4": run_b4, "L": run_l}


def set_aside_episode(run_dir: Path, arm: str, episode: int) -> list[Path]:
    """Before a unit is (re)run: rename an earlier attempt's B4 transcript or L ledger (with its
    -wal and -shm files) to *.discarded-<k>, so nothing is deleted and the new attempt starts
    clean. Returns the new names."""
    stem = {"B4": ("B4", ".jsonl", ("",)), "L": ("L", ".sqlite", ("", "-wal", "-shm")),
            "B5": ("B5", ".jsonl", ("",)), "L4": ("L4", ".sqlite", ("", "-wal", "-shm")), "L4-nogate": ("L4-nogate", ".sqlite", ("", "-wal", "-shm"))}.get(arm)  # fmt: skip
    if stem is None:
        return []
    directory, ext, suffixes = stem
    base = Path(run_dir) / directory / f"episode-{episode:02d}{ext}"
    if not base.exists():
        return []
    k = 1
    while Path(f"{base}.discarded-{k}").exists():
        k += 1
    moved = []
    for suffix in suffixes:
        old = Path(str(base) + suffix)
        if old.exists():
            new = Path(f"{base}.discarded-{k}{suffix}")
            old.rename(new)
            moved.append(new)
    return moved
