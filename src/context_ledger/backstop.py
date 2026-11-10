"""Payoff-gated backstop selection ("backstop-v1").

Cost model of a fold that replaces the contiguous window positions [i, j) with one summary
segment. Token counts are rendered tokens (render.py); money is pico-USD; prices are micro-USD
per MTok, so tokens x price is exact.

    L     rendered window tokens now
    P_i   rendered tokens of positions < i (byte-identical after the fold, so that cached
          prefix survives)
    D     (delta) rendered tokens of the victims minus rendered tokens of the summary segment,
          counted on the exact rendered summary string (header, aliases and newlines included)
    p_w   cache-write price for the policy TTL;  p_r  cache-read price
    G     generation cost of the summary: 0 for the extractive backstop; for a model fold the
          output tokens are already spent, so it is not counted either
    R     (R-hat) expected number of future calls that will read the window, the next included

Assuming the prefix is cached, the next call after the fold writes L - P_i - D tokens it would
otherwise have read, and every call, the next included, reads D fewer tokens:

    S_cache(i, j) = R * D * p_r - (L - P_i - D) * (p_w - p_r) - G

Break-even: S_cache > 0  <=>  R - 1 > R*, where
R* = ((L - P_i - D) * (p_w - p_r) + G) / (D * p_r) - 1 is `fold.payback_calls`.

Cache-cold rule: if the most recent `model_call` read nothing from the cache
(cache_read_input_tokens == 0), or there is no `model_call` yet, nothing is cached and the
rewrite term is 0: the next call writes everything anyway (folds after a TTL expiry are free).

Information-loss risk of folding verbatim evidence that may have to be re-fetched, per victim v:
age(v) = calls since v was last touched (appended, cited by a claim, or re-fetched);
q(v) = (q0_bp / 10000) / 2**(age(v) // half_life_calls);  rho(v) = (tok(v) + tau_tokens) * p_w;
Risk(i, j) = sum of q(v) * rho(v) over the victims.

Objective: J(i, j) = S_cache(i, j) - Risk(i, j), an exact Fraction of pico-USD.

Under calib-v3 (BUILD-V2) the window measure includes the envelope (envelope.py): position p
costs render.window_costs[p] and P_0 is the envelope base. The fold segment is a user text
block, so D also counts its text envelope, a message envelope if the position before i is not
user-side, and the change of the message envelope of the first message position q >= j, whose
predecessor becomes the fold: D = sum(cost[i:j]) - summary - text - message*[side(i-1) != user]
- message*([side(q) != user] - [side(q) != side(j-1)]). A run never splits a tool block, so no
tool_result envelope changes. This equals fold.fold_delta exactly (a property test checks it).

Measured window (BUILD-V2b). The estimator's L equals the calib-v3 measure of the request the
builder renders, exactly; but on L's content (alias headers, fold summary lines, many small
span blocks inside one tool_result) calib-v3 under-counts count_tokens by up to half (pilot
runs/pilot-v2a-haiku, W = 3000: L 3359 against 5118 counted at the same call). So a caller may
pass `measured`, the window as count_tokens counts the request it is about to send minus the
arm's tool definitions. The zones and the reach tests then use W = measured, and a run that
frees delta estimated tokens is predicted to leave W * (L - delta) / L (the measured tokens
per estimated token, applied to the rest); the payoff J stays in the estimator's units. The
decision record keeps "measured" so P9 re-runs the same choice. Without `measured`, W = L and
every decision is the same as before.

Bounded floor (BUILD-V2c, policy rules "v2c"). In pilot runs/pilot-v2b-haiku-w3000 (rho 10)
L abstained with budget_exceeded on 36 of its 40 questions, 30 of them before any call: the
window stayed above mandatory with nothing feasible for the rest of the episode. Two rules
caused it:
1. "recent" (appended after the latest call) never expired. A turn refused by BudgetExceeded
   makes no call, so the folds the backstop had just made, and every later question, stayed
   "recent", hence unfoldable, for good. Episode 16: one corpus_read of Lib/token.py put the
   first question at 3391 counted tokens (mandatory above 2850); the second question's fold
   freed 40; from then on the window was the system prompt, two backstop folds and questions,
   every one "recent", and all 18 later questions abstained with 0 calls.
2. The extractive summary is one line per victim, and a refold lists every sentence of the
   earlier fold. A line is about as large as a small span (that read was 56 one-line spans;
   its fold freed 40 of ~1300 tokens), and larger than a short message, so earlier questions
   and folds never shrink when folded line by line: they pile up turn after turn.
Under v2c, "recent" protects only the current turn (positions from the current question on;
an earlier turn has emitted its answer) and never a fold segment (a summary is not new
evidence the model has yet to see), and a refold line names the earlier fold by its one alias.
In the mandatory zone, when no run reaches hard with the usual lines, runs holding an earlier
fold or whose lines would not free tokens may be folded into one digest line instead
(fold.digest_sentence: counts, then each earlier fold as a whole and each span by its one
alias), the summary-of-summaries; the decision records "digest". So the history before the
current question can always be folded down to a line, while the per-span lines that say what
was folded are kept whenever the window does not need the digest. Before the first call on a
question, BudgetExceeded therefore means the system prompt, the question and the envelope alone
are over hard. A digest's tokens come from prefix sums like the per-victim lines'
(`summary_tokens`), and equal fold.fold_delta exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from .alias import alias_for
from .fold import (
    BackstopPolicyV1,
    digest_cite,
    digest_sentence,
    digest_text,
    extractive_summary_v1,
    fold_header,
    fold_id_for,
    fold_line,
    plan_fold_aliases,
    refs_of,
    render_fold,
)
from .ids import frac_str
from .model import Event, SummarySentence
from .pricing import PricingTable
from .render import TokenCounter, render_segment, tool_blocks, tool_use_ids, window_costs, window_items
from .store.blobs import BlobStore
from .tokens import envelope_of, segment_class
from .window import WindowState

ZONES = ("normal", "soft", "discretionary", "mandatory")


class BudgetExceeded(Exception):
    """The window is in the mandatory zone and no feasible run frees tokens. No call is made."""

    def __init__(self, used: int, budget: int, reason: str) -> None:
        super().__init__(f"window of {used} rendered tokens exceeds the mandatory zone of a {budget}-token budget: {reason}")
        self.used, self.budget, self.reason = used, budget, reason


class FoldRejected(Exception):
    """A model fold failed the gate. `reason` is fold_would_grow_window, payback_not_met or
    infeasible_run; `details` carries the numbers (delta; J, rstar, rhat; position, why)."""

    def __init__(self, reason: str, **details: object) -> None:
        shown = " ".join(f"{k}={v}" for k, v in details.items())
        super().__init__(f"{reason} {shown}".strip())
        self.reason, self.details = reason, details


def zone_of(used: int, policy: BackstopPolicyV1) -> str:
    """normal (<= soft), soft (<= hard), discretionary (<= mandatory) or mandatory, by integer
    comparison U * 10000 > X_bp * budget."""
    over = lambda bp: used * 10000 > bp * policy.budget_tokens  # noqa: E731
    if over(policy.mandatory_bp):
        return "mandatory"
    if over(policy.hard_bp):
        return "discretionary"
    return "soft" if over(policy.soft_bp) else "normal"


def s_cache(rhat: int, delta: int, L: int, P_i: int, p_w: int, p_r: int, G: int = 0, cold: bool = False) -> int:
    """S_cache = rhat * delta * p_r - rewrite * (p_w - p_r) - G, rewrite = L - P_i - delta (0 if cold)."""
    rewrite = 0 if cold else L - P_i - delta
    return rhat * delta * p_r - rewrite * (p_w - p_r) - G


def rstar(delta: int, L: int, P_i: int, p_w: int, p_r: int, G: int = 0, cold: bool = False) -> Fraction | None:
    """R* = (rewrite * (p_w - p_r) + G) / (delta * p_r) - 1, or None when delta * p_r == 0."""
    if delta * p_r == 0:
        return None
    rewrite = 0 if cold else L - P_i - delta
    return Fraction(rewrite * (p_w - p_r) + G, delta * p_r) - 1


@dataclass(frozen=True)
class RunEval:
    """One feasible run [i, j): what it frees and its payoff."""

    i: int
    j: int
    delta: int
    J: Fraction
    rstar: Fraction | None


@dataclass(frozen=True)
class BackstopDecision:
    zone: str
    L: int
    P_i: int
    i: int
    j: int
    delta: int
    rhat: int
    cache_cold: bool
    J: Fraction
    rstar: Fraction | None
    victims: tuple[str, ...]
    measured: int | None = None
    cited_waived: bool = False
    digest: bool = False

    def record(self, policy: BackstopPolicyV1, pricing_hash: str) -> dict:
        """The FoldRecord.policy of a backstop-v1 fold: params plus this decision record (with
        "measured" only for a decision on a measured window, "cited_waived" only when the
        answered turns' citations were waived, "digest" only when the summary is a v2c digest)."""
        return {
            "id": policy.id,
            "params": policy.to_dict(),
            "pricing_hash": pricing_hash,
            "zone": self.zone,
            "L": self.L,
            "P_i": self.P_i,
            "i": self.i,
            "j": self.j,
            "delta": self.delta,
            "rhat": self.rhat,
            "cache_cold": self.cache_cold,
            "J": frac_str(self.J),
            "rstar": None if self.rstar is None else frac_str(self.rstar),
            **({} if self.measured is None else {"measured": self.measured}),
            **({"cited_waived": True} if self.cited_waived else {}),
            **({"digest": True} if self.digest else {}),
        }


@dataclass(frozen=True)
class Analysis:
    """Everything the backstop computed for one window: zone, used tokens (the estimator's L),
    R-hat, cache state, every feasible run with delta > 0 (best first), the decision (None if
    no fold) and the measured window, if one was given."""

    zone: str
    used: int
    rhat: int
    cache_cold: bool
    runs: tuple[RunEval, ...]
    decision: BackstopDecision | None
    measured: int | None = None

    @property
    def window(self) -> int:
        """The W measure the zones were judged on: the measured window, else the estimator's."""
        return self.used if self.measured is None else self.measured


def _tiebreak(run: RunEval) -> tuple:
    """Higher J, then larger i (longer cached prefix), then shorter, then smaller j."""
    return (run.J, run.i, -(run.j - run.i), -run.j)


class PayoffModel:
    """Precomputed per-position facts for one window, so every run is evaluated in O(1):
    rendered tokens, feasibility, risk, and the bytes of each victim's summary line."""

    def __init__(
        self,
        state: WindowState,
        events: Sequence[Event],
        blobs: BlobStore,
        policy: BackstopPolicyV1,
        pricing: PricingTable,
        counter: TokenCounter,
        measured: int | None = None,
    ) -> None:
        self.state, self.blobs, self.policy, self.counter, self.measured = state, blobs, policy, counter, measured
        price = pricing.price(policy.model)
        self.p_w, self.p_r = price.cache_write(policy.ttl), price.cache_read
        segments = state.segments
        n = self.n = len(segments)
        self.env = envelope_of(counter)
        if self.env is None:
            self.tok = [counter.count(render_segment(state, seg, blobs), segment_class(state, seg)) for seg in segments]
        else:
            items = window_items(state, blobs)
            self.tok = window_costs(items, counter, self.env)
            sides = [item.side for item in items]
            # side_before[i]: the side of the last message position before i; side_from[j]: of
            # the first message position at or after j (None if there is none)
            self.side_before, self.side_from = [None] * (n + 1), [None] * (n + 1)
            for p in range(n):
                self.side_before[p + 1] = sides[p] if sides[p] is not None else self.side_before[p]
            for p in range(n - 1, -1, -1):
                self.side_from[p] = sides[p] if sides[p] is not None else self.side_from[p + 1]
        self.prefix = [0 if self.env is None else self.env.base]
        for t in self.tok:
            self.prefix.append(self.prefix[-1] + t)
        self.L = self.prefix[-1]
        self.W = self.L if measured is None else measured
        self.zone = zone_of(self.W, policy)
        self.rhat = policy.estimator.rhat(len(state.calls))

        # One pass over the ledger: when each segment entered, touches, recent citations, cache.
        calls = 0
        appended_seq: dict[str, int] = {}
        appended_calls: dict[str, int] = {}
        touched: dict[str, int] = {}
        claims_by_call: dict[str, list[tuple[str | None, set[str]]]] = {}  # call -> (turn, cited refs) per claim
        emitted: set[str] = set()
        last_read: int | None = None
        # The current question: the first user segment appended since the last answer_emitted
        # (or since the start). It is never a fold victim, as B4's compaction never summarizes
        # the current question; folding it would leave the model answering a question it can
        # no longer see.
        self.current_question: str | None = None
        awaiting_question = True
        for event in events:
            p = event.payload
            if event.type == "call":
                calls += 1
            elif event.type == "answer_emitted":
                awaiting_question = True
                emitted.add(p["turn_id"])
            elif event.type in ("segment_appended", "fold"):
                sid = p["segment"]["segment_id"] if event.type == "segment_appended" else p["fold"]["summary_segment_id"]
                appended_seq[sid], appended_calls[sid] = event.seq, calls
                if awaiting_question and event.type == "segment_appended" and p["segment"]["role"] == "user":
                    self.current_question, awaiting_question = sid, False
            elif event.type == "claim":
                refs = {c["ref"] for c in p["claim"]["citations"]}
                claims_by_call.setdefault(p["claim"]["call_id"], []).append((p["claim"].get("turn_id"), refs))
                touched.update(dict.fromkeys(refs, calls))
            elif event.type == "refetch":
                touched.update({r: calls for r in (p["span_id"], p["new_span_id"]) if r})
            elif event.type == "model_call":
                last_read = p["usage"]["cache_read_input_tokens"]
        self.cache_cold = not last_read
        recent_calls = state.calls[-policy.cold_m :] if policy.cold_m else []
        recent_claims = [claim for c in recent_calls for claim in claims_by_call.get(c, [])]
        cited = set().union(*(refs for _, refs in recent_claims))
        # cites of claims whose turn has not emitted its answer: all that still protects in the
        # mandatory zone once nothing else can be folded (see `analysis`)
        cited_open = set().union(*(refs for turn, refs in recent_claims if turn not in emitted))
        last_call_seq = state.last_call_seq

        self.blocks = tool_blocks(state, blobs)
        # A completed tool use/result pair (every tool_use of the block answered in it) is not
        # "recent": only tail_keep protects the newest one (BUILD-V2b). Calls do not advance
        # while BudgetExceeded refuses them, so "recent" would otherwise never expire.
        complete = [False] * n
        for start, end in set(self.blocks):
            if end - start > 1 and set(tool_use_ids(segments[start], blobs)) <= {seg.tool_use_id for seg in segments[start + 1 : end]}:
                complete[start:end] = [True] * (end - start)
        # tail_keep counts the current turn's items only (from the current question on): a turn
        # that ended without an answer call (budget_exceeded, rounds exhausted) leaves its last
        # tool pair at the end of the window, and it must not stay protected into the next turn.
        q_pos = next((p for p, seg in enumerate(segments) if seg.segment_id == self.current_question), 0)
        self.why: list[str | None] = []
        self.why_waived: list[str | None] = []
        shifts = []
        for p, seg in enumerate(segments):
            refs = refs_of(seg, state)
            if seg.pinned or seg.role == "system":
                why = "pinned"
            elif p >= max(n - policy.tail_keep, q_pos):
                why = "tail"
            elif seg.segment_id == self.current_question:
                why = "current_question"
            elif (
                not complete[p]
                and (last_call_seq is None or appended_seq.get(seg.segment_id, 0) > last_call_seq)
                and not (policy.bounded_floor and (p < q_pos or seg.ref_kind == "fold"))  # v2c: the current turn only, never a fold
            ):
                why = "recent"
            elif refs & cited:
                why = "cited"
            else:
                why = None
            self.why.append(why)
            self.why_waived.append(None if why == "cited" and not refs & cited_open else why)
            last_touch = max([appended_calls.get(seg.segment_id, 0)] + [touched[r] for r in refs if r in touched])
            shifts.append((calls - last_touch) // policy.half_life_calls)
        # Risk scaled by D = 10000 * 2**max_shift, so every run's risk is an exact integer / D.
        top = max(shifts, default=0)
        self.D = 10000 * 2**top
        self.risk = [0]
        for p in range(n):
            q_scaled = policy.q0_bp * 2 ** (top - shifts[p])
            self.risk.append(self.risk[-1] + q_scaled * (self.tok[p] + policy.tau_tokens) * self.p_w)

        # Extractive summary line bytes per victim (fast path for counters that depend only on
        # the byte length, as every estimator of tokens.py does).
        aliases = state.aliases
        self.count_bytes = getattr(counter, "count_bytes", None)
        self.fast = self.count_bytes is not None
        self.line_bytes = [0]
        self.slow: list[bool] = []
        self.sentences: list[SummarySentence] = []
        for sentence in extractive_summary_v1(state, [seg.segment_id for seg in segments], blobs, rules=policy.rules):
            shown = [c if c == "inference" else aliases.alias_of(c) for c in sentence.cites]
            self.sentences.append(sentence)
            self.slow.append(None in shown)
            line = fold_line("", sentence.text, " ".join(s for s in shown if s is not None))
            self.line_bytes.append(self.line_bytes[-1] + len(line.encode("utf-8")))
        self.base = base = aliases.count
        self.header_bytes = len(fold_header(alias_for(base)))
        self.alias_bytes = [0]
        for k in range(n):
            self.alias_bytes.append(self.alias_bytes[-1] + len(alias_for(base + 1 + k)))
        self.slow_prefix = [0]
        for flag in self.slow:
            self.slow_prefix.append(self.slow_prefix[-1] + flag)
        # v2c, mandatory zone: a run may be one digest line (fold.digest_sentence); per position,
        # prefix sums of folds, spans, digest cites, their alias bytes and cites without an alias
        self.bounded = policy.bounded_floor
        self.digest_prefix = [(0, 0, 0, 0, 0)]
        for seg in segments:
            cite = digest_cite(seg)
            shown = aliases.alias_of(cite) if cite is not None else None
            step = (seg.ref_kind == "fold", seg.ref_kind == "span", cite is not None, len(shown or ""), cite is not None and shown is None)
            self.digest_prefix.append(tuple(a + b for a, b in zip(self.digest_prefix[-1], step)))

    def victims(self, i: int, j: int) -> list[str]:
        return [seg.segment_id for seg in self.state.segments[i:j]]

    def summary_tokens(self, i: int, j: int, digest: bool = False) -> int:
        """Tokens of the exact rendered extractive summary of [i, j) (with `digest`, of its v2c
        digest line)."""
        return self._digest_tokens(i, j) if digest else self._lines_tokens(i, j)

    def digest_applies(self, i: int, j: int) -> bool:
        """v2c: [i, j) holds an earlier fold, or its per-victim lines would not free tokens."""
        folds = self.digest_prefix[j][0] - self.digest_prefix[i][0]
        return folds > 0 or self.delta(i, j) <= 0

    def _lines_tokens(self, i: int, j: int) -> int:
        if self.fast and self.slow_prefix[j] == self.slow_prefix[i]:
            size = self.header_bytes + self.line_bytes[j] - self.line_bytes[i] + self.alias_bytes[j - i]
            return self.count_bytes(size, "other")
        return self._render(i, j, self.sentences[i:j])

    def _digest_tokens(self, i: int, j: int) -> int:
        folds, spans, cites, cite_bytes, unaliased = (b - a for a, b in zip(self.digest_prefix[i], self.digest_prefix[j]))
        if self.fast and not unaliased:  # the cites go between the brackets, one space apart (raw messages have none)
            text = digest_text(folds, j - i - folds - spans, spans)
            size = self.header_bytes + len(fold_line(alias_for(self.base + 1), text, "").encode("utf-8")) + cite_bytes + max(cites - 1, 0)
            return self.count_bytes(size, "other")
        return self._render(i, j, [digest_sentence(self.state.segments[i:j])])

    def _render(self, i: int, j: int, sentences: list[SummarySentence]) -> int:
        fold_id = fold_id_for("backstop", i, self.victims(i, j))
        plan = plan_fold_aliases(self.state.aliases, fold_id, sentences)
        return self.counter.count(render_fold(fold_id, sentences, plan, self.state.aliases), "other")

    def seam(self, i: int, j: int) -> int:
        """The envelope part of the cost of folding [i, j) beyond the summary text (0 without an
        envelope): the fold's text envelope, its message envelope if the side before i is not
        user, and the change of the message envelope at the first message position q >= j."""
        env = self.env
        if env is None:
            return 0
        cost = env.text + (env.message if self.side_before[i] != "user" else 0)
        after = self.side_from[j]
        if after is not None:
            cost += env.message * ((after != "user") - (after != self.side_before[j]))
        return cost

    def delta(self, i: int, j: int, digest: bool = False) -> int:
        """Tokens folding [i, j) into its extractive summary (with `digest`, its v2c digest
        line) frees (fold.fold_delta, exactly)."""
        return self.prefix[j] - self.prefix[i] - self.summary_tokens(i, j, digest) - self.seam(i, j)

    def infeasible(self, i: int, j: int) -> tuple[int, str] | None:
        """(position, why) of the first position of [i, j) that cannot be folded, or None. A run
        that would split a tool use from its results fails with why "tool_pair"."""
        for p in range(i, j):
            if self.why[p] is not None:
                return p, self.why[p]
        if self.blocks[i][0] != i:
            return i, "tool_pair"
        if self.blocks[j - 1][1] != j:
            return j - 1, "tool_pair"
        return None

    def evaluate(self, i: int, j: int, delta: int) -> RunEval:
        """Payoff of folding [i, j) into a summary that frees `delta` tokens."""
        s = s_cache(self.rhat, delta, self.L, self.prefix[i], self.p_w, self.p_r, 0, self.cache_cold)
        J = Fraction(s * self.D - (self.risk[j] - self.risk[i]), self.D)
        return RunEval(i, j, delta, J, rstar(delta, self.L, self.prefix[i], self.p_w, self.p_r, 0, self.cache_cold))

    def analysis(self) -> Analysis:
        """Every feasible run with delta > 0 and the decision for the current zone (A.5). In
        the mandatory zone with no feasible run, the claims of turns that have emitted their
        answer stop protecting what they cite (BUILD-V2b; the decision records cited_waived):
        else no call is made, the cold_m calls never pass, and every later question abstains.
        v2c rules, mandatory zone: if no such run reaches hard, the runs whose summary would be
        a digest line (`digest_applies`) are tried too, the same way, and their decision is
        taken if the other reaches nothing or frees less (the decision records digest)."""
        runs, waived = self._tier()
        decision = self._decide(runs, waived)
        hard = self.policy.hard_bp
        if self.bounded and self.zone == "mandatory" and not (decision is not None and self._reaches(decision.delta, hard)):
            digests, digest_waived = self._tier(True)
            chosen = self._decide(digests, digest_waived, True)
            if chosen is not None and (decision is None or self._reaches(chosen.delta, hard) or chosen.delta > decision.delta):
                runs, decision = digests, chosen
        return Analysis(self.zone, self.L, self.rhat, self.cache_cold, runs, decision, self.measured)

    def _tier(self, digest: bool = False) -> tuple[tuple[RunEval, ...], bool]:
        runs = self._runs(self.why, digest)
        if not runs and self.zone == "mandatory" and self.why_waived != self.why:
            return self._runs(self.why_waived, digest), True
        return runs, False

    def _runs(self, why: list[str | None], digest: bool = False) -> tuple[RunEval, ...]:
        """The feasible runs with delta > 0 under the per-position reasons `why`, best first;
        with `digest`, only those `digest_applies` to, each folded into its digest line."""
        scored = []  # (J * D, i, j, delta): integers, so ranking is exact and fast
        for i in range(self.n):
            if why[i] is not None or self.blocks[i][0] != i:
                continue
            for j in range(i + 1, self.n + 1):
                if why[j - 1] is not None:
                    break
                if self.blocks[j - 1][1] != j:  # would split a tool use from its results
                    continue
                if digest and not self.digest_applies(i, j):
                    continue
                delta = self.delta(i, j, digest)
                if delta <= 0:
                    continue
                s = s_cache(self.rhat, delta, self.L, self.prefix[i], self.p_w, self.p_r, 0, self.cache_cold)
                scored.append((s * self.D - (self.risk[j] - self.risk[i]), i, j, delta))
        scored.sort(key=lambda r: (r[0], r[1], -(r[2] - r[1]), -r[2]), reverse=True)
        return tuple(self.evaluate(i, j, delta) for _, i, j, delta in scored)

    def _reaches(self, delta: int, bp: int) -> bool:
        """W * (L - delta) / L <= bp * budget / 10000; with W = L it is (L - delta) * 10000 <= bp * budget."""
        return self.W * (self.L - delta) * 10000 <= bp * self.policy.budget_tokens * self.L

    def _decide(self, runs: tuple[RunEval, ...], cited_waived: bool = False, digest: bool = False) -> BackstopDecision | None:
        if self.zone == "discretionary":
            positive = [r for r in runs if r.J > 0]
            pool = [r for r in positive if self._reaches(r.delta, self.policy.target_bp)] or positive
            best = max(pool, key=_tiebreak, default=None)
        elif self.zone == "mandatory":
            pool = [r for r in runs if self._reaches(r.delta, self.policy.hard_bp)]
            best = max(pool, key=_tiebreak) if pool else max(runs, key=lambda r: (r.delta, *_tiebreak(r)), default=None)
        else:
            best = None
        if best is None:
            return None
        return BackstopDecision(
            self.zone, self.L, self.prefix[best.i], best.i, best.j, best.delta, self.rhat, self.cache_cold,
            best.J, best.rstar, tuple(self.victims(best.i, best.j)), self.measured, cited_waived, digest,
        )  # fmt: skip


def select_backstop_v1(
    state: WindowState,
    events: Sequence[Event],
    blobs: BlobStore,
    policy: BackstopPolicyV1,
    pricing: PricingTable,
    counter: TokenCounter,
    measured: int | None = None,
) -> BackstopDecision | None:
    """Pure, deterministic backstop-v1 choice for this window, or None (`measured`: the
    count_tokens window the decision was made on, see the module docstring).

    Zones (A.5): at or below hard nothing; discretionary: the feasible run with maximal J, only
    if J > 0, preferring runs that bring the window down to target; mandatory: payback waived,
    the max-J run among those reaching hard, else max delta then max J (v2c: then the digest
    runs, see `PayoffModel.analysis`). Ties: higher J, larger
    i, shorter run, smaller j. Feasible runs (A.4) hold only unpinned, non-system positions
    outside the last tail_keep, other than the current question, appended before the latest
    call or part of a completed tool use/result pair (v2c rules: or before the current question,
    or a fold segment), not cited by a claim of the last cold_m calls (in the mandatory zone, if
    nothing else is feasible, only by a claim of a turn not yet answered), never split a tool
    use from its results, and free tokens (delta > 0). In the mandatory zone None means no run
    is feasible (the session then raises BudgetExceeded)."""
    return PayoffModel(state, events, blobs, policy, pricing, counter, measured).analysis().decision


__all__ = [
    "Analysis",
    "BackstopDecision",
    "BudgetExceeded",
    "FoldRejected",
    "PayoffModel",
    "RunEval",
    "ZONES",
    "rstar",
    "s_cache",
    "select_backstop_v1",
    "zone_of",
]
