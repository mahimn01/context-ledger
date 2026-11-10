"""SpendGuard: a hard, fail-closed global budget for paid API calls.

Storage. One ledger per API key (KEY_LEDGERS; the key profile picks it), each with its own
hash chain and lock, all gitignored: the aperture key's `.clp-spend/spend.jsonl` plus
`.clp-spend/lock`, and the research key's `.clp-spend/research/spend.jsonl` plus
`.clp-spend/research/lock`. An offline run uses `runs/<run_id>/fake-spend.jsonl` instead,
never a real one. One JSON line per record:

    {seq, ts, kind, run_id, model, request_sha256, prompt_tokens_upper, max_tokens,
     worst_case_pusd, actual_pusd, usage, outcome, ref, prev_hash, hash}

`kind` is reserve, settle, void, assume_spent or use_reserve; `ref` is the seq of the
reservation a settle, void or assume_spent closes (null on a reserve). A use_reserve record
(amounts null) marks that the run opened its guard with --use-reserve. Records are
hash-chained: `hash` is the sha256 of the record's canonical JSON without `hash`, and
`prev_hash` the previous record's hash ("0" * 64 first). Only hashes and counts are stored: no
headers, keys or message content.

Locking and durability. An exclusive non-blocking flock on `lock` is held for the guard's whole
life, so a second live process refuses to start. Each record is written, flushed and fsynced
before the network call that depends on it.

Cap. Each key has its own. The aperture key's HARD_MAX_USD is $150 (raised from $10 to $50 by
Mahimn on 2026-10-31, then to $150 on 2026-11-01). The research key's RESEARCH_HARD_MAX_USD is
$200 (Mahimn, 2026-11-01), with a RESEARCH_RESERVE_USD of $20 held back: its runs and
reservations must fit under $180 unless the run passes --use-reserve. Only Mahimn changes these,
by editing the constants. The global cap is the key's hard max (less its reserve without
--use-reserve), lowered by --global-cap if given; a run's cap (--budget) must fit in what
remains, or the guard refuses to start. spent = settled actual costs + assume_spent worst
cases; outstanding = open reservations. A reservation is refused, before any network call, if it would take the run or
the global total over its cap, or if its prompt bound exceeds PROMPT_CEILING (90,000 tokens: a
10% margin under Haiku 5.5's 100k long-context line, so the pilot never enters that tier).

On load the guard refuses a broken chain, an unparseable line, an unknown kind, or an open
reservation left by a crashed run until `spend resolve --assume-spent` closes it at its worst
case.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ..ids import GENESIS_HASH, canonical_json, sha256_hex
from ..pricing import PICO_PER_USD, PricingTable, usd

HARD_MAX_USD = 150  # Only Mahimn changes this: the absolute ceiling on all paid calls with the aperture key.
HARD_MAX_PUSD = HARD_MAX_USD * PICO_PER_USD
RESEARCH_HARD_MAX_USD = 200  # Only Mahimn changes this: the absolute ceiling on all paid calls with the research key.
RESEARCH_RESERVE_USD = 20  # Only Mahimn changes this: held back under the research key's ceiling unless --use-reserve.
PROMPT_CEILING = 90_000
KINDS = ("reserve", "settle", "void", "assume_spent", "use_reserve")
FIELDS = (
    "seq", "ts", "kind", "run_id", "model", "request_sha256", "prompt_tokens_upper", "max_tokens",
    "worst_case_pusd", "actual_pusd", "usage", "outcome", "ref", "prev_hash", "hash",
)  # fmt: skip
_USD = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]{1,12})?$")


@dataclass(frozen=True)
class KeyLedger:
    """One API key's spend ledger (its own hash chain and lock) and caps, by key profile."""

    profile: str
    ledger: str  # the ledger's path under the spend directory; its lock is the sibling `lock`
    hard_max_usd: int
    reserve_usd: int

    def path(self, spend_dir: str | Path) -> Path:
        return Path(spend_dir) / self.ledger

    def cap_pusd(self, use_reserve: bool = False) -> int:
        """What spent plus outstanding may reach: the hard max, less the reserve unless --use-reserve."""
        return (self.hard_max_usd - (0 if use_reserve else self.reserve_usd)) * PICO_PER_USD


KEY_LEDGERS = {
    "research": KeyLedger("research", "research/spend.jsonl", RESEARCH_HARD_MAX_USD, RESEARCH_RESERVE_USD),
    "aperture": KeyLedger("aperture", "spend.jsonl", HARD_MAX_USD, 0),
}


class SpendError(Exception):
    """Base of every SpendGuard refusal."""


class SpendLedgerCorrupt(SpendError):
    """The spend ledger does not parse or its chain does not verify."""


class OpenReservation(SpendError):
    """A crashed run left a reservation open; resolve it before spending again."""


class SpendLocked(SpendError):
    """Another live process holds the spend lock."""


class BudgetRefused(SpendError):
    """A cap or the prompt ceiling refuses this spend (no network call was made)."""


class GuardInvariantBroken(SpendError):
    """A call cost more than its reserved worst case. The run halts."""


def usd_to_pusd(text: str | int) -> int:
    """Exact pico-USD of a plain decimal dollar amount ("0.75", "10", 4)."""
    text = str(text)
    if _USD.match(text) is None:
        raise ValueError(f"not a dollar amount: {text!r}")
    return int(Fraction(text) * PICO_PER_USD)


def prompt_upper_from_count(count: int) -> int:
    """ceil(count * 1.05) + 64: the prompt bound from the free count_tokens endpoint."""
    return -((-count * 105) // 100) + 64


def prompt_upper_fallback(body: dict, estimator=None) -> int:
    """The conservative bound when counting fails: ceil(calib-v1 upper-envelope estimate of the
    body x 1.5) + 2048, or ceil(bytes / 2) + 2048 without a calibration."""
    data = {k: v for k, v in body.items() if k != "max_tokens"}
    size = len(canonical_json(data))
    if estimator is not None and getattr(estimator, "estimator", None) == "calib-v1":
        return -((-estimator.count_bytes(size, "json") * 3) // 2) + 2048
    return (size + 1) // 2 + 2048


def request_sha256(body: dict) -> str:
    return sha256_hex(canonical_json(body))


def record_hash(record: dict) -> str:
    return sha256_hex(canonical_json({k: v for k, v in record.items() if k != "hash"}))


def read_spend(path: str | Path) -> list[dict]:
    """Every record of a spend ledger, verified. Raises SpendLedgerCorrupt on an unparseable
    line, an unknown kind, a missing field, a broken chain or a close of no open reservation."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    prev = GENESIS_HASH
    open_seqs: set[int] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            raise SpendLedgerCorrupt(f"line {n} does not parse") from None
        if not isinstance(record, dict) or set(record) != set(FIELDS):
            raise SpendLedgerCorrupt(f"line {n} does not have exactly the spend record fields")
        if record["kind"] not in KINDS:
            raise SpendLedgerCorrupt(f"line {n} has unknown kind {record['kind']!r}")
        if record["seq"] != n or record["prev_hash"] != prev or record_hash(record) != record["hash"]:
            raise SpendLedgerCorrupt(f"line {n}: the hash chain does not verify")
        if record["kind"] == "reserve":
            open_seqs.add(record["seq"])
        elif record["kind"] == "use_reserve":
            pass  # a marker: it closes nothing
        elif record["ref"] not in open_seqs:
            raise SpendLedgerCorrupt(f"line {n} closes reservation {record['ref']!r}, which is not open")
        else:
            open_seqs.discard(record["ref"])
        records.append(record)
        prev = record["hash"]
    return records


@dataclass(frozen=True)
class Totals:
    spent: int
    outstanding: int
    open: tuple[int, ...]  # seqs of open reservations
    per_run: dict
    per_model: dict


def totals(records: list[dict]) -> Totals:
    """spent (settled actual + assume_spent worst), outstanding (open reservations), and the
    spent amount per run and per model, in pico-USD."""
    reserves = {r["seq"]: r for r in records if r["kind"] == "reserve"}
    closed = {r["ref"] for r in records if r["kind"] != "reserve"}
    spent, per_run, per_model = 0, {}, {}
    for r in records:
        amount = r["actual_pusd"] if r["kind"] == "settle" else r["worst_case_pusd"] if r["kind"] == "assume_spent" else 0
        if amount:
            spent += amount
            per_run[r["run_id"]] = per_run.get(r["run_id"], 0) + amount
            per_model[r["model"]] = per_model.get(r["model"], 0) + amount
    still_open = tuple(sorted(set(reserves) - closed))
    return Totals(spent, sum(reserves[s]["worst_case_pusd"] for s in still_open), still_open, per_run, per_model)


@dataclass(frozen=True)
class Reservation:
    seq: int
    model: str
    request_sha256: str
    prompt_tokens_upper: int
    max_tokens: int
    worst_case_pusd: int


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class SpendGuard:
    """The fail-closed budget of one process. `key` gives the caps (default: the aperture key's,
    the original single ledger; open_live passes the key profile's). `run_cap_usd` (--budget)
    defaults to everything that remains under the global cap. `use_reserve` releases the key's
    reserve and is written to the ledger as a use_reserve record. `resolve=True` opens a ledger
    with open reservations, only so `resolve_assume_spent` can close them."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        pricing: PricingTable,
        key: KeyLedger = KEY_LEDGERS["aperture"],
        use_reserve: bool = False,
        global_cap_usd: str | int | None = None,
        run_cap_usd: str | int | None = None,
        lock_path: str | Path | None = None,
        clock: Callable[[], str] = _now,
        resolve: bool = False,
    ) -> None:
        self.path = Path(path)
        self.run_id, self.pricing, self.clock = run_id, pricing, clock
        self.key, self.use_reserve = key, use_reserve
        self.halted = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = open(lock_path if lock_path is not None else self.path.parent / "lock", "a+")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise SpendLocked(f"another process holds {self._lock.name}") from None
        try:
            self.records = read_spend(self.path)
            state = totals(self.records)
            if state.open and not resolve:
                raise OpenReservation(f"reservations {list(state.open)} were left open by a crashed run; run `spend resolve --assume-spent`")
            self.global_cap = key.cap_pusd(use_reserve)
            if global_cap_usd is not None:
                self.global_cap = min(self.global_cap, usd_to_pusd(global_cap_usd))
            remaining = self.global_cap - state.spent - state.outstanding
            if run_cap_usd is None:
                self.run_cap = max(0, remaining)
            else:
                self.run_cap = usd_to_pusd(run_cap_usd)
                if self.run_cap > remaining:
                    raise BudgetRefused(f"--budget {run_cap_usd} is more than the {usd(max(0, remaining))} left under the global cap{self._cap_note()}")
            if use_reserve and key.reserve_usd and not resolve:
                self._append(kind="use_reserve", outcome="use_reserve")
        except BaseException:
            self.close()
            raise

    # State

    def _cap_note(self) -> str:
        """Whose cap refused, and whether the key's reserve is held back."""
        if self.key.reserve_usd and not self.use_reserve:
            return f" ({self.key.profile} key: ${self.key.hard_max_usd} less its ${self.key.reserve_usd} reserve, which only --use-reserve releases)"
        return f" ({self.key.profile} key)"

    def _totals(self) -> Totals:
        return totals(self.records)

    @property
    def spent(self) -> int:
        return self._totals().spent

    @property
    def outstanding(self) -> int:
        return self._totals().outstanding

    @property
    def remaining(self) -> int:
        """What the global cap still allows, after spent and outstanding."""
        t = self._totals()
        return self.global_cap - t.spent - t.outstanding

    def run_used(self) -> int:
        """This run's spent plus its open reservations."""
        t = self._totals()
        open_here = sum(r["worst_case_pusd"] for r in self.records if r["seq"] in t.open and r["run_id"] == self.run_id)
        return t.per_run.get(self.run_id, 0) + open_here

    # Writing

    def _append(self, **fields) -> dict:
        prev = self.records[-1]["hash"] if self.records else GENESIS_HASH
        record = {key: None for key in FIELDS}
        record.update(fields, seq=len(self.records) + 1, ts=self.clock(), run_id=self.run_id, prev_hash=prev)
        record["hash"] = record_hash(record)
        line = canonical_json(record).decode("utf-8") + "\n"
        with open(self.path, "a", encoding="utf-8") as out:
            out.write(line)
            out.flush()
            os.fsync(out.fileno())
        self.records.append(record)
        return record

    def reserve(self, *, model: str, request_sha256: str, prompt_tokens_upper: int, max_tokens: int) -> Reservation:
        """Reserve the worst case of a call before it is made, or raise BudgetRefused."""
        if self.halted:
            raise GuardInvariantBroken("the guard halted after a call cost more than its reservation")
        if prompt_tokens_upper > PROMPT_CEILING:
            raise BudgetRefused(f"a {prompt_tokens_upper}-token prompt bound is above the {PROMPT_CEILING}-token ceiling")
        worst = self.pricing.worst_case_pusd(model, prompt_tokens_upper, max_tokens)
        t = self._totals()
        if t.spent + t.outstanding + worst > self.global_cap:
            raise BudgetRefused(f"worst case {worst} pUSD would exceed the global cap ({self.remaining} pUSD left){self._cap_note()}")
        if self.run_used() + worst > self.run_cap:
            raise BudgetRefused(f"worst case {worst} pUSD would exceed the run cap ({self.run_cap - self.run_used()} pUSD left)")
        record = self._append(
            kind="reserve", model=model, request_sha256=request_sha256, prompt_tokens_upper=prompt_tokens_upper,
            max_tokens=max_tokens, worst_case_pusd=worst, outcome="reserved",
        )  # fmt: skip
        return Reservation(record["seq"], model, request_sha256, prompt_tokens_upper, max_tokens, worst)

    def _close(self, kind: str, reservation: Reservation, outcome: str, **extra) -> dict:
        return self._append(
            kind=kind, model=reservation.model, request_sha256=reservation.request_sha256,
            prompt_tokens_upper=reservation.prompt_tokens_upper, max_tokens=reservation.max_tokens,
            worst_case_pusd=reservation.worst_case_pusd, outcome=outcome, ref=reservation.seq, **extra,
        )  # fmt: skip

    def settle(self, reservation: Reservation, usage: dict, ttl: str = "5m") -> int:
        """Record the API-reported cost of a successful call. If it is above the reserved worst
        case, the record is still written, then GuardInvariantBroken halts the run."""
        actual = self.pricing.cost_pusd(reservation.model, usage, ttl)
        self._close("settle", reservation, "ok", actual_pusd=actual, usage=dict(usage))
        if actual > reservation.worst_case_pusd:
            self.halted = True
            raise GuardInvariantBroken(f"call cost {actual} pUSD, more than its {reservation.worst_case_pusd} pUSD reservation")
        return actual

    def void(self, reservation: Reservation, outcome: str) -> None:
        """A failed request with an HTTP error response: not billed."""
        self._close("void", reservation, outcome, actual_pusd=0)

    def assume_spent(self, reservation: Reservation, outcome: str) -> None:
        """An ambiguous failure (timeout, connection error): counted at its worst case."""
        self._close("assume_spent", reservation, outcome)

    def resolve_assume_spent(self) -> list[int]:
        """Close every open reservation as assume_spent at its worst case; returns their seqs."""
        t = self._totals()
        by_seq = {r["seq"]: r for r in self.records}
        for seq in t.open:
            r = by_seq[seq]
            self._close("assume_spent", Reservation(seq, r["model"], r["request_sha256"], r["prompt_tokens_upper"], r["max_tokens"], r["worst_case_pusd"]), "resolved_after_crash")
        return list(t.open)

    def status(self) -> dict:
        """The key's ledger_status, plus this guard's global cap, what remains under it, the run's
        cap and whether --use-reserve was given (pico-USD)."""
        status = ledger_status(self.key, self.records)
        return {
            **status,
            "global_cap_pusd": self.global_cap,
            "remaining_pusd": self.global_cap - status["spent_pusd"] - status["outstanding_pusd"],
            "run_cap_pusd": self.run_cap,
            "use_reserve": self.use_reserve,
        }

    def close(self) -> None:
        if not self._lock.closed:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()

    def __enter__(self) -> SpendGuard:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def ledger_status(key: KeyLedger, records: list[dict]) -> dict:
    """One key's ledger as `spend status` shows it (pico-USD): its hard max, reserve and usable
    cap (the hard max less the reserve), spent, outstanding, what remains usable without
    --use-reserve, spent per run and per model, its open reservations and the runs that
    passed --use-reserve."""
    t = totals(records)
    usable = key.cap_pusd(use_reserve=False)
    return {
        "key_profile": key.profile,
        "hard_max_pusd": key.hard_max_usd * PICO_PER_USD,
        "reserve_pusd": key.reserve_usd * PICO_PER_USD,
        "usable_cap_pusd": usable,
        "spent_pusd": t.spent,
        "outstanding_pusd": t.outstanding,
        "remaining_usable_pusd": usable - t.spent - t.outstanding,
        "per_run_pusd": dict(sorted(t.per_run.items())),
        "per_model_pusd": dict(sorted(t.per_model.items())),
        "open": list(t.open),
        "reserve_runs": [r["run_id"] for r in records if r["kind"] == "use_reserve"],
        "records": len(records),
    }


def render_status(status: dict) -> str:
    """One key's ledger_status as plain text, dollars to 6 decimals, every line led by the key
    profile (so a script can pick one line with a fixed prefix)."""
    k = status["key_profile"]
    lines = [
        f"{k} hard max         {usd(status['hard_max_pusd'])}",
        f"{k} reserve          {usd(status['reserve_pusd'])}" + (" (only --use-reserve releases it)" if status["reserve_pusd"] else ""),
        f"{k} usable cap       {usd(status['usable_cap_pusd'])}",
        f"{k} spent            {usd(status['spent_pusd'])}",
        f"{k} outstanding      {usd(status['outstanding_pusd'])}",
        f"{k} remaining usable {usd(status['remaining_usable_pusd'])}",
    ]
    lines += [f"{k} run {run}: {usd(v)}" for run, v in status["per_run_pusd"].items()]
    lines += [f"{k} model {model}: {usd(v)}" for model, v in status["per_model_pusd"].items()]
    lines += [f"{k} --use-reserve given by runs {status['reserve_runs']}"] if status["reserve_runs"] else []
    return "\n".join(lines)


def render_combined(statuses: list[dict]) -> str:
    """The total over every key's ledger_status: spent, outstanding and remaining usable."""
    total = lambda field: usd(sum(s[field] for s in statuses))  # noqa: E731
    return "\n".join([
        f"combined spent            {total('spent_pusd')}",
        f"combined outstanding      {total('outstanding_pusd')}",
        f"combined remaining usable {total('remaining_usable_pusd')}",
    ])  # fmt: skip
