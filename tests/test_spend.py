"""B.2: SpendGuard. Every refusal happens before any network call; the ledger is hash-chained,
locked, durable and fail-closed."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from hypothesis import given, settings, strategies as st

from context_ledger.live.spend import (
    HARD_MAX_PUSD,
    HARD_MAX_USD,
    PROMPT_CEILING,
    BudgetRefused,
    GuardInvariantBroken,
    OpenReservation,
    SpendGuard,
    SpendLedgerCorrupt,
    SpendLocked,
    prompt_upper_fallback,
    prompt_upper_from_count,
    read_spend,
    render_status,
    totals,
    usd_to_pusd,
)
from context_ledger.pricing import PICO_PER_USD, VERIFIED_PRICING_PATH, PricingTable
from context_ledger.tokens import CLASSES

TABLE = PricingTable.load(VERIFIED_PRICING_PATH, live=True)
HAIKU = "claude-haiku-5-5"
CLOCK = lambda: "2026-10-31T00:00:00+00:00"  # noqa: E731


def guard(tmp_path, **kw) -> SpendGuard:
    kw.setdefault("run_id", "run1")
    return SpendGuard(tmp_path / "spend.jsonl", pricing=TABLE, clock=CLOCK, **kw)


def reserve(g: SpendGuard, upper: int = 1000, max_tokens: int = 100):
    return g.reserve(model=HAIKU, request_sha256="ab" * 32, prompt_tokens_upper=upper, max_tokens=max_tokens)


def test_prompt_bounds():
    assert prompt_upper_from_count(1000) == 1050 + 64 and prompt_upper_from_count(1) == 2 + 64
    body = {"model": HAIKU, "max_tokens": 1024, "messages": [{"role": "user", "content": "x" * 1000}]}
    size = len(json.dumps({k: v for k, v in body.items() if k != "max_tokens"}, separators=(",", ":"), sort_keys=True))
    assert prompt_upper_fallback(body) == (size + 1) // 2 + 2048

    from fractions import Fraction

    from context_ledger.tokens import CalibEstimator, ClassFit

    est = CalibEstimator(HAIKU, {c: ClassFit(Fraction("0.3"), Fraction(2), Fraction("0.4")) for c in CLASSES}, "0" * 64)
    assert prompt_upper_fallback(body, est) == -((-est.count_bytes(size, "json") * 3) // 2) + 2048


def test_refusal_at_the_exact_boundary(tmp_path):
    worst = TABLE.worst_case_pusd(HAIKU, 1000, 100)
    with guard(tmp_path, global_cap_usd="1", run_cap_usd="1") as g:
        g.run_cap = 2 * worst + 1  # the run may spend exactly two worst cases, plus one pico
        reserve(g)
        reserve(g)
        assert g.run_used() == 2 * worst
        before = len(read_spend(g.path))
        with pytest.raises(BudgetRefused, match="run cap"):
            reserve(g)
        assert len(read_spend(g.path)) == before  # nothing was written
    with guard(tmp_path / "b", run_cap_usd="1") as g:
        g.run_cap = worst
        reserve(g)  # exactly at the cap is allowed
    with guard(tmp_path / "c", run_cap_usd="1") as g:
        g.run_cap = worst - 1
        with pytest.raises(BudgetRefused):
            reserve(g)


def test_reservation_is_durable_before_the_call(tmp_path):
    with guard(tmp_path) as g:
        r = reserve(g)
        on_disk = read_spend(tmp_path / "spend.jsonl")  # what a crash right now would leave
        assert [x["kind"] for x in on_disk] == ["reserve"] and on_disk[0]["worst_case_pusd"] == r.worst_case_pusd
        g.settle(r, {"input_tokens": 10, "output_tokens": 5})


def test_settle_math_against_hand_computed_usage(tmp_path):
    usage = {"input_tokens": 120, "cache_creation_input_tokens": 600, "cache_read_input_tokens": 5000, "output_tokens": 80,
             "cache_creation_5m": 400, "cache_creation_1h": 200}  # fmt: skip
    with guard(tmp_path) as g:
        r = reserve(g, upper=6000, max_tokens=100)
        actual = g.settle(r, usage)
        assert actual == 120 * 100000 + 400 * 125000 + 200 * 200000 + 5000 * 10000 + 80 * 500000 == 192_000_000
        assert g.spent == actual and g.outstanding == 0
        settle = g.records[-1]
        assert settle["kind"] == "settle" and settle["ref"] == r.seq and settle["usage"] == usage and settle["actual_pusd"] == actual


def test_void_costs_nothing(tmp_path):
    with guard(tmp_path) as g:
        r = reserve(g)
        assert g.outstanding == r.worst_case_pusd
        g.void(r, "http_400")
        assert (g.spent, g.outstanding) == (0, 0) and g.records[-1]["outcome"] == "http_400"


def test_assume_spent_counts_the_worst_case(tmp_path):
    with guard(tmp_path) as g:
        r = reserve(g)
        g.assume_spent(r, "timeout")
        assert g.spent == r.worst_case_pusd and g.outstanding == 0


def test_a_crash_mid_call_blocks_the_next_start_until_resolved(tmp_path):
    g = guard(tmp_path)
    r = reserve(g)
    g.close()  # the process died before the response
    with pytest.raises(OpenReservation, match="resolve"):
        guard(tmp_path)
    with guard(tmp_path, resolve=True) as fixer:
        assert fixer.resolve_assume_spent() == [r.seq]
    with guard(tmp_path) as again:
        assert again.spent == r.worst_case_pusd and again.records[-1]["outcome"] == "resolved_after_crash"


def test_chain_tamper_is_detected(tmp_path):
    with guard(tmp_path) as g:
        g.settle(reserve(g), {"input_tokens": 10, "output_tokens": 5})
    path = tmp_path / "spend.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    cheaper = lines[0].replace(f'"worst_case_pusd":{first["worst_case_pusd"]}', '"worst_case_pusd":1')
    assert cheaper != lines[0]
    for bad in (
        [cheaper, lines[1]],  # an edited amount
        [lines[1]],  # a missing record
        lines + ["{not json"],
        [lines[0].replace('"kind":"reserve"', '"kind":"refund"')],
    ):
        path.write_text("\n".join(bad) + "\n")
        with pytest.raises(SpendLedgerCorrupt):
            guard(tmp_path)


def test_lock_contention(tmp_path):
    with guard(tmp_path):
        with pytest.raises(SpendLocked):
            guard(tmp_path)
        code = (
            "import fcntl, sys\n"
            f"f = open({str(tmp_path / 'lock')!r}, 'a+')\n"
            "try:\n    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\nexcept BlockingIOError:\n    sys.exit(3)\n"
        )
        assert subprocess.run([sys.executable, "-c", code], timeout=30).returncode == 3  # a second process too
    with guard(tmp_path):  # released on close
        pass


def test_run_cap_is_clamped_to_the_global_remaining(tmp_path):
    with guard(tmp_path, run_id="earlier") as g:
        r = reserve(g)
        g.settle(r, {"input_tokens": 0, "output_tokens": 0})
    with guard(tmp_path, run_id="big", global_cap_usd="1") as g:
        r = g.reserve(model=HAIKU, request_sha256="cd" * 32, prompt_tokens_upper=80_000, max_tokens=1024)
        g.settle(r, {"input_tokens": 80_000, "output_tokens": 1024})  # $0.008512
        spent_before = g.spent
    remaining = usd_to_pusd("1") - spent_before
    with pytest.raises(BudgetRefused, match="left under the global cap"):
        guard(tmp_path, run_id="next", global_cap_usd="1", run_cap_usd="1")
    with guard(tmp_path, run_id="next", global_cap_usd="1") as g:
        assert g.run_cap == remaining  # no --budget: the run may use what remains
    with guard(tmp_path, run_id="next", global_cap_usd="1", run_cap_usd="0.5") as g:
        assert g.run_cap == usd_to_pusd("0.5") and g.run_used() == 0


def test_hard_max_cannot_be_exceeded(tmp_path):
    assert HARD_MAX_USD == 150 and HARD_MAX_PUSD == 150 * PICO_PER_USD
    with guard(tmp_path, global_cap_usd="200") as g:  # clamped to the hard max
        assert g.global_cap == HARD_MAX_PUSD and g.status()["global_cap_pusd"] == HARD_MAX_PUSD
    with pytest.raises(BudgetRefused):
        guard(tmp_path, run_cap_usd="150.000000000001")
    for bad in ("-1", "1e3", "ten", "0.0000000000001"):
        with pytest.raises(ValueError):
            usd_to_pusd(bad)


def test_actual_above_worst_halts_the_run(tmp_path):
    with guard(tmp_path) as g:
        r = reserve(g, upper=100, max_tokens=10)
        with pytest.raises(GuardInvariantBroken):
            g.settle(r, {"input_tokens": 5000, "output_tokens": 10})
        assert g.records[-1]["kind"] == "settle"  # recorded before halting
        with pytest.raises(GuardInvariantBroken):
            reserve(g)


def test_the_long_context_tier_is_never_entered(tmp_path):
    with guard(tmp_path) as g:
        ok = reserve(g, upper=PROMPT_CEILING, max_tokens=1024)
        assert ok.worst_case_pusd == PROMPT_CEILING * 200000 + 1024 * 500000  # base-tier prices
        g.void(ok, "test")
        with pytest.raises(BudgetRefused, match="ceiling"):
            reserve(g, upper=PROMPT_CEILING + 1, max_tokens=1024)
        assert prompt_upper_from_count(85_653) == PROMPT_CEILING < prompt_upper_from_count(85_654)  # a large count is refused, not priced


def test_status_and_totals(tmp_path):
    with guard(tmp_path) as g:
        g.settle(reserve(g), {"input_tokens": 1000, "output_tokens": 0})
        open_r = reserve(g)
        status = g.status()
        assert status["spent_pusd"] == 1000 * 100000 and status["outstanding_pusd"] == open_r.worst_case_pusd
        assert status["per_run_pusd"] == {"run1": 1000 * 100000} and status["per_model_pusd"] == {HAIKU: 1000 * 100000}
        assert "remaining" in render_status(status)
        g.void(open_r, "test")
    assert totals(read_spend(tmp_path / "spend.jsonl")).open == ()
    raw = (tmp_path / "spend.jsonl").read_text()
    assert all(set(json.loads(line)) == {"seq", "ts", "kind", "run_id", "model", "request_sha256", "prompt_tokens_upper", "max_tokens",
               "worst_case_pusd", "actual_pusd", "usage", "outcome", "ref", "prev_hash", "hash"} for line in raw.splitlines())  # fmt: skip


@settings(max_examples=200, deadline=None)
@given(count=st.integers(0, 80_000), max_tokens=st.integers(1, 4096), data=st.data())
def test_actual_never_exceeds_worst_within_the_reserved_bounds(count, max_tokens, data):
    upper = prompt_upper_from_count(count)
    if upper > PROMPT_CEILING:
        return
    model = data.draw(st.sampled_from([HAIKU, "claude-sonnet-5-5"]))
    prompt = data.draw(st.integers(0, upper))
    read = data.draw(st.integers(0, prompt))
    write = data.draw(st.integers(0, prompt - read))
    hour = data.draw(st.integers(0, write))
    usage = {"input_tokens": prompt - read - write, "cache_read_input_tokens": read, "cache_creation_input_tokens": write,
             "output_tokens": data.draw(st.integers(0, max_tokens)), "cache_creation_5m": write - hour, "cache_creation_1h": hour}  # fmt: skip
    assert TABLE.cost_pusd(model, usage) <= TABLE.worst_case_pusd(model, upper, max_tokens)


def test_every_entry_point_defaults_to_the_keys_hard_max(tmp_path):
    """BUILD-KEY: no entry point lowers the cap by default (global_cap_usd None), so each guard's
    cap is its key's: aperture $150; research $200 less its $20 reserve."""
    import inspect
    from context_ledger.eval.runner import offline_client
    from context_ledger.live.entry import open_live
    from context_ledger.live.spend import KEY_LEDGERS
    assert HARD_MAX_USD == 150
    for fn in (open_live, offline_client):
        assert inspect.signature(fn).parameters["global_cap_usd"].default is None
    with guard(tmp_path / "a") as g:
        assert g.global_cap == HARD_MAX_PUSD and g.key is KEY_LEDGERS["aperture"]
    with guard(tmp_path / "r", key=KEY_LEDGERS["research"]) as g:
        assert g.global_cap == 180 * PICO_PER_USD
