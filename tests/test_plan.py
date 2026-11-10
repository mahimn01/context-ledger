"""E: the cost planner reproduces the section E math exactly from the default parameters."""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

from context_ledger.eval.plan import HAIKU, PRIOR, SONNET, CallShape, call_pusd, episode_pusd, latest_live_smoke, measured_params, plan, render_plan
from context_ledger.pricing import PICO_PER_USD, PricingTable

TABLE = PricingTable.verified_default()
USD = Fraction(PICO_PER_USD)


def dollars(pusd, places: int) -> Decimal:
    value = Fraction(pusd) / USD
    return (Decimal(value.numerator) / Decimal(value.denominator)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def test_haiku_numbers_exactly():
    agent = call_pusd(TABLE, HAIKU, PRIOR[HAIKU].agent)
    # 8.5k x 0.01 + 1.5k x 0.125 + 0.1k x 0.10 = $0.0002825 input side, + 200 x 0.50 / 1M = $0.0001 output.
    assert agent == 8500 * 10000 + 1500 * 125000 + 100 * 100000 + 200 * 500000 == 382_500_000
    assert dollars(agent, 5) == Decimal("0.00038")
    ep = episode_pusd(TABLE, HAIKU, PRIOR[HAIKU])
    assert ep["L"] == 48 * agent == 18_360_000_000 and dollars(ep["L"], 3) == Decimal("0.018")
    summary = call_pusd(TABLE, HAIKU, PRIOR[HAIKU].summary)
    assert summary == 13600 * 100000 + 1500 * 500000 == 2_110_000_000  # $0.00136 + $0.00075
    assert ep["B4"] == ep["L"] + 4 * summary == 26_800_000_000 and dollars(ep["B4"], 3) == Decimal("0.027")
    assert ep["B3"] == 8 * (16000 * 100000 + 200 * 500000) == 13_600_000_000 and dollars(ep["B3"], 4) == Decimal("0.0136")


def test_grid_and_the_briefs_rounding():
    p = plan(TABLE, window=16_000)  # the brief's prior was stated at W = 16,000
    assert p["haiku_grid"] == 36 * (18_360_000_000 + 26_800_000_000 + 13_600_000_000) == 2_115_360_000_000
    assert p["haiku_grid_2x"] == 4_230_720_000_000
    # The brief's "about $2.11" and "$4.22" come from the per-episode figures rounded to
    # $0.018, $0.027 and $0.0136 first; the exact grid is $2.11536 (2x: $4.23072).
    rounded = 36 * (Fraction("0.018") + Fraction("0.027") + Fraction("0.0136"))
    assert rounded == Fraction("2.1096") and dollars(rounded * USD, 2) == Decimal("2.11") and dollars(2 * rounded * USD, 2) == Decimal("4.22")
    assert abs(Fraction(p["haiku_grid"]) / USD - rounded) / rounded < Fraction(1, 200)


def test_sonnet_numbers_exactly():
    p = plan(TABLE, window=16_000)  # the brief's prior was stated at W = 16,000
    assert p["sonnet_call"]["agent"] == 8500 * 100000 + 1500 * 2500000 + 100 * 2000000 + 400 * 10000000 == 8_800_000_000
    assert p["sonnet_episode"]["L"] == 48 * 8_800_000_000 == 422_400_000_000 and dollars(p["sonnet_episode"]["L"], 2) == Decimal("0.42")
    assert p["sonnet_episode"]["B4"] == 422_400_000_000 + 4 * (13600 * 2000000 + 1500 * 10000000) == 591_200_000_000
    assert dollars(p["sonnet_episode"]["B4"], 2) == Decimal("0.59")
    assert p["sonnet_design_subset"] == 24 * (422_400_000_000 + 591_200_000_000) and dollars(p["sonnet_design_subset"], 0) == Decimal("24")
    assert p["sonnet_design_subset"] < 33 * PICO_PER_USD  # the full design subset fits its $33 sub-cap


def test_replanned_grid():
    p = plan(TABLE, window=16_000)  # the brief's prior was stated at W = 16,000
    items = p["items"]
    assert items["smoke"] == 3 * (Fraction(18_360_000_000 + 26_800_000_000 + 13_600_000_000, 8)) + Fraction(18_360_000_000 + 26_800_000_000, 8)
    assert dollars(items["smoke"], 2) == Decimal("0.03") and items["calibration"] == 0
    assert p["episodes"] == {"haiku_grid": 12, "sonnet_subset": 12} and p["shrunk"] == []
    assert items["sonnet_subset"] == p["sonnet_design_subset"] == 24 * (422_400_000_000 + 591_200_000_000)  # $24.3264
    assert dollars(items["haiku_grid"], 1) == Decimal("2.1")
    assert p["total"] == sum(items.values()) and dollars(p["total"], 2) == Decimal("26.47")
    assert p["sub_caps"] == {"smoke": 750_000_000_000, "calibration": None, "haiku_grid": 6_000_000_000_000,
                             "sonnet_subset": 33_000_000_000_000, "reserve": 10_250_000_000_000}  # fmt: skip
    planned = sum(v for k, v in p["sub_caps"].items() if v and k != "reserve")
    assert planned == Fraction("39.75") * PICO_PER_USD <= 40 * PICO_PER_USD
    assert planned + p["sub_caps"]["reserve"] == 50 * PICO_PER_USD and p["sub_caps"]["reserve"] >= 10 * PICO_PER_USD  # >= 20% reserve
    assert p["fits"] and p["remaining"] == 50 * PICO_PER_USD
    assert not plan(TABLE, remaining_pusd=30 * PICO_PER_USD, window=16_000)["fits"]  # 1.2 x $26.47 > $30
    text = render_plan(p)
    for needle in ("$2.115360", "$0.0003825", "$0.018360", "$0.026800", "$0.013600", "$24.326400", "$50 hard cap", "Verdict: FITS"):
        assert needle in text, needle


def smoke_run(tmp_path, out_tokens: int):
    run = tmp_path / "smoke-live"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"run_id": "smoke-live", "live": True, "smoke": True, "model": HAIKU}))
    rows = []
    for arm in ("B4", "L"):
        for q in range(4):
            for _ in range(3):
                rows.append({"arm": arm, "qid": f"e01q{q}", "kind": "answer", "usage": {"input_tokens": 50, "cache_read_input_tokens": 9000,
                             "cache_creation_input_tokens": 1000, "output_tokens": out_tokens}})  # fmt: skip
    rows.append({"arm": "B4", "qid": "e01q3", "kind": "summary", "usage": {"input_tokens": 12000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 900}})
    rows += [{"arm": "B3", "qid": f"e01q{q}", "kind": "answer", "usage": {"input_tokens": 15000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 150}} for q in range(3)]
    (run / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run


def test_measured_smoke_replaces_the_prior(tmp_path):
    run = smoke_run(tmp_path, 300)
    assert latest_live_smoke(tmp_path) == run
    params = measured_params(run)
    m = params[HAIKU]
    assert m.agent == CallShape(9000, 1000, 50, 300) and m.calls_per_question == 3 and m.rag == CallShape(0, 0, 15000, 150)
    assert m.summaries_per_episode == Fraction(1, 4) * 8 and params[SONNET] == PRIOR[SONNET] and "measured" in m.source
    assert plan(TABLE, params)["haiku_grid"] != plan(TABLE)["haiku_grid"]
    heavy = {**PRIOR, SONNET: replace(PRIOR[SONNET], calls_per_question=Fraction(9))}
    hp = plan(TABLE, heavy)
    assert hp["shrunk"] == ["sonnet_subset"] and hp["episodes"]["sonnet_subset"] < 12
    assert hp["items"]["sonnet_subset"] * Fraction(6, 5) <= hp["sub_caps"]["sonnet_subset"]
    (tmp_path / "offline").mkdir()
    (tmp_path / "offline" / "run.json").write_text(json.dumps({"live": False, "smoke": True}))
    assert latest_live_smoke(tmp_path) == run  # offline runs never replace the prior
