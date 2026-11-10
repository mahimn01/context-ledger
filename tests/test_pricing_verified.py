"""A.1: the verified 2026-10-31 price table, the Haiku long-context tier and worst-case bounds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from context_ledger.ids import canonical_json
from context_ledger.pricing import VERIFIED_PRICING_PATH, LongContextPrice, ModelPrice, PricingError, PricingTable

OLD = Path(__file__).resolve().parents[1] / "pricing" / "anthropic-2026-10.json"
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"


@pytest.fixture
def table() -> PricingTable:
    return PricingTable.load(VERIFIED_PRICING_PATH, live=True)


def test_verified_table_contents(table):
    assert table.verified and table.as_of == "2026-10-31" and len(table.sources) == 6
    assert all(url.startswith("https://platform.claude.com/docs/en/") for url in table.sources)
    haiku, sonnet = table.price(HAIKU), table.price(SONNET)
    assert (haiku.input, haiku.cache_write_5m, haiku.cache_write_1h, haiku.cache_read, haiku.output) == (100000, 125000, 200000, 10000, 500000)
    assert haiku.long_context == LongContextPrice(100000, 500000, 2500000, 625000, 1000000, 50000)
    assert (sonnet.input, sonnet.cache_write_5m, sonnet.cache_write_1h, sonnet.cache_read, sonnet.output) == (2000000, 2500000, 4000000, 100000, 10000000)
    assert sonnet.long_context is None
    assert haiku.min_cache_tokens == sonnet.min_cache_tokens == 512
    assert PricingTable.from_dict(json.loads(canonical_json(table.to_dict()))) == table
    assert table == PricingTable.verified_default()


def test_old_table_still_loads_and_hashes_as_before():
    old = PricingTable.load(OLD)
    assert old.verified is False and old.sources == ()
    assert "sources" not in old.to_dict() and all("long_context" not in m for m in old.to_dict()["models"].values())
    # The M2 demo and its tests pin this hash through the ledger's pricing_set event.
    assert old.pricing_hash() == PricingTable.from_dict(json.loads(OLD.read_text())).pricing_hash()


def test_live_refuses_an_unverified_table():
    with pytest.raises(PricingError, match="not verified"):
        PricingTable.load(OLD, live=True)


def usage(prompt: int, output: int = 0, read: int = 0, write: int = 0) -> dict:
    return {"input_tokens": prompt - read - write, "cache_read_input_tokens": read, "cache_creation_input_tokens": write, "output_tokens": output}


def test_tier_boundary_at_exactly_100000_and_100001(table):
    at = table.cost_pusd(HAIKU, usage(100_000, 1000, read=40_000, write=10_000))
    assert at == 50_000 * 100000 + 10_000 * 125000 + 40_000 * 10000 + 1000 * 500000
    above = table.cost_pusd(HAIKU, usage(100_001, 1000, read=40_000, write=10_000))
    # The whole request, output included, moves to the long-context tier.
    assert above == 50_001 * 500000 + 10_000 * 625000 + 40_000 * 50000 + 1000 * 2500000
    assert table.cost_pusd(SONNET, usage(100_001, 1)) == 100_001 * 2000000 + 10000000  # Sonnet has one tier


def test_one_hour_writes_use_the_long_tier_too(table):
    split = {**usage(150_000, 10, write=20_000), "cache_creation_5m": 5_000, "cache_creation_1h": 15_000}
    assert table.cost_pusd(HAIKU, split) == 130_000 * 500000 + 5_000 * 625000 + 15_000 * 1000000 + 10 * 2500000


def test_worst_case(table):
    assert table.worst_case_pusd(HAIKU, 90_000, 1024) == 90_000 * 200000 + 1024 * 500000
    assert table.worst_case_pusd(HAIKU, 100_000, 1024) == 100_000 * 200000 + 1024 * 500000
    assert table.worst_case_pusd(HAIKU, 100_001, 1024) == 100_001 * 1000000 + 1024 * 2500000
    assert table.worst_case_pusd(SONNET, 1000, 10) == 1000 * 4000000 + 10 * 10000000
    with pytest.raises(PricingError):
        table.worst_case_pusd(HAIKU, -1, 10)


def test_tier_limit_still_refuses_above_it():
    price = ModelPrice("m", 1, 1, 1, 1, 1, 0, 50, LongContextPrice(10, 2, 2, 2, 2, 2))
    table = PricingTable("t", "2026-10-31", "test", True, {"m": price})
    assert table.cost_pusd("m", usage(11)) == 22 and table.cost_pusd("m", usage(10)) == 10
    with pytest.raises(PricingError, match="above the 50-token price tier"):
        table.cost_pusd("m", usage(51))
    with pytest.raises(PricingError, match="above the long-context threshold"):
        ModelPrice("m", 1, 1, 1, 1, 1, 0, 10, LongContextPrice(10, 2, 2, 2, 2, 2))


def test_strict_long_context_and_sources(table):
    d = table.to_dict()
    for bad in (
        {**d, "sources": "one url"},
        {**d, "sources": [""]},
        {**d, "models": {HAIKU: {**d["models"][HAIKU], "long_context": {"threshold_prompt_tokens": 1}}}},
        {**d, "models": {HAIKU: {**d["models"][HAIKU], "long_context": {**d["models"][HAIKU]["long_context"], "input": -1}}}},
        {**d, "models": {HAIKU: {**d["models"][HAIKU], "long_context": {**d["models"][HAIKU]["long_context"], "threshold_prompt_tokens": 0}}}},
    ):
        with pytest.raises((PricingError, TypeError)):
            PricingTable.from_dict(bad)


@settings(max_examples=200, deadline=None)
@given(
    upper=st.integers(0, 200_000),
    max_tokens=st.integers(0, 4096),
    data=st.data(),
)
def test_actual_never_exceeds_the_worst_case(upper, max_tokens, data):
    table = PricingTable.load(VERIFIED_PRICING_PATH, live=True)
    model = data.draw(st.sampled_from([HAIKU, SONNET]))
    prompt = data.draw(st.integers(0, upper))
    read = data.draw(st.integers(0, prompt))
    write = data.draw(st.integers(0, prompt - read))
    five = data.draw(st.integers(0, write))
    u = {**usage(prompt, data.draw(st.integers(0, max_tokens)), read, write), "cache_creation_5m": five, "cache_creation_1h": write - five}
    assert table.cost_pusd(model, u) <= table.worst_case_pusd(model, upper, max_tokens)
