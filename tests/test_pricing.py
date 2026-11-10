from __future__ import annotations

import json
from pathlib import Path

import pytest

from context_ledger.fold import BackstopPolicyV1
from context_ledger.pricing import ModelPrice, PricingError, PricingTable, usd
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.window import replay

PRICING = Path(__file__).resolve().parents[1] / "pricing" / "anthropic-2026-10.json"


@pytest.fixture
def table() -> PricingTable:
    return PricingTable.load(PRICING)


def test_shipped_table(table):
    assert table.verified is False and "verify against https://platform.claude.com" in table.source
    assert sorted(table.models) == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert table.price("claude-haiku-5-5") == ModelPrice("claude-haiku-5-5", 100000, 500000, 125000, 200000, 10000, 4096, 100000)
    assert table.price("claude-sonnet-5-5") == ModelPrice("claude-sonnet-5-5", 2000000, 10000000, 2500000, 4000000, 100000, 1024, None)
    assert PricingTable.from_dict(json.loads(json.dumps(table.to_dict()))) == table
    assert len(table.pricing_hash()) == 64


def test_cost_is_exact_pico_usd(table):
    usage = {"input_tokens": 1000, "cache_creation_input_tokens": 3000, "cache_read_input_tokens": 5000, "output_tokens": 200}
    # 1000*0.10 + 3000*0.125 + 5000*0.01 + 200*0.50 dollars per MTok, in pico-USD:
    assert table.cost_pusd("claude-haiku-5-5", usage) == 1000 * 100000 + 3000 * 125000 + 5000 * 10000 + 200 * 500000 == 625_000_000
    assert usd(625_000_000) == "$0.000625"
    assert table.cost_pusd("claude-haiku-5-5", usage, ttl="1h") == 625_000_000 + 3000 * (200000 - 125000)
    split = {**usage, "cache_creation_5m": 1000, "cache_creation_1h": 2000}
    assert table.cost_pusd("claude-haiku-5-5", split, ttl="1h") == 1000 * 100000 + 1000 * 125000 + 2000 * 200000 + 5000 * 10000 + 200 * 500000
    partial = {**usage, "cache_creation_5m": 1000}  # the rest of the writes at the request ttl
    assert table.cost_pusd("claude-haiku-5-5", partial, ttl="1h") == 1000 * 100000 + 1000 * 125000 + 2000 * 200000 + 5000 * 10000 + 200 * 500000
    assert table.cost_pusd("claude-sonnet-5-5", {"input_tokens": 1, "output_tokens": 1}) == 12_000_000


def test_cost_errors(table):
    with pytest.raises(PricingError):
        table.cost_pusd("claude-opus-9", {"input_tokens": 1})
    with pytest.raises(PricingError):  # Haiku prices cover prompts up to 100k tokens only
        table.cost_pusd("claude-haiku-5-5", {"input_tokens": 60_000, "cache_read_input_tokens": 40_001})
    assert table.cost_pusd("claude-haiku-5-5", {"input_tokens": 60_000, "cache_read_input_tokens": 40_000}) > 0
    for bad in ({"input_tokens": -1}, {"input_tokens": 1.0}, {"input_tokens": True}):
        with pytest.raises(PricingError):
            table.cost_pusd("claude-haiku-5-5", bad)
    with pytest.raises(PricingError):
        table.cost_pusd("claude-haiku-5-5", {"cache_creation_input_tokens": 1, "cache_creation_5m": 2})
    with pytest.raises(PricingError):
        table.cost_pusd("claude-haiku-5-5", {"cache_creation_input_tokens": 1}, ttl="10m")


def test_strict_validation(table, tmp_path):
    d = table.to_dict()
    haiku = d["models"]["claude-haiku-5-5"]
    bad_tables = [
        {**d, "extra": 1},
        {k: v for k, v in d.items() if k != "as_of"},
        {**d, "verified": "no"},
        {**d, "models": {}},
        {**d, "models": {"claude-haiku-5-5": {**haiku, "input": -5}}},
        {**d, "models": {"claude-haiku-5-5": {**haiku, "bonus": 1}}},
        {**d, "models": {"claude-haiku-5-5": {k: v for k, v in haiku.items() if k != "cache_read"}}},
        {**d, "models": {"claude-haiku-5-5": {**haiku, "cache_read": True}}},
    ]
    for bad in bad_tables:
        with pytest.raises((PricingError, TypeError)):
            PricingTable.from_dict(bad)
    floaty = tmp_path / "float.json"
    floaty.write_text(json.dumps(d).replace('"input": 100000', '"input": 0.1'))
    with pytest.raises(PricingError):
        PricingTable.load(floaty)


def test_hash_tracks_content(table):
    d = table.to_dict()
    d["models"]["claude-haiku-5-5"]["output"] += 1
    assert PricingTable.from_dict(d).pricing_hash() != table.pricing_hash()


def test_session_records_pricing_once_and_replay_keeps_it(table):
    ledger, blobs = Ledger(":memory:"), MemoryBlobStore()
    s = Session(ledger, blobs, 1000, BackstopPolicyV1(budget_tokens=1000), pricing=table)
    assert [e.type for e in s.events] == ["pricing_set"]
    assert s.pricing == table and replay(s.events).pricing_hash == table.pricing_hash()
    again = Session(ledger, blobs, 1000, BackstopPolicyV1(budget_tokens=1000), pricing=table)
    assert len(again.events) == 1
    d = table.to_dict()
    d["models"]["claude-haiku-5-5"]["cache_read"] = 20000
    changed = PricingTable.from_dict(d)
    again.set_pricing(changed)
    assert [e.type for e in again.events] == ["pricing_set", "pricing_set"]
    assert replay(again.events).pricing == changed.to_dict() and again.pricing == changed
