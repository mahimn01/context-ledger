from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest

from context_ledger.ids import GENESIS_HASH, canonical_json, sha256_hex
from context_ledger.model import (
    Citation,
    Claim,
    Event,
    FoldRecord,
    Knowledge,
    PyAnchor,
    Segment,
    Source,
    Span,
    SummarySentence,
    TextAnchor,
)
from context_ledger.store import IntegrityError, Ledger, MemoryBlobStore, SqliteBlobStore, rechain, verify_chain

TEXT_ANCHOR = TextAnchor(start=2, end=5, exact="llo", prefix="he", suffix=" w")
PY_ANCHOR = PyAnchor(path="m.py", qualname="C.f", node_kind="function", ast_hash="ab" * 32, start=0, end=9, commit=None)
RECORDS = [
    Source("src_1", "file:///m.py", "v1", "a" * 64, "text/x-python"),
    TEXT_ANCHOR,
    PY_ANCHOR,
    Span("s" * 64, "src_1", "a" * 64, TEXT_ANCHOR, "b" * 64, 3),
    Span("t" * 64, "src_1", "a" * 64, PY_ANCHOR, "c" * 64, 9),
    Segment("seg_1", "tool", "span", "s" * 64, "verbatim", False, "b" * 64, None),
    Segment("seg_2", "system", "raw", "d" * 64, "verbatim", True, "d" * 64, None),
    SummarySentence("about f", ("s" * 64, "inference")),
    FoldRecord(
        "fold_1",
        1,
        ("seg_1",),
        (SummarySentence("about f", ("s" * 64,)),),
        "backstop",
        {"id": "backstop-v0", "budget_tokens": 600, "hard": "0.85"},
        12,
        "seg_f",
        "e" * 64,
    ),
    Citation("s" * 64, "llo"),
    Claim("claim_1", "call_1", "x is 3", (Citation("s" * 64, "x = 3"), Citation("inference")), "grounded", {"tier": 1}),
]


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: type(r).__name__)
def test_records_round_trip_through_canonical_json(record):
    decoded = json.loads(canonical_json(record.to_dict()))
    assert type(record).from_dict(decoded) == record


def test_knowledge_rank_order():
    order = [Knowledge.VERBATIM, Knowledge.SNIPPET, Knowledge.SUMMARY_ONLY, Knowledge.DROPPED, Knowledge.UNSEEN, Knowledge.STALE]
    assert [k.rank for k in order] == [5, 4, 3, 2, 1, 0]
    assert Knowledge("summary_only") is Knowledge.SUMMARY_ONLY


def test_append_and_read_back(ledger):
    assert len(ledger) == 0 and ledger.head() == GENESIS_HASH
    e1 = ledger.append("source_registered", {"source": RECORDS[0].to_dict()})
    e2 = ledger.append("segment_appended", {"segment": RECORDS[5].to_dict()})
    assert (e1.seq, e2.seq) == (1, 2)
    assert e1.prev_hash == GENESIS_HASH and e2.prev_hash == e1.hash
    assert ledger.head() == e2.hash and len(ledger) == 2
    events = ledger.events()
    assert events == [e1, e2]
    assert Segment.from_dict(events[1].payload["segment"]) == RECORDS[5]
    assert verify_chain(events) == []


def test_chain_hash_formula(ledger):
    event = ledger.append("evict", {"segment_ids": ["seg_1"], "reason": "test"})
    body = canonical_json({"seq": 1, "type": "evict", "payload": {"segment_ids": ["seg_1"], "reason": "test"}})
    assert event.hash == sha256_hex(bytes.fromhex(GENESIS_HASH) + body)


def test_append_rejects_floats(ledger):
    with pytest.raises(TypeError):
        ledger.append("call", {"x": 1.5})
    assert len(ledger) == 0


def test_verify_chain_detects_edits_reorders_and_gaps(ledger):
    for i in range(4):
        ledger.append("evict", {"segment_ids": [f"s{i}"], "reason": "r"})
    events = ledger.events()
    edited = list(events)
    edited[1] = replace(edited[1], payload={"segment_ids": ["tampered"], "reason": "r"})
    assert verify_chain(edited) == [2]
    assert verify_chain(events[:1] + events[2:]) == [3, 4]  # gap: seq 3 sits at position 2
    assert verify_chain(list(reversed(events))) != []
    assert verify_chain(rechain(edited)) == []
    garbage = [Event(1, "evict", {"x": 1.5}, "not-hex", "h")]
    assert verify_chain(garbage) == [1]


@pytest.mark.parametrize("statement", ["UPDATE events SET type = 'x'", "DELETE FROM events"])
def test_events_table_is_append_only(ledger, statement):
    ledger.append("evict", {"segment_ids": [], "reason": "r"})
    with pytest.raises((sqlite3.IntegrityError, sqlite3.DatabaseError), match="append-only"):
        ledger._conn.execute(statement)
    assert len(ledger.events()) == 1


@pytest.mark.parametrize("statement", ["UPDATE blobs SET data = x'00'", "DELETE FROM blobs"])
def test_blobs_table_is_append_only(statement):
    with SqliteBlobStore(":memory:") as store:
        store.put(b"hello")
        with pytest.raises((sqlite3.IntegrityError, sqlite3.DatabaseError), match="append-only"):
            store._conn.execute(statement)
        assert store.get(sha256_hex(b"hello")) == b"hello"


def test_sqlite_blob_store_put_is_idempotent():
    with SqliteBlobStore(":memory:") as store:
        h1 = store.put(b"data")
        h2 = store.put(b"data")
        assert h1 == h2 == sha256_hex(b"data")
        assert store.has(h1) and not store.has("0" * 64)
        with pytest.raises(KeyError):
            store.get("0" * 64)


def test_memory_blob_store_detects_corruption():
    store = MemoryBlobStore()
    h = store.put(b"original")
    assert store.get(h) == b"original"
    corrupted = store.copy()
    corrupted.data[h] = b"tampered"
    with pytest.raises(IntegrityError):
        corrupted.get(h)
    assert store.get(h) == b"original"
    with pytest.raises(KeyError):
        store.get("f" * 64)


def test_wal_file_mode_and_reopen(tmp_path):
    db = tmp_path / "ledger.sqlite"
    with Ledger(db) as led, SqliteBlobStore(db) as store:
        mode = led._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        h = store.put(b"blob")
        e1 = led.append("evict", {"segment_ids": [], "reason": "a"})
    with Ledger(db) as led, SqliteBlobStore(db) as store:
        assert len(led) == 1 and led.head() == e1.hash
        e2 = led.append("evict", {"segment_ids": [], "reason": "b"})
        assert e2.prev_hash == e1.hash
        assert verify_chain(led.events()) == []
        assert store.get(h) == b"blob"
