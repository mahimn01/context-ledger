"""Append-only, hash-chained event ledger on SQLite.

Chain rule: hash_i = sha256(bytes.fromhex(hash_{i-1}) + canonical_json({"seq": i, "type": t,
"payload": p})), with hash_0 = "0" * 64. SQLite triggers reject every UPDATE and DELETE.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..ids import GENESIS_HASH, canonical_json, sha256_hex
from ..model import Event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
"""


def chain_hash(prev_hash: str, seq: int, type_: str, payload: dict) -> str:
    """Return the chain hash of event `seq` given the previous event's hash."""
    body = canonical_json({"seq": seq, "type": type_, "payload": payload})
    return sha256_hex(bytes.fromhex(prev_hash) + body)


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection; file databases use WAL journaling, ":memory:" does not."""
    conn = sqlite3.connect(str(path))
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    return conn


class Ledger:
    """The event log. `append` is the only write path."""

    def __init__(self, path: str | Path) -> None:
        self._conn = open_db(path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        row = self._conn.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        self._len, self._head = (row[0], row[1]) if row else (0, GENESIS_HASH)

    def append(self, type_: str, payload: dict) -> Event:
        """Append one event, computing its seq and chain hash, and return it.

        Raises TypeError if `payload` is not canonical-JSON data (e.g. contains a float)."""
        seq = self._len + 1
        prev = self._head
        digest = chain_hash(prev, seq, type_, payload)
        self._conn.execute(
            "INSERT INTO events (seq, type, payload, prev_hash, hash) VALUES (?, ?, ?, ?, ?)",
            (seq, type_, canonical_json(payload).decode("utf-8"), prev, digest),
        )
        self._conn.commit()
        self._len, self._head = seq, digest
        return Event(seq, type_, payload, prev, digest)

    def events(self) -> list[Event]:
        """Return all events in seq order, payloads decoded from JSON."""
        rows = self._conn.execute("SELECT seq, type, payload, prev_hash, hash FROM events ORDER BY seq").fetchall()
        return [Event(seq, type_, json.loads(payload), prev, digest) for seq, type_, payload, prev, digest in rows]

    def head(self) -> str:
        """Return the hash of the last event, or the genesis hash if the ledger is empty."""
        return self._head

    def __len__(self) -> int:
        return self._len

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def verify_chain(events: Sequence[Event]) -> list[int]:
    """Return the seqs of events that break the chain (pure; never raises on bad data).

    An event is reported if its seq is not its 1-based position, its prev_hash differs from
    the previous event's hash (genesis for the first), or its hash does not recompute."""
    bad: list[int] = []
    expected_prev = GENESIS_HASH
    for position, event in enumerate(events, start=1):
        seq = event.seq if isinstance(getattr(event, "seq", None), int) else position
        try:
            ok = (
                event.seq == position
                and event.prev_hash == expected_prev
                and chain_hash(event.prev_hash, event.seq, event.type, event.payload) == event.hash
            )
        except Exception:
            ok = False
        if not ok:
            bad.append(seq)
        expected_prev = getattr(event, "hash", None)
    return bad


def rechain(events: Iterable[Event]) -> list[Event]:
    """Return copies of `events` renumbered 1..n with prev_hash and hash recomputed.

    Used to build ledgers in tests and the demo (e.g. to show that a re-chained tampered
    ledger still fails the content invariants even though P11 holds)."""
    out: list[Event] = []
    prev = GENESIS_HASH
    for seq, event in enumerate(events, start=1):
        digest = chain_hash(prev, seq, event.type, event.payload)
        out.append(Event(seq, event.type, event.payload, prev, digest))
        prev = digest
    return out
