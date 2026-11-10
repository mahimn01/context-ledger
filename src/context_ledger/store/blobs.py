"""Content-addressed blob stores: S[h] is bytes whose sha256 hex is h."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..ids import sha256_hex
from .ledger import open_db

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs (hash TEXT PRIMARY KEY, data BLOB);
CREATE TRIGGER IF NOT EXISTS blobs_no_update BEFORE UPDATE ON blobs
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS blobs_no_delete BEFORE DELETE ON blobs
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
"""


class IntegrityError(Exception):
    """A stored blob no longer hashes to its key."""


class BlobStore(Protocol):
    def put(self, data: bytes) -> str:
        """Store `data` (idempotent) and return its sha256 hex."""
        ...

    def get(self, h: str) -> bytes:
        """Return the bytes for `h`; KeyError if absent, IntegrityError if they re-hash differently."""
        ...

    def has(self, h: str) -> bool:
        """Return True if a blob is stored under `h`."""
        ...


def _verified(h: str, data: bytes | None) -> bytes:
    if data is None:
        raise KeyError(h)
    if sha256_hex(data) != h:
        raise IntegrityError(f"blob {h} does not re-hash to its key")
    return data


class SqliteBlobStore:
    """Blob table in the ledger's SQLite file, append-only like the events table."""

    def __init__(self, path: str | Path) -> None:
        self._conn = open_db(path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def put(self, data: bytes) -> str:
        h = sha256_hex(data)
        self._conn.execute("INSERT OR IGNORE INTO blobs (hash, data) VALUES (?, ?)", (h, data))
        self._conn.commit()
        return h

    def get(self, h: str) -> bytes:
        row = self._conn.execute("SELECT data FROM blobs WHERE hash = ?", (h,)).fetchone()
        return _verified(h, None if row is None else bytes(row[0]))

    def has(self, h: str) -> bool:
        return self._conn.execute("SELECT 1 FROM blobs WHERE hash = ?", (h,)).fetchone() is not None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteBlobStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class MemoryBlobStore:
    """Dict-backed store. `data` is public so tests can corrupt entries on a `copy()`."""

    def __init__(self, data: dict[str, bytes] | None = None) -> None:
        self.data: dict[str, bytes] = dict(data or {})

    def put(self, data: bytes) -> str:
        h = sha256_hex(data)
        self.data.setdefault(h, bytes(data))
        return h

    def get(self, h: str) -> bytes:
        return _verified(h, self.data.get(h))

    def has(self, h: str) -> bool:
        return h in self.data

    def copy(self) -> MemoryBlobStore:
        return MemoryBlobStore(self.data)
