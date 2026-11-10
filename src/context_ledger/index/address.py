"""AddressIndex: exact lookups over a built address index (build.py), at one rho level.

lookup(name) is EXACT: a qualified name (`bz2.BZ2File.readline`, `xml.dom.SYNTAX_ERR`) is found
only as itself; a bare name (`dahlia_mint`) by its last component, as the gate anchors a bare
target; a path (`Lib/bz2.py`) as that file. When nothing matches, the nearest names are listed,
each labelled a DIFFERENT identifier and why: a sibling under the longest prefix that exists,
within two edits (`INVALID_STATE_ERR` for `INALID_STATE_ERR`); the same last component in
another module; a long shared suffix or prefix of the last component.

Cost. Every search is a binary search over one sorted table of C-rho (rank = position), one
probe per step, counted in `visited`: an exact lookup is ceil(log2(N + 1)) probes; with no match,
one more search per prefix level of the name and one each in the last-component tables, so
O(d log N) for a name of d components. Rows read after a search (the matches, the siblings of
the longest prefix, the neighbours in the suffix and prefix orders) are counted in `scanned`
and capped (MAX_MATCHES, MAX_SIBLINGS, NEIGHBOURS), never proportional to N. The root's children
(every top-level module) are never scanned. Each probe is one SQLite rowid lookup.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .build import INDEX_VERSION, span_sha

MAX_MATCHES = 8
MAX_SIBLINGS = 2048
NEIGHBOURS = 4
MAX_EDITS = 2
OUTLINE_LIMIT = 60
_ADDRESS = re.compile(r"^\s*(?P<path>[^\s:#]+):(?P<start>\d+)(?:-(?P<end>\d+))?(?:#(?P<sha>[0-9a-fA-F]{4,64}))?\s*$")


class IndexError_(Exception):
    """The address index is missing, of another version or lacks the rho asked for."""


@dataclass(frozen=True)
class Address:
    """An exact, content-addressed place: a name's kind, path, 1-based inclusive lines and the
    sha256 of those lines; `key` is the qualified name ("" for a file or directory)."""

    key: str
    kind: str
    path: str
    start: int
    end: int
    sha: str
    summary: str = ""

    @property
    def ref(self) -> str:
        """path:start-end#sha8, what read() takes."""
        return f"{self.path}:{self.start}-{self.end}#{self.sha[:8]}"

    def line(self) -> str:
        shown = f"`{self.key}` " if self.key else ""
        return f"{shown}{self.kind} {self.ref}" + (f"  {self.summary}" if self.summary else "")

    def to_dict(self) -> dict:
        return {"key": self.key, "kind": self.kind, "path": self.path, "start": self.start, "end": self.end, "sha256": self.sha}


@dataclass(frozen=True)
class NearMiss:
    address: Address
    why: str  # "1 edit", "same name in another module", "shares the suffix `_STATE_ERR`", ...

    def line(self) -> str:
        return f"different identifier ({self.why}): {self.address.line()}"


@dataclass(frozen=True)
class Lookup:
    query: str
    exact: tuple[Address, ...]
    near: tuple[NearMiss, ...]
    visited: int
    scanned: int

    def render(self) -> str:
        if self.exact:
            head = f"`{self.query}`: {len(self.exact)} exact address{'es' if len(self.exact) > 1 else ''}"
            return "\n".join([head, *(f"  {a.line()}" for a in self.exact)])
        lines = [f"no exact address for `{self.query}` in this corpus."]
        if self.near:
            lines.append("nearest DIFFERENT identifiers (none of them is the one asked):")
            lines += [f"  {n.line()}" for n in self.near]
        else:
            lines.append("no similar names either.")
        return "\n".join(lines)


def edit_distance(a: str, b: str, cap: int = MAX_EDITS) -> int:
    """Optimal string alignment distance (insert, delete, substitute, swap adjacent), or cap + 1
    as soon as it must exceed `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


def _common_prefix(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def parse_address(text: str) -> tuple[str, int, int, str | None] | None:
    """(path, start, end, sha prefix or None) of "path:start-end#sha", "path:start-end" or
    "path:line"; None if `text` is not an address."""
    m = _ADDRESS.match(text)
    if m is None:
        return None
    start = int(m["start"])
    end = int(m["end"]) if m["end"] else start
    if start < 1 or end < start:
        return None
    return m["path"], start, end, (m["sha"].lower() if m["sha"] else None)


def clean_name(name: str) -> str:
    return name.strip().strip("`").strip().removesuffix("()")


class AddressIndex:
    """One rho level of an address index file, opened read-only."""

    def __init__(self, path: str | Path, rho: int) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise IndexError_(f"no address index at {self.path}; run `eval index build` first")
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        row = self.db.execute("SELECT value FROM meta WHERE key = 'meta'").fetchone()
        self.meta = json.loads(row[0]) if row else {}
        if self.meta.get("version") != INDEX_VERSION:
            raise IndexError_(f"{self.path} is not an {INDEX_VERSION} index")
        if str(rho) not in self.meta["levels"]:
            raise IndexError_(f"{self.path} has no level for rho={rho} (it has {sorted(self.meta['levels'])})")
        self.rho = rho
        self.level = self.meta["levels"][str(rho)]
        self.size = {t: self.db.execute(f"SELECT count(*) FROM {t}_{rho}").fetchone()[0] for t in ("sym", "leaf", "rleaf", "node")}
        self.visited = self.scanned = 0

    @property
    def root(self) -> str:
        return self.level["root"]

    def close(self) -> None:
        self.db.close()

    # binary search, one probe per step

    def _lower_bound(self, table: str, columns: str, key: tuple) -> int:
        lo, hi = 0, self.size[table]
        while lo < hi:
            mid = (lo + hi) // 2
            row = self.db.execute(f"SELECT {columns} FROM {table}_{self.rho} WHERE rank = ?", (mid,)).fetchone()
            self.visited += 1
            if tuple(row) < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _rows(self, table: str, columns: str, start: int, limit: int) -> list[tuple]:
        rows = self.db.execute(f"SELECT {columns} FROM {table}_{self.rho} WHERE rank >= ? ORDER BY rank LIMIT ?", (start, limit)).fetchall()
        self.scanned += len(rows)
        return rows

    _SYM = "parent, leaf, kind, path, start, end, sha, summary, rank"

    @staticmethod
    def _address(row: tuple) -> Address:
        parent, leaf, kind, path, start, end, sha, summary = row[:8]
        return Address(f"{parent}.{leaf}" if parent else leaf, kind, path, start, end, sha, summary)

    def _sym(self, rank: int) -> tuple:
        row = self.db.execute(f"SELECT {self._SYM} FROM sym_{self.rho} WHERE rank = ?", (rank,)).fetchone()
        self.scanned += 1
        return row

    def _equal(self, parent: str, leaf: str) -> list[tuple]:
        """Rows of sym with exactly (parent, leaf)."""
        start = self._lower_bound("sym", "parent, leaf", (parent, leaf))
        rows = self._rows("sym", self._SYM, start, MAX_MATCHES)
        return [r for r in rows if (r[0], r[1]) == (parent, leaf)]

    def _children(self, parent: str, limit: int = MAX_SIBLINGS) -> list[tuple]:
        start = self._lower_bound("sym", "parent", (parent,))
        out = []
        while len(out) < limit:
            rows = self._rows("sym", self._SYM, start + len(out), min(256, limit - len(out)))
            mine = [r for r in rows if r[0] == parent]
            out += mine
            if len(mine) < len(rows) or not rows:
                break
        return out

    def _leaf_rows(self, leaf: str) -> list[tuple]:
        start = self._lower_bound("leaf", "leaf", (leaf,))
        rows = self._rows("leaf", "leaf, sym", start, MAX_MATCHES)
        return [self._sym(sym) for name, sym in rows if name == leaf]

    def _neighbours(self, table: str, column: str, value: str) -> list[tuple[str, tuple]]:
        """(name, sym row) of the NEIGHBOURS rows on each side of `value`'s place in `table`."""
        at = self._lower_bound(table, column, (value,))
        rows = self._rows(table, f"{column}, sym", max(0, at - NEIGHBOURS), 2 * NEIGHBOURS)
        return [(name, self._sym(sym)) for name, sym in rows]

    # public

    def exact(self, name: str) -> list[Address]:
        """The exact addresses of `name` (module docstring); a path gives its file. A name
        without "/" is a dotted name first (`uuid.uuid8.c` is a parameter), a top-level file only
        if no name matches."""
        name = clean_name(name)
        if "/" in name:
            node = self.file(name)
            return [node] if node is not None else []
        found = self._names(name)
        if not found and name.endswith((".py", ".rst", ".txt", ".c", ".h", ".md")):
            node = self.file(name)
            return [node] if node is not None else []
        return found

    def _names(self, name: str) -> list[Address]:
        parent, _, leaf = name.rpartition(".")
        rows = self._equal(parent, leaf) if parent else self._equal("", leaf) + [r for r in self._leaf_rows(leaf) if r[0] != ""]
        seen, out = set(), []
        for r in rows:
            if r[2] != "param" or parent:
                a = self._address(r)
                if (a.key, a.path, a.start) not in seen:
                    seen.add((a.key, a.path, a.start))
                    out.append(a)
        return out[:MAX_MATCHES]

    def lookup(self, name: str, near: int = 5) -> Lookup:
        """Exact addresses of `name`, else up to `near` nearest different identifiers."""
        self.visited = self.scanned = 0
        query = clean_name(name)
        found = self.exact(query)
        misses: list[NearMiss] = [] if found else self._near(query, near)
        return Lookup(query, tuple(found), tuple(misses), self.visited, self.scanned)

    def _near(self, query: str, limit: int) -> list[NearMiss]:
        if "/" in query:
            return []
        parts = query.split(".")
        leaf = parts[-1]
        found: dict[str, NearMiss] = {}

        def add(row: tuple, why: str) -> None:
            a = self._address(row)
            if a.key != query and a.key not in found and a.kind != "param":
                found[a.key] = NearMiss(a, why)

        # 1. siblings under the longest existing prefix, within MAX_EDITS of the next component
        depth = len(parts) - 1
        while depth > 0 and not self._equal(".".join(parts[: depth - 1]), parts[depth - 1]):
            depth -= 1
        if depth > 0:
            prefix, wanted = ".".join(parts[:depth]), parts[depth]
            scored = []
            for r in self._children(prefix):
                d = edit_distance(wanted, r[1])
                if r[1] != wanted and d <= (1 if len(wanted) <= 4 else MAX_EDITS):
                    scored.append((d, r[1], r))
            for d, _, r in sorted(scored)[:limit]:
                add(r, f"{d} edit{'s' if d > 1 else ''} from `{wanted}` under `{prefix}`")
        # 2. the same last component elsewhere
        for r in self._leaf_rows(leaf):
            add(r, "the same name in another place")
        # 3. a long shared suffix, then a long shared prefix, of the last component
        need = max(4, len(leaf) // 2)
        for name, r in sorted(self._neighbours("rleaf", "rleaf", leaf[::-1]), key=lambda nr: -_common_prefix(nr[0], leaf[::-1])):
            n = _common_prefix(name, leaf[::-1])
            if name[::-1] != leaf and n >= need:
                add(r, f"shares the ending `{leaf[-n:]}`")
        for name, r in self._neighbours("leaf", "leaf", leaf):
            n = _common_prefix(name, leaf)
            if name != leaf and n >= need:
                add(r, f"shares the beginning `{leaf[:n]}`")
        return list(found.values())[:limit]

    def file(self, path: str) -> Address | None:
        """The file node at `path` as an address of all its lines, or None."""
        parent, _, name = path.strip().rpartition("/")
        start = self._lower_bound("node", "parent, name", (parent, name))
        rows = self._rows("node", "parent, name, kind, hash, summary, lines", start, 1)
        if not rows or rows[0][:2] != (parent, name) or rows[0][2] != "file":
            return None
        _, _, _, file_hash, summary, lines = rows[0]
        return Address("", "file", path.strip(), 1, lines, file_hash, summary)

    def outline(self, target: str, limit: int = OUTLINE_LIMIT) -> str:
        """The children of a directory, a file (its top-level names) or a qualified name, one
        line each with its address and summary; at most `limit` lines."""
        self.visited = self.scanned = 0
        target = clean_name(target).rstrip("/")
        if target in ("", ".", "/"):
            return self._dir_outline("", limit)
        start = self._lower_bound("node", "parent", (target,))
        if self._rows("node", "parent", start, 1)[:1] == [(target,)]:
            return self._dir_outline(target, limit)
        node = self.file(target)
        if node is not None:
            if not target.endswith(".py"):
                return f"{node.line()}\n(a text file: read its lines)"
            from ..anchor.python import module_parts

            return self._name_outline(".".join(module_parts(target)), limit, head=node.line())
        found = [a for a in self.exact(target) if a.key]
        if len(found) == 1:
            return self._name_outline(found[0].key, limit)
        if found:
            return f"`{target}` names {len(found)} places; outline one of them:\n" + "\n".join(f"  {a.line()}" for a in found)
        return f"error: unknown_node `{target}` (outline takes a directory, a file path or a qualified name; lookup finds names)"

    def _dir_outline(self, directory: str, limit: int) -> str:
        start = self._lower_bound("node", "parent", (directory,))
        rows = [r for r in self._rows("node", "parent, name, kind, hash, summary, lines", start, limit + 1) if r[0] == directory]
        lines = [f"{directory or '(root)'}/ :"]
        for parent, name, kind, h, summary, n in rows[:limit]:
            path = f"{parent}/{name}" if parent else name
            lines.append(f"  {path}/  dir, {summary}" if kind == "dir" else f"  {path}:1-{n}#{h[:8]}  file  {summary}")
        if len(rows) > limit:
            lines.append(f"  (more entries: outline a subdirectory, or lookup a name)")
        return "\n".join(lines)

    def _name_outline(self, key: str, limit: int, head: str | None = None) -> str:
        rows = self._children(key, limit + 1)
        lines = [head or f"`{key}`:"]
        lines += [f"  {self._address(r).line()}" for r in rows[:limit]]
        if not rows:
            lines.append("  (no names defined inside)")
        if len(rows) > limit:
            lines.append(f"  (more names: lookup one by its qualified name)")
        return "\n".join(lines)

    def verify(self, path: str, start: int, end: int, sha: str, text: str) -> bool:
        """The lines start..end of `text` (the file at `path`) hash to the sha (prefix) given."""
        return span_sha(text, start, end).startswith(sha.lower())


__all__ = ["Address", "AddressIndex", "IndexError_", "Lookup", "NearMiss", "clean_name", "edit_distance", "parse_address"]
