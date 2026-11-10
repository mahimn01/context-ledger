"""Build the address index of a corpus (BUILD-V4 item 1): once per corpus, no model calls, from
the corpus file texts alone (never a question, a gold field or any run artifact).

One SQLite file, address-v4.sqlite, with these tables per rho level r (each level holds only the
files of C-r, so a lookup at rho r searches C-r's own sorted keys):

    sym_r(rank, parent, leaf, kind, path, start, end, sha, summary)
        every addressable Python name, sorted by (parent, leaf, path, start); rank 0..N-1 is its
        position. The qualified name is parent + "." + leaf (parent "" for a top-level module).
        kind: "module" (the file; parent the package), "class", "function", "constant" (a simple
        Name target of a module-level assignment), "param" (a parameter with a default; parent
        the function). Defs and classes at any depth except inside a function (<locals>). The
        address is the 1-based inclusive line range (decorators included) and the sha256 of
        those lines' UTF-8 bytes joined by "\\n". Qualified names follow anchor.python
        (module_parts of the path, then the qualname), the names Tier 1 and the gate anchor by.
    leaf_r(rank, leaf, sym)     the non-param names by their last component (sym = the rank in
                                sym_r): the same name in another module
    rleaf_r(rank, rleaf, sym)   the same by the reversed last component: a shared suffix
    node_r(rank, parent, name, kind, hash, summary, lines)
        the Merkle tree of directories and files, sorted by (parent, name): a file's hash is the
        sha256 of its bytes; a directory's is the sha256 of its children's "kind name hash"
        lines in sorted order, so the root hash (meta) certifies every byte of C-r. Within a
        file, the def/class/constant addresses of sym_r are content-addressed line ranges.

Summaries are deterministic and never hold a value: a def's signature with each default shown
as "=…", "class Name(bases)", "NAME = …", a module's first docstring line (or its def/class
counts), a text file's first line. A value is only ever seen by reading its address.

Sorting is done by SQLite (BINARY collation compares UTF-8 bytes, the order of Python's str
comparison), so a build holds no corpus-sized list in memory.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

from ..anchor.python import module_parts, python_anchors

INDEX_VERSION = "address-v4"
INDEX_FILE = "address-v4.sqlite"
ELIDED = "…"
_SUMMARY = 100


def span_sha(text: str, start: int, end: int) -> str:
    """The sha256 of lines start..end (1-based, inclusive) of `text`, joined by newlines."""
    return hashlib.sha256("\n".join(text.split("\n")[start - 1 : end]).encode("utf-8")).hexdigest()


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _SUMMARY else text[: _SUMMARY - 1] + ELIDED


def _parse(text: str) -> ast.Module | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def _args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The parameter list with every default elided."""
    a = node.args
    positional = [*a.posonlyargs, *a.args]
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    out = [p.arg + ("=" + ELIDED if d is not None else "") for p, d in zip(positional, defaults)]
    if a.posonlyargs:
        out.insert(len(a.posonlyargs), "/")
    if a.vararg:
        out.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        out.append("*")
    out += [p.arg + ("=" + ELIDED if d is not None else "") for p, d in zip(a.kwonlyargs, a.kw_defaults)]
    if a.kwarg:
        out.append("**" + a.kwarg.arg)
    return ", ".join(out)


def _doc_line(node) -> str:
    doc = ast.get_docstring(node, clean=True) if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else None
    return next((line.strip() for line in (doc or "").splitlines() if line.strip()), "")


def summary_of(node: ast.AST, name: str) -> str:
    """The one-line summary of a def, class or constant (module docstring)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        head = f"{'async def' if isinstance(node, ast.AsyncFunctionDef) else 'def'} {node.name}({_args(node)})"
    elif isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        head = f"class {node.name}" + (f"({bases})" if bases else "")
    else:
        return _clip(f"{name} = {ELIDED}")
    doc = _doc_line(node)
    return _clip(head + (f": {doc}" if doc else ""))


def _params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, ast.expr]]:
    a = node.args
    positional = [*a.posonlyargs, *a.args]
    out = list(zip([p.arg for p in positional[len(positional) - len(a.defaults) :]], a.defaults))
    out += [(p.arg, d) for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
    return out


def file_symbols(path: str, text: str) -> list[tuple[str, str, str, int, int, str]]:
    """(parent, leaf, kind, start, end, summary) of every addressable name of one file: the
    module itself (if it has a module name), then its anchors (anchor.python.python_anchors, the
    spans L registers) outside function bodies, and each defaulted parameter of a def. A .py file
    that does not parse gives only its module entry."""
    parts = module_parts(path)
    if not path.endswith(".py") or not parts:
        return []
    module = ".".join(parts)
    n = text.count("\n") + 1
    tree = _parse(text)
    doc = _doc_line(tree) if tree is not None else ""
    out = [(".".join(parts[:-1]), parts[-1], "module", 1, n, _clip(doc) if doc else "")]
    if tree is None:
        return out
    nodes: dict[tuple[int, int, str], ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes[(node.lineno, node.col_offset, node.name)] = node
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anchors = python_anchors(text, path)
    line_of = _line_starts(text)
    for anchor in anchors:
        if "<locals>" in anchor.qualname:
            continue
        start, end = line_of(anchor.start), line_of(max(anchor.start, anchor.end - 1))
        qual = anchor.qualname.split(".")
        parent = ".".join([module, *qual[:-1]])
        if anchor.node_kind == "assign":
            out.append((parent, qual[-1], "constant", start, end, summary_of(None, qual[-1])))
            continue
        node = _node_at(nodes, text, anchor, qual[-1])
        kind = "class" if anchor.node_kind == "class" else "function"
        out.append((parent, qual[-1], kind, start, end, summary_of(node, qual[-1]) if node is not None else _clip(f"{kind} {qual[-1]}")))
        if kind == "function" and node is not None:
            for param, default in _params(node):
                out.append((f"{parent}.{qual[-1]}", param, "param", start, default.end_lineno or start, _clip(f"parameter {param}={ELIDED} of {qual[-1]}")))
    return out


def _line_starts(text: str) -> Callable[[int], int]:
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    import bisect

    return lambda offset: bisect.bisect_right(starts, offset)


def _node_at(nodes: dict, text: str, anchor, name: str):
    """The def/class node an anchor names (its `def`/`class` line: the anchor starts at the
    first decorator, so match by name among the nodes whose line falls inside the anchor)."""
    first = text.count("\n", 0, anchor.start) + 1
    last = text.count("\n", 0, anchor.end) + 1
    found = [node for (line, _, n), node in nodes.items() if n == name and first <= line <= last]
    return min(found, key=lambda node: node.lineno) if found else None


def file_summary(path: str, text: str) -> str:
    """A file's one-line summary: a .py module's first docstring line, else its def and class
    counts; another file's first non-blank line."""
    if path.endswith(".py"):
        tree = _parse(text)
        if tree is None:
            return "python (does not parse)"
        doc = _doc_line(tree)
        if doc:
            return _clip(doc)
        defs = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body)
        classes = sum(isinstance(n, ast.ClassDef) for n in tree.body)
        return f"python module: {defs} functions, {classes} classes"
    return _clip(next((line for line in text.splitlines() if line.strip()), ""))


_SCHEMA_RAW = """
CREATE TABLE raw_sym (parent TEXT, leaf TEXT, kind TEXT, path TEXT, start INTEGER, end INTEGER, sha TEXT, summary TEXT, rho INTEGER);
CREATE TABLE raw_file (path TEXT, hash TEXT, summary TEXT, lines INTEGER, rho INTEGER);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def build_address_index(
    texts: dict[str, str],
    member_rho: dict[str, int],
    rhos: Iterable[int],
    out: str | Path,
    *,
    corpus_id: str,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Write `out` (address-v4.sqlite) for the corpus whose files are `texts` (path -> text),
    each in C-r for every r >= member_rho[path]; return the meta (also in the file's meta
    table): version, corpus id, and per rho the counts and the Merkle root."""
    out = Path(out)
    tmp = out.with_name(out.name + ".building")
    if tmp.exists():
        tmp.unlink()
    rhos = sorted(set(rhos))
    db = sqlite3.connect(tmp)
    try:
        db.executescript(_SCHEMA_RAW)
        for k, path in enumerate(sorted(texts)):
            text = texts[path]
            rho = member_rho[path]
            file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            db.execute("INSERT INTO raw_file VALUES (?, ?, ?, ?, ?)", (path, file_hash, file_summary(path, text), text.count("\n") + 1, rho))
            rows = [(parent, leaf, kind, path, start, end, span_sha(text, start, end), summary, rho)
                    for parent, leaf, kind, start, end, summary in file_symbols(path, text)]  # fmt: skip
            db.executemany("INSERT INTO raw_sym VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            if progress is not None and k and k % 2000 == 0:
                progress(f"address index: parsed {k} of {len(texts)} files")
        meta: dict = {"version": INDEX_VERSION, "corpus_id": corpus_id, "rhos": rhos, "levels": {}}
        for rho in rhos:
            meta["levels"][str(rho)] = _level(db, rho)
            if progress is not None:
                level = meta["levels"][str(rho)]
                progress(f"address index: C{rho}: {level['symbols']} names, {level['files']} files, root {level['root'][:12]}")
        db.execute("DROP TABLE raw_sym")
        db.execute("DROP TABLE raw_file")
        db.execute("INSERT INTO meta VALUES ('meta', ?)", (json.dumps(meta, sort_keys=True),))
        db.commit()
        db.execute("VACUUM")
    finally:
        db.close()
    tmp.replace(out)
    return meta


def _level(db: sqlite3.Connection, rho: int) -> dict:
    """The sorted tables of C-rho and its Merkle root."""
    db.executescript(f"""
        CREATE TABLE sym_{rho} (rank INTEGER PRIMARY KEY, parent TEXT, leaf TEXT, kind TEXT, path TEXT, start INTEGER, end INTEGER, sha TEXT, summary TEXT);
        INSERT INTO sym_{rho} SELECT row_number() OVER (ORDER BY parent, leaf, path, start) - 1, parent, leaf, kind, path, start, end, sha, summary
            FROM raw_sym WHERE rho <= {rho};
        CREATE TABLE leaf_{rho} (rank INTEGER PRIMARY KEY, leaf TEXT, sym INTEGER);
        INSERT INTO leaf_{rho} SELECT row_number() OVER (ORDER BY leaf, rank) - 1, leaf, rank FROM sym_{rho} WHERE kind != 'param';
        CREATE TABLE rleaf_{rho} (rank INTEGER PRIMARY KEY, rleaf TEXT, sym INTEGER);
    """)  # fmt: skip
    rows = db.execute(f"SELECT rank, leaf FROM sym_{rho} WHERE kind != 'param'").fetchall()
    rows.sort(key=lambda r: (r[1][::-1], r[0]))
    db.executemany(f"INSERT INTO rleaf_{rho} VALUES (?, ?, ?)", ((n, leaf[::-1], rank) for n, (rank, leaf) in enumerate(rows)))
    files = db.execute("SELECT path, hash, summary, lines FROM raw_file WHERE rho <= ? ORDER BY path", (rho,)).fetchall()
    nodes, root = merkle(files)
    db.execute(f"CREATE TABLE node_{rho} (rank INTEGER PRIMARY KEY, parent TEXT, name TEXT, kind TEXT, hash TEXT, summary TEXT, lines INTEGER)")
    db.executemany(f"INSERT INTO node_{rho} VALUES (?, ?, ?, ?, ?, ?, ?)", ((n, *row) for n, row in enumerate(sorted(nodes))))
    symbols = db.execute(f"SELECT count(*) FROM sym_{rho}").fetchone()[0]
    return {"symbols": symbols, "names": len(rows), "files": len(files), "nodes": len(nodes), "root": root}


def merkle(files: list[tuple[str, str, str, int]]) -> tuple[list[tuple[str, str, str, str, str, int]], str]:
    """(nodes as (parent, name, kind, hash, summary, lines), root hash) of the directory tree of
    `files` ((path, sha256, summary, lines)). A directory's hash is the sha256 of its children's
    "kind name hash" lines, sorted by name; its summary counts its files (lines 0)."""
    children: dict[str, dict[str, tuple[str, str, str, int]]] = {"": {}}
    counts: dict[str, int] = {"": 0}
    for path, file_hash, summary, lines in files:
        parts = path.split("/")
        for depth in range(len(parts) - 1):
            parent, name = "/".join(parts[:depth]), parts[depth]
            here = "/".join(parts[: depth + 1])
            children.setdefault(here, {})
            counts[here] = counts.get(here, 0) + 1
            children.setdefault(parent, {})[name] = ("dir", "", "", 0)
        children.setdefault("/".join(parts[:-1]), {})[parts[-1]] = ("file", file_hash, summary, lines)
        counts[""] += 1
    hashes: dict[str, str] = {}

    def digest(directory: str) -> str:
        lines = []
        for name in sorted(children.get(directory, {})):
            kind, h, _, _ = children[directory][name]
            if kind == "dir":
                h = digest(f"{directory}/{name}" if directory else name)
            lines.append(f"{kind} {name} {h}\n")
        hashes[directory] = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
        return hashes[directory]

    root = digest("")
    nodes = []
    for directory, kids in children.items():
        for name, (kind, h, summary, lines) in kids.items():
            here = f"{directory}/{name}" if directory else name
            nodes.append((directory, name, kind, hashes[here] if kind == "dir" else h, f"{counts.get(here, 0)} files" if kind == "dir" else summary, lines))
    return nodes, root


__all__ = ["ELIDED", "INDEX_FILE", "INDEX_VERSION", "build_address_index", "file_symbols", "file_summary", "merkle", "span_sha", "summary_of"]
