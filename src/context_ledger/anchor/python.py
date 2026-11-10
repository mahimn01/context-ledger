"""Python symbol anchors built with the stdlib `ast` module.

Offsets are character offsets into newline-normalized source text. `ast` reports columns as
UTF-8 byte offsets inside a line; they are converted to character offsets here.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import replace

from ..ids import sha256_hex
from ..model import PyAnchor, TextAnchor
from .text import Resolution, resolve_text


class _Offsets:
    """Maps (lineno, UTF-8 byte column) positions from `ast` to character offsets."""

    def __init__(self, text: str) -> None:
        self.lines = text.split("\n")
        self.starts: list[int] = []
        pos = 0
        for line in self.lines:
            self.starts.append(pos)
            pos += len(line) + 1

    def char(self, lineno: int, byte_col: int) -> int:
        line = self.lines[lineno - 1]
        return self.starts[lineno - 1] + len(line.encode("utf-8")[:byte_col].decode("utf-8"))


def _ast_hash(node: ast.AST) -> str:
    """Hash of the node's structure with its own name blanked (assigns: the value only), so
    formatting, comments and a pure rename keep the hash."""
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        target = node.value
    else:
        target = copy.deepcopy(node)
        target.name = ""
    return sha256_hex(ast.dump(target, annotate_fields=True, include_attributes=False).encode("utf-8"))


def _collect_defs(node: ast.AST, prefix: str, out: list[tuple[str, str, ast.AST]]) -> None:
    """Append (qualname, node_kind, node) for every def and class below `node`, at any depth."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = prefix + child.name
            out.append((qualname, "function", child))
            _collect_defs(child, qualname + ".<locals>.", out)
        elif isinstance(child, ast.ClassDef):
            qualname = prefix + child.name
            out.append((qualname, "class", child))
            _collect_defs(child, qualname + ".", out)
        else:
            _collect_defs(child, prefix, out)


def _module_assigns(tree: ast.Module) -> list[tuple[str, str, ast.AST]]:
    """(name, "assign", node) for each simple Name target of a module-level Assign or a valued AnnAssign."""
    out: list[tuple[str, str, ast.AST]] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            out.extend((t.id, "assign", stmt) for t in stmt.targets if isinstance(t, ast.Name))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            out.append((stmt.target.id, "assign", stmt))
    return out


def module_parts(path: str) -> list[str]:
    """The dotted module a source path defines, as parts: "Lib/a/b.py" -> [a, b];
    "Lib/pkg/__init__.py" -> [pkg]; a path outside Lib/ keeps its directories."""
    stem = path[len("Lib/") :] if path.startswith("Lib/") else path
    stem = stem[: -len(".py")] if stem.endswith(".py") else stem
    parts = [p for p in stem.split("/") if p]
    return parts[:-1] if parts and parts[-1] == "__init__" else parts


def anchored_path(anchor: PyAnchor) -> tuple[str, ...]:
    """The full anchored path of a symbol: its module parts (`module_parts` of the anchor's
    path) followed by its qualname parts with "<locals>" dropped. multiprocessing/context.py
    and BaseContext.f give (multiprocessing, context, BaseContext, f)."""
    return (*module_parts(anchor.path), *(p for p in anchor.qualname.split(".") if p and p != "<locals>"))


def python_anchors(source_text: str, path: str, commit: str | None = None) -> list[PyAnchor]:
    """Return anchors for every function, async function and class at any depth, and every
    simple Name target of a module-level assignment, ordered by position.

    Qualnames follow Python: `Outer.method`, `f.<locals>.g`, constants by name. A range starts
    at the `@` of the first decorator (if any) and ends at end_lineno/end_col_offset.
    Raises SyntaxError if `source_text` does not parse."""
    tree = ast.parse(source_text)
    offsets = _Offsets(source_text)
    found: list[tuple[str, str, ast.AST]] = []
    _collect_defs(tree, "", found)
    found.extend(_module_assigns(tree))
    anchors = []
    for qualname, node_kind, node in found:
        start = offsets.char(node.lineno, node.col_offset)
        decorators = getattr(node, "decorator_list", None)
        if decorators:
            first = decorators[0]
            start = source_text.rfind("@", 0, offsets.char(first.lineno, first.col_offset))
        end = offsets.char(node.end_lineno, node.end_col_offset)
        anchors.append(
            PyAnchor(
                path=path,
                qualname=qualname,
                node_kind=node_kind,
                ast_hash=_ast_hash(node),
                start=start,
                end=end,
                commit=commit,
            )
        )
    anchors.sort(key=lambda a: (a.start, -a.end, a.qualname))
    return anchors


def resolve_python(anchor: PyAnchor, new_source_text: str, old_text: str | None = None) -> Resolution:
    """Re-resolve a Python symbol anchor against new source text.

    - Unparseable source: with `old_text` (the span's previous exact text), fall back to
      resolve_text on a context-free TextAnchor and add note "fallback"; else DANGLING
      "syntax_error".
    - Same qualname and kind present: same ast_hash gives OK (offsets unchanged) or MOVED
      (new offsets); a different hash gives STALE.
    - Qualname absent: a unique node of the same kind and ast_hash gives MOVED "renamed" with
      the new qualname; otherwise DANGLING."""
    try:
        anchors = python_anchors(new_source_text, anchor.path, anchor.commit)
    except (SyntaxError, ValueError):
        if old_text is None:
            return Resolution("DANGLING", anchor, "syntax_error")
        text_anchor = TextAnchor(start=anchor.start, end=anchor.end, exact=old_text, prefix="", suffix="")
        res = resolve_text(text_anchor, new_source_text)
        return replace(res, note="fallback" + (f":{res.note}" if res.note else ""))

    same_name = [a for a in anchors if a.qualname == anchor.qualname and a.node_kind == anchor.node_kind]
    if same_name:
        matching = [a for a in same_name if a.ast_hash == anchor.ast_hash]
        if not matching:
            return Resolution("STALE", anchor, "changed")
        if any((a.start, a.end) == (anchor.start, anchor.end) for a in matching):
            return Resolution("OK", anchor)
        return Resolution("MOVED", matching[0])

    renamed = [a for a in anchors if a.node_kind == anchor.node_kind and a.ast_hash == anchor.ast_hash]
    if len(renamed) == 1:
        return Resolution("MOVED", renamed[0], "renamed")
    return Resolution("DANGLING", anchor, "not_found" if not renamed else "ambiguous")
