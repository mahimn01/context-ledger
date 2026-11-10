"""Python source facts the corpus, the perturbation and the question generator share (stdlib ast).

Qualnames follow the anchors' scheme (anchor/python.py): `Outer.method`, `f.<locals>.g`.
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass

from ..anchor.python import module_parts

UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def parse(text: str) -> ast.Module | None:
    """The module's AST, or None if it does not parse (or has null bytes). Syntax warnings of
    old-style escapes in the corpus are silenced."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def module_path(path: str) -> str:
    """"Lib/a/b.py" -> "a.b"; "Lib/pkg/__init__.py" -> "pkg"; other files keep their directories.
    The same rule Tier 1 uses for anchored paths (anchor.python.module_parts)."""
    return ".".join(module_parts(path))


def literal_value(node: ast.expr) -> tuple[str, object] | None:
    """("int" | "float" | "str", value) for a literal int, float or str (a negative number
    counts), never a bool or None; None for anything else."""
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        node, sign = node.operand, -1
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool) or node.value is None:
        return None
    value = node.value
    if isinstance(value, int):
        return "int", sign * value
    if isinstance(value, float):
        return "float", sign * value
    if isinstance(value, str) and sign == 1:
        return "str", value
    return None


@dataclass(frozen=True)
class LiteralSite:
    """A module constant (`kind` "constant", `param` None) or a parameter default ("default")."""

    kind: str
    qualname: str  # the constant's name, or the function's qualname
    param: str | None
    node: ast.expr

    @property
    def span(self) -> tuple[int, int, int, int]:
        n = self.node
        return n.lineno, n.col_offset, n.end_lineno, n.end_col_offset


def functions(tree: ast.Module) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every function and method at any depth, with its qualname, in source order."""
    out: list[tuple[str, ast.AST]] = []

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, _FUNCS):
                qualname = prefix + node.name
                out.append((qualname, node))
                visit(node.body, qualname + ".<locals>.")
            elif isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for field in ("body", "orelse", "finalbody", "handlers"):
                    for child in getattr(node, field, []) or []:
                        visit(child.body if isinstance(child, ast.ExceptHandler) else [child], prefix)

    visit(tree.body, "")
    return out


def parameter_defaults(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, ast.expr]]:
    """(parameter name, default expression) for every parameter that has a default."""
    args = func.args
    positional = [*args.posonlyargs, *args.args]
    pairs = list(zip(positional[len(positional) - len(args.defaults) :], args.defaults))
    pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
    return [(a.arg, d) for a, d in pairs]


def literal_sites(tree: ast.Module) -> list[LiteralSite]:
    """Module-level `NAME = <literal>` (upper-case names, one Name target) and literal
    parameter defaults of every function, in source order."""
    sites = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            name, value = node.target.id, node.value
        else:
            continue
        if UPPER.match(name) and literal_value(value) is not None:
            sites.append(LiteralSite("constant", name, None, value))
    for qualname, func in functions(tree):
        for param, default in parameter_defaults(func):
            if literal_value(default) is not None:
                sites.append(LiteralSite("default", qualname, param, default))
    return sorted(sites, key=lambda s: s.span)


def direct_raises(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Bare class names raised in the function's own body (`raise X`, `raise X(...)`,
    `raise mod.X(...)`), not in nested functions, classes or lambdas; re-raises are skipped."""
    names: set[str] = set()
    stack = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPES):
            continue
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name):
                names.add(exc.id)
            elif isinstance(exc, ast.Attribute):
                names.add(exc.attr)
        stack.extend(ast.iter_child_nodes(node))
    return names


def defined_names(tree: ast.Module) -> list[str]:
    """Names this module defines: every function and class at any depth, and module-level
    assignment targets (one entry per definition)."""
    names = [qualname.rsplit(".", 1)[-1] for qualname, _ in functions(tree)]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.append(node.name)
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        names += [t.id for t in targets if isinstance(t, ast.Name)]
    return names


def top_level_starts(tree: ast.Module) -> list[int]:
    """1-based first lines (decorators included) of each top-level def and class."""
    starts = []
    for node in tree.body:
        if isinstance(node, (*_FUNCS, ast.ClassDef)):
            starts.append(min([node.lineno, *(d.lineno for d in node.decorator_list)]))
    return starts
