"""Deterministic perturbation of the corpus (C.2), so answers cannot come from memorization.
Ground truth is read from the perturbed source. stdlib `ast` and `tokenize` only.

1. Constants. Module-level `NAME = <literal>` (upper-case names; int, float or str) and literal
   parameter defaults of every function get new, plausible, seeded values
   (random.Random(f"{seed}:{path}:{qualname}:{param}")); booleans and None are never touched.
   - ints keep their sign and digit count, are never the original, never 0 or 1, and never a
     power of two unless the original was one;
   - floats keep their magnitude, with one significant digit changed;
   - strings get one interior letter or digit substituted, or (if too short) a two-letter
     suffix, in the same quote style.
   Only plain literals are rewritten (decimal ints and floats; single-line strings without
   prefixes or escapes). Each is rewritten by its exact source span, so formatting outside the
   edited literals is preserved byte for byte (never ast.unparse).
2. Identifiers. A seeded sample of rare function names (6+ characters, defined exactly once in
   the corpus, in at most 10 files, not dunder, keyword or builtin, and never inside any string
   literal) is renamed consistently: NAME tokens (and words in comments) of .py files through
   `tokenize`, whole words in every other file. New names are plausible snake_case built from
   a seeded word list, absent from the original corpus. Names defined in gold files come
   first, so locate questions can use them.
3. Checks. Every perturbed .py file that parsed before must still parse; one that does not is
   reverted to its original and excluded from question generation.
4. Record. Every edit is {path, line, kind, qualname, param, old, new}: kind "constant",
   "default", "rename" (the definition) or "rename_ref" (any other occurrence). An answer equal
   to `old` is parametric leakage.
"""

from __future__ import annotations

import ast
import builtins
import io
import keyword
import random
import re
import tokenize
from collections import Counter
from collections.abc import Sequence

from . import pyscan

MIN_NAME = 6
MAX_REF_FILES = 10
N_RENAMES = 200
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INT = re.compile(r"^-?(0|[1-9][0-9]*)$")
_FLOAT = re.compile(r"^-?([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$")
WORDS = (
    "amber anchor apron arbor atlas autumn badge barley basin beacon birch bishop bramble breeze brook "
    "cabin candle canyon cedar chalk chapel cinder clover cobalt comet copper coral cotton crater crimson "
    "dahlia delta desert drift ember fable falcon fennel fern ferry flint fossil garnet glacier granite "
    "grove harbor hazel heron hollow indigo island ivory jasper juniper kettle lagoon lantern larch "
    "lattice lemon linen lotus maple marble meadow mesa mica mint mosaic moss nectar nickel oasis ochre "
    "olive onyx opal orbit orchid otter oyster paddle pebble pepper pigeon pine plume pollen poplar "
    "prairie quarry quartz quill raven reef ridge river robin rust saffron sage sandal satin "
    "sequoia shale shore sierra silver slate sorrel sparrow spruce summit swallow tamarind thicket thistle "
    "timber topaz tundra tulip umber valley velvet violet walnut willow wren yarrow zephyr"
).split()


def char_offsets(text: str) -> tuple[list[int], list[str]]:
    """(char index of each line start, the lines) for converting ast positions."""
    lines = text.split("\n")
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    return starts, lines


def char_span(starts: list[int], lines: list[str], lineno: int, col: int, end_lineno: int, end_col: int) -> tuple[int, int]:
    """ast (1-based line, UTF-8 byte column) positions as character offsets into the text."""

    def at(line: int, byte_col: int) -> int:
        return starts[line - 1] + len(lines[line - 1].encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))

    return at(lineno, col), at(end_lineno, end_col)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def new_int(source: str, rng: random.Random) -> str | None:
    """A new decimal int with the same sign and digit count (see the module docstring)."""
    if not _INT.match(source):
        return None
    sign, digits = ("-", source[1:]) if source.startswith("-") else ("", source)
    value, d = int(digits), len(digits)
    low, high = (0, 9) if d == 1 else (10 ** (d - 1), 10**d - 1)

    def ok(v: int) -> bool:
        return v != value and v not in (0, 1) and (_is_power_of_two(value) or not _is_power_of_two(v))

    if d <= 2:
        choices = [v for v in range(low, high + 1) if ok(v)]
        return sign + str(rng.choice(choices)) if choices else None
    for _ in range(100):
        v = rng.randint(low, high)
        if ok(v):
            return sign + str(v)
    return None


def new_float(source: str, rng: random.Random) -> str | None:
    """The float's text with one significant digit changed (never the leading one to 0)."""
    if not _FLOAT.match(source) or float(source) == 0:
        return None
    exp_at = next((i for i, c in enumerate(source) if c in "eE"), len(source))
    mantissa = source[:exp_at]
    positions = [i for i, c in enumerate(mantissa) if c.isdigit()]
    first = next(i for i in positions if mantissa[i] != "0")
    significant = [i for i in positions if i >= first]
    i = rng.choice(significant)
    choices = [c for c in "0123456789" if c != source[i] and not (i == first and c == "0")]
    return source[:i] + rng.choice(choices) + source[i + 1 :]


def new_str(source: str, rng: random.Random) -> str | None:
    """The string literal with one interior letter or digit substituted (same character class),
    or a two-letter suffix if it has none; same quotes. Only plain one-line literals."""
    if len(source) < 2 or source[0] not in "'\"" or source[-1] != source[0] or source[:3] in ("'''", '"""') or "\\" in source or "\n" in source:
        return None
    quote, content = source[0], source[1:-1]
    interior = [i for i in range(1, len(content) - 1) if content[i].isascii() and content[i].isalnum()]
    if not interior:
        suffix = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(2))
        return quote + content + suffix + quote
    i = rng.choice(interior)
    c = content[i]
    pool = "0123456789" if c.isdigit() else "abcdefghijklmnopqrstuvwxyz" if c.islower() else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return quote + content[:i] + rng.choice([x for x in pool if x != c]) + content[i + 1 :] + quote


def new_value(kind: str, source: str, rng: random.Random) -> str | None:
    """The rewrite of an "int", "float" or "str" literal (new_int, new_float or new_str)."""
    return {"int": new_int, "float": new_float, "str": new_str}[kind](source, rng)


def perturbable(kind: str, source: str) -> bool:
    """True if a literal of this kind and source text has a form literal_edits rewrites (decided
    by the form alone: the rng only picks among values)."""
    return new_value(kind, source, random.Random(0)) is not None


def literal_edits(path: str, text: str, tree: ast.Module, seed: int) -> list[tuple[int, int, str, dict]]:
    """(start, end, new text, record) for each perturbable literal of a parsed .py file."""
    starts, lines = char_offsets(text)
    out = []
    for site in pyscan.literal_sites(tree):
        start, end = char_span(starts, lines, *site.span)
        source = text[start:end]
        kind, _ = pyscan.literal_value(site.node)
        rng = random.Random(f"{seed}:{path}:{site.qualname}:{site.param}")
        new = new_value(kind, source, rng)
        if new is None or new == source:
            continue
        record = {"path": path, "line": site.node.lineno, "kind": site.kind, "qualname": site.qualname, "param": site.param, "old": source, "new": new}
        out.append((start, end, new, record))
    return out


def string_words(tree: ast.Module) -> set[str]:
    """Every identifier-like word inside a string literal (docstrings and f-string text too)."""
    words: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            words.update(_WORD.findall(node.value))
    return words


def choose_renames(texts: dict[str, str], trees: dict[str, ast.Module], gold: Sequence[str], seed: int, n: int = N_RENAMES) -> dict[str, tuple[str, str, str]]:
    """{old name: (new name, defining path, qualname)} for a seeded sample of eligible names."""
    defined = Counter(name for tree in trees.values() for name in pyscan.defined_names(tree))
    in_strings = set().union(*(string_words(t) for t in trees.values())) if trees else set()
    file_words = {p: set(_WORD.findall(t)) for p, t in texts.items()}
    files_with = Counter(w for words in file_words.values() for w in words)
    reserved = set(keyword.kwlist) | set(keyword.softkwlist) | set(dir(builtins))
    eligible: dict[str, tuple[str, str]] = {}
    for path in sorted(trees):
        for qualname, _ in pyscan.functions(trees[path]):
            name = qualname.rsplit(".", 1)[-1]
            if (
                len(name.lstrip("_")) >= MIN_NAME
                and defined[name] == 1
                and not (name.startswith("__") and name.endswith("__"))
                and name not in reserved
                and name not in in_strings
                and files_with[name] <= MAX_REF_FILES
            ):
                eligible[name] = (path, qualname)
    rng = random.Random(f"{seed}:rename")
    gold_set = set(gold)
    first = sorted(name for name, (path, _) in eligible.items() if path in gold_set)
    others = sorted(name for name in eligible if name not in set(first))
    rng.shuffle(others)
    chosen = (first + others)[:n]
    taken = set(files_with)
    out = {}
    for old in chosen:
        parts = max(2, len([p for p in old.strip("_").split("_") if p]))
        for _ in range(1000):
            new = ("_" if old.startswith("_") else "") + "_".join(rng.choice(WORDS) for _ in range(min(parts, 3)))
            if new not in taken:
                break
        else:
            continue
        taken.add(new)
        out[old] = (new, *eligible[old])
    return out


def python_rename_edits(path: str, text: str, renames: dict[str, tuple[str, str, str]]) -> list[tuple[int, int, str, dict]]:
    """Edits of NAME tokens (and comment words) naming a renamed function."""
    starts, _ = char_offsets(text)
    words = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, sorted(renames, key=len, reverse=True))) + r")(?![\w])")
    out = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return []
    for k, tok in enumerate(tokens):
        (row, col) = tok.start
        if tok.type == tokenize.NAME and tok.string in renames:
            spots = [(col, tok.string)]
        elif tok.type == tokenize.COMMENT:
            spots = [(col + m.start(), m.group(1)) for m in words.finditer(tok.string)]
        else:
            continue
        after_def = k > 0 and tokens[k - 1].type == tokenize.NAME and tokens[k - 1].string == "def"
        for c, old in spots:
            new, def_path, qualname = renames[old]
            start = starts[row - 1] + c
            is_def = tok.type == tokenize.NAME and path == def_path and after_def
            record = {"path": path, "line": row, "kind": "rename" if is_def else "rename_ref", "qualname": qualname if is_def else None, "param": None, "old": old, "new": new}
            out.append((start, start + len(old), new, record))
    return out


def prose_rename_edits(path: str, text: str, renames: dict[str, tuple[str, str, str]]) -> list[tuple[int, int, str, dict]]:
    """Whole-word edits of renamed names in a non-Python file."""
    pattern = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, sorted(renames, key=len, reverse=True))) + r")(?![\w])")
    out = []
    for m in pattern.finditer(text):
        old = m.group(1)
        line = text.count("\n", 0, m.start()) + 1
        record = {"path": path, "line": line, "kind": "rename_ref", "qualname": None, "param": None, "old": old, "new": renames[old][0]}
        out.append((m.start(), m.end(), renames[old][0], record))
    return out


def apply_edits(text: str, edits: Sequence[tuple[int, int, str, dict]]) -> str:
    """Apply non-overlapping (start, end, new) edits; ValueError on an overlap."""
    ordered = sorted(edits, key=lambda e: e[0])
    for a, b in zip(ordered, ordered[1:]):
        if b[0] < a[1]:
            raise ValueError(f"overlapping edits at {a[0]} and {b[0]}")
    out = text
    for start, end, new, _ in reversed(ordered):
        out = out[:start] + new + out[end:]
    return out


def perturb_corpus(files: Sequence[tuple[str, str]], order: Sequence[str], gold: Sequence[str], seed: int, n_renames: int = N_RENAMES):
    """Perturb the corpus files (see the module docstring). Returns (perturbed text by path,
    edit records in path and position order, reverted paths). The signature is the one
    corpus.build_corpus calls."""
    texts = dict(files)
    trees = {p: t for p, t in ((p, pyscan.parse(texts[p])) for p in texts if p.endswith(".py")) if t is not None}
    renames = choose_renames(texts, trees, gold, seed, n_renames)
    rename_words = set(renames)
    perturbed, records, excluded = {}, [], []
    for path in order:
        if path not in texts:
            continue
        text = texts[path]
        edits = literal_edits(path, text, trees[path], seed) if path in trees else []
        if rename_words & set(_WORD.findall(text)):
            present = {k: v for k, v in renames.items() if k in text}
            edits += python_rename_edits(path, text, present) if path.endswith(".py") else prose_rename_edits(path, text, present)
        if not edits:
            continue
        new_text = apply_edits(text, edits)
        if path in trees and pyscan.parse(new_text) is None:
            excluded.append(path)
            continue
        perturbed[path] = new_text
        records += [e[3] for e in sorted(edits, key=lambda e: e[0])]
    return perturbed, records, excluded


__all__ = ["apply_edits", "choose_renames", "literal_edits", "new_float", "new_int", "new_str", "perturb_corpus"]
