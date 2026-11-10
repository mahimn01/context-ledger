"""Tier 1: quotes must really occur in the cited verbatim spans, and they must witness the claim.

A claim's checkable tokens are split in two:

- IDENTIFIERS name code: dotted or underscored names (`a.b`, `a_b`), call forms (`f()`), and
  backticked spans that fully match the identifier grammar (`BaseContext.method`, `p`).
- VALUES are what the claim asserts about it: numbers (compared by canonical Decimal), quoted
  strings (by their literal value), backticked spans that are not identifiers (a backticked
  number is a number, a backticked string literal is its value, anything else an exact
  token) and exact tokens (hex/octal/binary literals, versions, dates, IP addresses).

A VALUE must be covered by the verbatim quotes of the passing citations: the quote is the only
witness that the span says this value. An IDENTIFIER passes if (a) the quotes cover it, or (b)
it equals a contiguous run of whole components of the full anchored path of a passing
citation's span (anchor.python.anchored_path: module parts from the source path, then the
qualname without "<locals>"). The anchor is the principled witness for a name: the ledger
resolved that span by parsing the source, so the span IS `multiprocessing.context`'s
`BaseContext.violet_juniper_sparrow`, whether or not the defining line spells the class name.
When the quote occurs exactly once in the cited span, the innermost registered Python anchor
of the same source version that contains it witnesses names too: quoting a method's `def`
line from a cited class span anchors `mod.Class.method`, because parsing placed that line in
that method (`innermost_anchor`).
Anchoring never vouches for a value, so `BaseContext.violet_juniper_sparrow = 'ignore'`
against the quote `violet_juniper_sparrow = 'igoore'` still fails (value_not_covered). Spans
without a Python anchor (text spans) only get rule (a). A citation passes when it cites a span
that is verbatim in the window (Tier 0) with a quote that occurs in the span text after
normalize_for_match and is at least min(20, len(span text)) characters long.

Two more witnesses (v2), both resting on how the ledger registered the passing citation's span:

- SOURCE-PATH WITNESS. Before values and identifiers are checked, every occurrence in the
  normalized claim text (backticked or bare) of the source path of a passing citation's span,
  or of a suffix of it made of whole path components ending in the file name
  (`Lib/re/_constants.py`, `re/_constants.py`, `_constants.py`), is removed. The ledger
  registered that span from that file, so the path is witnessed the way the anchor witnesses
  names. The path of any other file is not removed and still has to be covered by a quote.
- MODULE-PREFIX WITNESS. For a passing citation of a `.py` source with module path M
  (`module_parts` of the source path, as in `anchored_path`), an identifier `P.N` passes if P
  is a run of trailing components of M (for `dbm.sqlite3`: `sqlite3` or `dbm.sqlite3`) and N
  is witnessed by that same citation: covered by its quote as a whole token, or anchored in its
  anchored paths. This holds for spans with or without a Python anchor.

Neither witnesses a value: a wrong number or string still fails value_not_covered.

Failure reasons: no_valid_quote, value_not_covered (checked first), identifier_not_anchored.
The last two carry `detail`: the offending tokens, in claim order.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal

from ..anchor.python import anchored_path, module_parts
from ..anchor.text import normalize_for_match
from ..model import Claim, Knowledge, PyAnchor
from ..store.blobs import BlobStore
from ..window import WindowState, kappa, ref_kind_of
from .tier0 import Verdict

_MIN_QUOTE = 20
# A number: optional sign; digits grouped either by "," in strict groups of three or by "_"
# between digits; optional fraction; optional exponent.
_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+(?:_\d+)*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
# Matched as exact strings, never as numbers: hex/octal/binary literals and dotted, dashed or
# coloned digit groups with two or more separators (versions, IP addresses, dates, times).
_EXACT = r"0[xXoObB][0-9A-Fa-f_]+|\d+(?:[-./:]\d+){2,}(?!\w)"
_NUMERIC_SCAN = re.compile(rf"(?<![\w.])(?:({_EXACT})|({_NUMBER})(?!\d|\.\d))")
_VALUE_TOKEN = re.compile(
    r"`([^`]+)`"  # backticked span
    r"|(?<!\w)\"([^\"\n]+)\"(?!\w)"  # double-quoted string
    r"|(?<!\w)'([^'\n]+)'(?!\w)"  # single-quoted string (not an apostrophe)
    rf"|(?<![\w.])({_EXACT})"  # literal, version, IP address or date: exact string
    rf"|(?<![\w.])({_NUMBER})(?!\d|\.\d)"  # number not glued to a word or a version
    r"|(\w+(?:[._]\w+)+(?:\(\))?|\w+\(\))"  # code-like token: a_b, a.b, f()
)
# The identifier grammar: dotted Python names, optionally called with no arguments.
IDENTIFIER = re.compile(r"[^\W\d]\w*(?:\.[^\W\d]\w*)*(?:\(\))?")
_FULL_NUMBER = re.compile(_NUMBER)


def _backticked(content: str) -> tuple[str, str]:
    """(kind, token) of a backticked span: an identifier, a number, a string literal's value,
    or else the exact text."""
    if IDENTIFIER.fullmatch(content):
        return "identifier", content
    if _FULL_NUMBER.fullmatch(content):
        return "number", content
    try:
        value = ast.literal_eval(content)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        value = None
    if isinstance(value, str) and value and content[:1] in "'\"" and "\n" not in value:
        return "exact", value
    return "exact", content


def value_tokens_typed(text: str) -> list[tuple[str, str]]:
    """The checkable tokens of a claim, in order, as (kind, token), kind "identifier", "number"
    or "exact" (see the module docstring for the split).

    Numbers follow the grammar: optional sign, digits grouped by "_" or by "," in strict groups
    of three, optional fraction and exponent; not preceded by a word char or ".", and not
    followed by a digit or ".digit" (so "3.15.0" is not a number). Hex/octal/binary literals,
    versions, IP addresses and dates are exact tokens, and so are quoted strings (contents
    only). Code-like tokens (with `_` or `.` between word chars, or ending in "()", returned
    without the "()") are identifiers when they match the identifier grammar and exact tokens
    otherwise (e.g. "3.15.0rc1")."""
    tokens = []
    for match in _VALUE_TOKEN.finditer(text):
        backtick, double, single, exact, number, code = match.groups()
        if backtick is not None:
            tokens.append(_backticked(backtick))
        elif number is not None:
            tokens.append(("number", number))
        elif code is not None:
            code = code.removesuffix("()")
            tokens.append(("identifier" if IDENTIFIER.fullmatch(code) else "exact", code))
        else:
            tokens.append(("exact", next(g for g in (double, single, exact) if g is not None)))
    return tokens


def value_tokens(text: str) -> list[str]:
    """The token strings of `value_tokens_typed` (identifiers and values), in order."""
    return [token for _, token in value_tokens_typed(text)]


def numeric_value(token: str) -> Decimal:
    """Canonical value of a number token: grouping removed, Decimal, normalize() in a context
    wide enough that nothing is rounded. So 300 == 300.0 == 3e2 == 3E+2 == +300 and
    1,000 == 1_000 == 1000, while -5 != 5."""
    digits = token.replace(",", "").replace("_", "")
    exact = Context(prec=len(digits) + 1, Emax=MAX_EMAX, Emin=MIN_EMIN)
    return Decimal(digits).normalize(exact)


def numeric_values(text: str) -> set[Decimal]:
    """Canonical values of every number token in `text` (exact-string tokens are skipped, so a
    date never yields its year and a version never yields its parts)."""
    return {numeric_value(m.group(2)) for m in _NUMERIC_SCAN.finditer(text) if m.group(2) is not None}


def _contains_token(haystack: str, token: str) -> bool:
    """`token` occurs in `haystack` not glued to word characters (so "3" is not found in "37")."""
    return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", haystack) is not None


def missing_values(claim_text: str, covered: str) -> list[str]:
    """The VALUE tokens of `claim_text` that `covered` does not cover (both already normalized),
    in claim order without repeats: a number is covered if some number token of `covered` has
    an equal canonical value; any other value must occur as a whole token (M1's rule).
    Identifiers are not checked here."""
    numbers: set[Decimal] | None = None
    missing = []
    for kind, token in value_tokens_typed(claim_text):
        if kind == "number":
            numbers = numeric_values(covered) if numbers is None else numbers
            if numeric_value(token) not in numbers:
                missing.append(token)
        elif kind == "exact" and not _contains_token(covered, normalize_for_match(token)):
            missing.append(token)
    return list(dict.fromkeys(missing))


def values_covered(claim_text: str, covered: str) -> bool:
    """Every VALUE token of `claim_text` occurs in `covered` (`missing_values` is empty)."""
    return not missing_values(claim_text, covered)


def is_anchored(identifier: str, path: tuple[str, ...]) -> bool:
    """True if `identifier` (without a trailing "()") is a contiguous run of whole components
    of `path`: "context.BaseContext" is in (multiprocessing, context, BaseContext, f);
    "BaseCont" and "Context.f" are not."""
    parts = identifier.removesuffix("()").split(".")
    n = len(parts)
    return any(list(path[k : k + n]) == parts for k in range(len(path) - n + 1))


@dataclass(frozen=True)
class Witness:
    """What one passing citation witnesses: its normalized quote, the anchored paths of its
    span (and of the innermost anchor holding its quote), and the module path of its source if
    that is a .py file (None otherwise)."""

    quote: str
    paths: tuple[tuple[str, ...], ...] = ()
    module: tuple[str, ...] | None = None

    def names(self, name: str) -> bool:
        """`name` is covered by this citation's quote as a whole token, or anchored in its paths."""
        return _contains_token(self.quote, name) or any(is_anchored(name, p) for p in self.paths)


def module_prefixed(name: str, witness: Witness) -> bool:
    """The module-prefix witness: `name` is P.N where P is a run of trailing components of the
    citation's module path and N is witnessed by the same citation (`Witness.names`). For
    module dbm.sqlite3, `sqlite3.X` and `dbm.sqlite3.X` qualify; `db.sqlite3.X` does not."""
    module = witness.module
    if not module:
        return False
    parts = name.split(".")
    for s in range(1, min(len(parts), len(module) + 1)):
        if tuple(parts[:s]) == module[len(module) - s :] and witness.names(".".join(parts[s:])):
            return True
    return False


def missing_identifiers(claim_text: str, covered: str, paths: Sequence[tuple[str, ...]], witnesses: Sequence[Witness] = ()) -> list[str]:
    """The IDENTIFIERS of `claim_text` (normalized) not witnessed, in claim order without
    repeats. An identifier is witnessed if `covered` holds it as a whole token, it is anchored
    in one of `paths` (the anchored paths of the passing citations), or a passing citation's
    module path prefixes it (`module_prefixed`)."""
    missing = []
    for kind, token in value_tokens_typed(claim_text):
        if kind != "identifier":
            continue
        name = normalize_for_match(token).removesuffix("()")
        if _contains_token(covered, name) or any(is_anchored(name, path) for path in paths) or any(module_prefixed(name, w) for w in witnesses):
            continue
        missing.append(name)
    return list(dict.fromkeys(missing))


def identifiers_witnessed(claim_text: str, covered: str, paths: Sequence[tuple[str, ...]], witnesses: Sequence[Witness] = ()) -> bool:
    """Every IDENTIFIER of `claim_text` is witnessed (`missing_identifiers` is empty)."""
    return not missing_identifiers(claim_text, covered, paths, witnesses)


def path_suffixes(path: str) -> list[str]:
    """`path` and each suffix of it made of whole components ending in the file name, longest
    first: Lib/re/_constants.py, re/_constants.py, _constants.py."""
    parts = [p for p in path.split("/") if p]
    return ["/".join(parts[k:]) for k in range(len(parts))]


def strip_source_paths(text: str, paths: Sequence[str]) -> str:
    """`text` with every occurrence of each path in `paths`, or of a whole-component suffix of
    it (`path_suffixes`), replaced by a space: backticked (`Lib/x.py`) or bare, never glued to
    a word character or a path character on either side (so `other_x.py` and `x.pyc` stay)."""
    suffixes = sorted({normalize_for_match(s) for p in paths for s in path_suffixes(p)}, key=lambda s: (-len(s), s))
    for suffix in suffixes:
        bare = rf"(?<![\w/.]){re.escape(suffix)}(?!\w|[/.]\w)"
        text = re.sub(rf"`{re.escape(suffix)}`|{bare}", " ", text)
    return text


def _within(inner, outer) -> bool:
    return outer.start <= inner.start <= inner.end <= outer.end


def innermost_anchor(state: WindowState, span, span_text: str, quote: str, blobs: BlobStore):
    """The innermost registered Python-anchored span lying inside `span` (same source version)
    whose text contains the normalized `quote`, if `quote` occurs exactly once in `span_text`
    (the span's normalized text); else None. Nested symbols form a chain, so it is unique."""
    if span_text.count(quote) != 1:
        return None
    inner = span
    for other in state.spans.values():
        a = other.anchor
        if (
            isinstance(a, PyAnchor)
            and other.source_hash == span.source_hash
            and other.span_id != inner.span_id
            and _within(a, inner.anchor)
            and quote in normalize_for_match(blobs.get(other.text_hash).decode("utf-8"))
        ):
            inner = other
    return None if inner is span else inner


@dataclass(frozen=True)
class Passing:
    """One passing citation (BUILD-V3): its span ref, its source's uri, the quote as the model
    wrote it, and what it witnesses."""

    ref: str
    uri: str
    quote: str
    witness: Witness


def passing_citations(claim: Claim, state: WindowState, blobs: BlobStore) -> tuple[tuple[str, ...], list[Passing]]:
    """(the verbatim span refs `claim` cites directly, sorted; its passing citations, in claim
    order). A citation passes when it cites a span verbatim in the window with a quote that is
    a substring of the span text after normalize_for_match and at least
    min(20, len(normalized span text)) chars. `tier1` and the answer gate read the same list."""
    verbatim_cited: list[str] = []
    passing: list[Passing] = []
    for citation in claim.citations:
        ref = citation.ref
        if ref_kind_of(ref) != "span" or kappa(state, ref) is not Knowledge.VERBATIM:
            continue
        verbatim_cited.append(ref)
        if not citation.quote:
            continue
        span = state.spans[ref]
        span_text = normalize_for_match(blobs.get(span.text_hash).decode("utf-8"))
        quote = normalize_for_match(citation.quote)
        if quote and quote in span_text and len(quote) >= min(_MIN_QUOTE, len(span_text)):
            paths: list[tuple[str, ...]] = []
            if isinstance(span.anchor, PyAnchor):
                paths.append(anchored_path(span.anchor))
                inner = innermost_anchor(state, span, span_text, quote, blobs)
                if inner is not None:
                    paths.append(anchored_path(inner.anchor))
            source = state.sources.get(span.source_id)
            uri = source.uri if source is not None else ""
            passing.append(Passing(ref, uri, citation.quote, Witness(quote, tuple(paths), tuple(module_parts(uri)) if uri.endswith(".py") else None)))
    return tuple(sorted(set(verbatim_cited))), passing


def tier1(claim: Claim, state: WindowState, blobs: BlobStore) -> Verdict:
    """Check every direct citation of a verbatim span: its quote is required, must be a
    substring of the span text after normalize_for_match, and must be at least
    min(20, len(normalized span text)) chars (`passing_citations`). Then every VALUE of the
    claim must be covered by the normalized concatenation of the passing quotes
    (`missing_values`), and every IDENTIFIER covered by them, anchored in a passing citation's
    anchored path or in that of the innermost anchor holding its quote (`innermost_anchor`),
    or module-prefixed by a passing citation's module (`missing_identifiers`). Both checks run
    on the claim text after the passing citations' source paths are removed
    (`strip_source_paths`).

    Pass only if at least one citation passes and both hold. On failure the action is
    "refetch" with the cited verbatim spans as support, and value_not_covered or
    identifier_not_anchored carry the offending tokens as `detail`; the caller decides
    abstention."""
    refetch, found = passing_citations(claim, state, blobs)
    passing = [p.ref for p in found]
    witnesses = [p.witness for p in found]
    sources = [p.uri for p in found]
    if not passing:
        return Verdict(1, False, "refetch", "no_valid_quote", refetch)
    text = strip_source_paths(normalize_for_match(claim.text), sources)
    covered = " ".join(w.quote for w in witnesses)
    values = missing_values(text, covered)
    if values:
        return Verdict(1, False, "refetch", "value_not_covered", refetch, detail=tuple(values))
    names = missing_identifiers(text, covered, [p for w in witnesses for p in w.paths], witnesses)
    if names:
        return Verdict(1, False, "refetch", "identifier_not_anchored", refetch, detail=tuple(names))
    return Verdict(1, True, "accept", "quotes_verified", tuple(sorted(set(passing))))
