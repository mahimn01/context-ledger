"""The answer gate (BUILD-V3): part of L's egress, after Tier 0 and Tier 1 per claim.

Pilot v2c: all 11 of L's errors passed Tier 1. Each quote was verbatim in its cited span, but
the span was about another entity than the one the question asked (`stat.ST_ATIME` answered
from xml.dom's NOT_SUPPORTED_ERR = 6), or the cited note had been updated later. Tier 1 checks
a claim against its quotes; nothing tied the answer to the question. The gate does, from the
question TEXT only (never a gold or metadata field of the question), exactly and never by
look-alike:

1. Staleness, per claim. A cited note span ("note:<pos>") is stale when a note registered after
   it in the ledger contains the exact service token and attribute phrase of a pair it states
   ("the <attribute> of service <service>"): the note was updated. The answer is refused with
   stale_superseded, naming the latest such note. (A code span whose source changed is stale by
   kappa already: Tier 0 makes the claim needs_refetch and the client routes it to ctx_refetch.)
2. The asked target (`asked_target`), parsed from the question text:
   - code: every backticked span that fully matches tier1.IDENTIFIER; the QUALIFIED target is
     the one with the most dotted components (then the longest); the others (a parameter name)
     are secondary targets;
   - session: "the current <attribute> of service <service>?" gives (attribute, service).
   No target: the gate passes vacuously (no_target).
3. Anchoring. Among the passing citations of the answer's claims (tier1.passing_citations), the
   WITNESSING ones name the target:
   - a code target T = M.N...: T is a contiguous run of whole components of the citation's
     anchored path (its span's, or the innermost anchor's holding its quote), or the source's
     module parts are a proper prefix of T's components and the rest of T occurs in the quote
     as a whole token (`SRE_FLAG_DOTALL = 16` from Lib/re/_constants.py witnesses
     `re._constants.SRE_FLAG_DOTALL`). A bare target must be the last component of an anchored
     path, or occur in the quote in a def, class or assignment position. A secondary target
     must occur as a whole token in a witnessing quote.
   - a session target: the cited span is a note whose text holds the exact service token
     (letters, digits, "-" and "_" are token characters) and the exact attribute phrase (word
     boundaries).
   Whole token means not glued to [A-Za-z0-9_] (to [A-Za-z0-9_-] for a service): so
   INVALID_STATE_ERR never witnesses INALID_STATE_ERR, MINUS never witnesses ENDMARKER, xml/dom
   never witnesses stat, sadr-routerd never witnesses sadr-router and shard label never
   witnesses shard key. No witnessing citation: target_not_witnessed.
4. The value. Every value token of the answer must be covered by a witnessing quote (each token
   by at least one of them; all of them are about the target): tier1.value_tokens_typed on the
   answer, numbers compared canonically and anything else as a whole token, or the answer's bare
   words when that finds none (`qc46r`, `None`); for a raises question its exception class
   names, bare; for a locate question the answer's module path must equal the module parts of a
   witnessing citation's source. On a multi-hop question ("Resolve any named constant to its
   literal value") the value may instead be covered by the quote of a second passing citation
   that contains, as a whole token, a name the witnessing quote gives the asked parameter or
   constant as its default or value (the hop). Otherwise value_not_with_target.

The gate is pure: it reads the question text, the answer, the claims and the window. It is part
of L's mechanism and never the grader.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..anchor.python import module_parts
from ..anchor.text import normalize_for_match
from ..model import Claim
from ..store.blobs import BlobStore
from ..window import WindowState, ref_kind_of
from .tier1 import IDENTIFIER, Passing, is_anchored, numeric_value, numeric_values, passing_citations, value_tokens_typed

MULTIHOP = "Resolve any named constant to its literal value"
NOTE = "note:"
_BACKTICKED = re.compile(r"`([^`\n]+)`")
_SESSION = re.compile(r"\bthe current (?P<attribute>[a-z][a-z ]*?) of service (?P<service>[A-Za-z0-9_-]+)\?")
_NOTE_PAIR = re.compile(r"\bthe (?P<attribute>[a-z][a-z ]*?) of service (?P<service>[A-Za-z0-9_-]+)")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


@dataclass(frozen=True)
class Target:
    """What a question asks about. `kind` is "code" or "session"; `name` the qualified code
    target or the service; `secondary` the other identifiers a code question names (a
    parameter); `attribute` a session question's attribute phrase; `question` the shape of a
    code question as its text states it: "value", "multihop", "raises" or "locate"."""

    kind: str
    name: str
    secondary: tuple[str, ...] = ()
    attribute: str | None = None
    question: str = "value"

    def label(self) -> str:
        if self.kind == "session":
            return f"the {self.attribute} of service {self.name}"
        return self.name + "".join(f" parameter {p}" for p in self.secondary)


@dataclass(frozen=True)
class GateVerdict:
    """The gate's verdict on one answer. `detail`: the targets not witnessed, or the answer
    tokens no witnessing quote covers; `stale`: (cited note span, the latest note span updating
    it) pairs; `witnesses`: the witnessing span refs of an anchored answer."""

    passed: bool
    reason: str
    target: str | None = None
    detail: tuple[str, ...] = ()
    stale: tuple[tuple[str, str], ...] = ()
    witnesses: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d: dict = {"passed": self.passed, "reason": self.reason, "target": self.target}
        if self.detail:
            d["detail"] = list(self.detail)
        if self.stale:
            d["stale"] = [list(pair) for pair in self.stale]
        if self.witnesses:
            d["witnesses"] = list(self.witnesses)
        return d


def asked_target(question_text: str) -> Target | None:
    """The target of a question, from its text alone (module docstring, item 2), or None."""
    session = _SESSION.search(question_text)
    if session is not None:
        return Target("session", session["service"], attribute=session["attribute"], question="session")
    names = list(dict.fromkeys(n.removesuffix("()") for n in _BACKTICKED.findall(question_text) if IDENTIFIER.fullmatch(n)))
    if not names:
        return None
    qualified = max(names, key=lambda n: (n.count("."), len(n)))
    if "Which module defines" in question_text:
        shape = "locate"
    elif "Which exception classes" in question_text:
        shape = "raises"
    else:
        shape = "multihop" if MULTIHOP in question_text else "value"
    return Target("code", qualified, tuple(n for n in names if n != qualified), question=shape)


def whole(text: str, token: str, chars: str = "A-Za-z0-9_") -> bool:
    """`token` occurs in `text` not glued to a character of the class `chars`."""
    return re.search(rf"(?<![{chars}]){re.escape(token)}(?![{chars}])", text) is not None


def names_service(text: str, service: str, attribute: str) -> bool:
    """`text` holds the exact service token and the exact attribute phrase."""
    return whole(text, service, "A-Za-z0-9_-") and whole(text, attribute)


def _defines(quote: str, name: str) -> bool:
    """`name` in a def, class or assignment position of a line of the quote as written."""
    n = re.escape(name)
    for line in quote.splitlines():
        if re.search(rf"(?<![A-Za-z0-9_])(?:def|class)\s+{n}(?![A-Za-z0-9_])", line):
            return True
        if re.match(rf"\s*{n}\s*(?::[^=]*)?=(?!=)", line) or re.search(rf";\s*{n}\s*(?::[^=]*)?=(?!=)", line):
            return True
    return False


def witnesses_code(name: str, p: Passing) -> bool:
    """Passing citation `p` witnesses code target `name` (module docstring, item 3)."""
    w = p.witness
    parts = name.split(".")
    if len(parts) == 1:
        return any(path and path[-1] == name for path in w.paths) or _defines(p.quote, name)
    if any(is_anchored(name, path) for path in w.paths):
        return True
    module = w.module
    if module and len(module) < len(parts) and tuple(parts[: len(module)]) == module:
        return whole(w.quote, ".".join(parts[len(module) :]))
    return False


def witnesses_session(target: Target, p: Passing, state: WindowState, blobs: BlobStore) -> bool:
    """Passing citation `p` cites a note whose text names the target's service and attribute."""
    text = normalize_for_match(blobs.get(state.spans[p.ref].text_hash).decode("utf-8"))
    return p.uri.startswith(NOTE) and names_service(text, target.name, target.attribute or "")


def superseding_note(span_id: str, state: WindowState, blobs: BlobStore) -> str | None:
    """The span of the latest note registered after `span_id`'s note that names a pair (service
    token and attribute phrase) the note states; None if `span_id` is no note or nothing updates
    it. Registration order is the order of `state.sources` (replay keeps the ledger's order)."""
    span = state.spans.get(span_id)
    source = state.sources.get(span.source_id) if span is not None else None
    if source is None or not source.uri.startswith(NOTE):
        return None
    text = normalize_for_match(blobs.get(source.content_hash).decode("utf-8"))
    pairs = [(m["service"], m["attribute"]) for m in _NOTE_PAIR.finditer(text)]
    order = list(state.sources)
    newest = None
    for source_id in order[order.index(source.source_id) + 1 :]:
        other = state.sources[source_id]
        if other.uri.startswith(NOTE):
            later = normalize_for_match(blobs.get(other.content_hash).decode("utf-8"))
            if any(names_service(later, service, attribute) for service, attribute in pairs):
                newest = other
    if newest is None:
        return None
    return next((s.span_id for s in state.spans.values() if s.source_id == newest.source_id), newest.uri)


def stale_citations(claims: Sequence[Claim], state: WindowState, blobs: BlobStore) -> tuple[tuple[str, str], ...]:
    """(cited note span, the latest note span updating it) for every stale note the claims cite."""
    pairs = []
    for claim in claims:
        for citation in claim.citations:
            if ref_kind_of(citation.ref) == "span" and citation.ref in state.spans:
                newer = superseding_note(citation.ref, state, blobs)
                if newer is not None:
                    pairs.append((citation.ref, newer))
    return tuple(dict.fromkeys(pairs))


def answer_text(answer: object) -> str:
    """The answer as text: a list comma-separated (v1 answers), None empty."""
    if isinstance(answer, (list, tuple)):
        return ", ".join(str(a) for a in answer)
    return "" if answer is None else str(answer)


def value_tokens(answer: str, target: Target) -> list[tuple[str, str]]:
    """(kind, token) the witnessing quotes must cover: "number" or "exact" (module docstring,
    item 4). Never used for a locate question."""
    if target.question == "raises":
        names = [n.rsplit(".", 1)[-1] for n in _NAME.findall(answer) if n != "and"]
        return [("exact", n) for n in dict.fromkeys(names)]
    typed = value_tokens_typed(answer)
    if typed:
        return [("number", t) if k == "number" else ("exact", t.removesuffix("()")) for k, t in typed]
    return [("exact", w) for w in dict.fromkeys(_WORD.findall(answer))]


def covers(quote: str, kind: str, token: str) -> bool:
    """A normalized quote covers a token: a number by canonical value, anything else as a whole token."""
    if kind == "number":
        return numeric_value(token) in numeric_values(quote)
    return whole(quote, normalize_for_match(token))


def hop_names(quote: str, target: Target) -> set[str]:
    """The names a quote gives the asked parameter (or, with none, the asked constant) as its
    default or value: `p=NAME`, `X = M.NAME`; each dotted name and its last component."""
    subjects = target.secondary or (target.name.rsplit(".", 1)[-1],)
    found: set[str] = set()
    for subject in subjects:
        for m in re.finditer(rf"(?<![A-Za-z0-9_.]){re.escape(subject)}\s*(?::[^=,()]*)?=(?!=)\s*([A-Za-z_][A-Za-z0-9_.]*)", quote):
            found |= {m.group(1), m.group(1).rsplit(".", 1)[-1]}
    return found


def answer_module(answer: str) -> tuple[str, ...]:
    """A locate answer as module parts: "Lib/a/b.py", "a/b.py" and "a.b" give (a, b)."""
    text = answer.strip().strip("`'\"")
    text = text[4:] if text.startswith("Lib/") else text
    text = text[:-3] if text.endswith(".py") else text
    parts = [p for p in re.split(r"[/.]", text) if p]
    return tuple(parts[:-1] if parts and parts[-1] == "__init__" else parts)


def answer_gate(question_text: str, answer: object, claims: Sequence[Claim], state: WindowState, blobs: BlobStore) -> GateVerdict:
    """The gate's verdict on `answer` with the claims listed for it (module docstring)."""
    target = asked_target(question_text)
    label = target.label() if target is not None else None
    stale = stale_citations(claims, state, blobs)
    if stale:
        return GateVerdict(False, "stale_superseded", label, stale=stale)
    if target is None:
        return GateVerdict(True, "no_target")
    passing = [p for claim in claims for p in passing_citations(claim, state, blobs)[1]]
    if target.kind == "session":
        witnessing = [p for p in passing if witnesses_session(target, p, state, blobs)]
    else:
        witnessing = [p for p in passing if witnesses_code(target.name, p)]
    if not witnessing:
        return GateVerdict(False, "target_not_witnessed", label, detail=(target.name,))
    unnamed = [n for n in target.secondary if not any(whole(p.witness.quote, n) for p in witnessing)]
    if unnamed:
        return GateVerdict(False, "target_not_witnessed", label, detail=tuple(unnamed))
    text = answer_text(answer)
    if target.question == "locate":
        module = answer_module(text)
        if not any(p.uri.endswith(".py") and tuple(module_parts(p.uri)) == module for p in witnessing):
            return GateVerdict(False, "value_not_with_target", label, detail=(text,))
    else:
        tokens = value_tokens(text, target)
        missing = [t for k, t in tokens if not any(covers(p.witness.quote, k, t) for p in witnessing)]
        if missing and target.question == "multihop" and _hop_covers(tokens, witnessing, passing, target):
            missing = []
        if missing:
            return GateVerdict(False, "value_not_with_target", label, detail=tuple(dict.fromkeys(missing)))
    return GateVerdict(True, "anchored", label, witnesses=tuple(dict.fromkeys(p.ref for p in witnessing)))


def _hop_covers(tokens: list[tuple[str, str]], witnessing: list[Passing], passing: list[Passing], target: Target) -> bool:
    """A second passing citation names a hop of a witnessing quote and covers every token."""
    for first in witnessing:
        names = hop_names(first.witness.quote, target)
        for second in passing:
            if second is not first and any(whole(second.witness.quote, n) for n in names) and all(covers(second.witness.quote, k, t) for k, t in tokens):
                return True
    return False


__all__ = ["MULTIHOP", "GateVerdict", "Target", "answer_gate", "asked_target", "stale_citations", "superseding_note", "whole"]
