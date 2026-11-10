"""Questions v2 (BUILD-V2 item 6): harder, longer episodes, generated from the PERTURBED source.
Version v1 (questions.py) is unchanged and stays the default for old corpus ids.

Templates. v1's default, constant, raises and locate, plus the near-miss unanswerable (a real
name with one edit, absent from every corpus file), and two MULTI-HOP templates, answerable
only by reading two places:

    default_ref   What value does parameter `p` of `mod.f` default to? Resolve any named
                  constant to its literal value.
                  The default is a Name or an Attribute that resolves to a module-level
                  constant with a perturbed literal: in the same module, or (preferred) in
                  another corpus module through `from M import N`, `import M` + `M.N` or a
                  relative import.
    constant_ref  What is the value of the module constant `mod.X`? Resolve any named constant
                  to its literal value.
                  X = Y or X = M.Y (X = Y | Z is out of scope) resolving the same way.

Resolution is static and one hop: the name must be bound exactly once at module level (never
re-bound in an if/try branch, never shadowed by an enclosing class body for a method default),
and the target must be an upper-case constant with a perturbed literal (so memorization cannot
answer; `upstream_answer` is the unperturbed literal). Multi-hop rows carry `gold_paths`,
`gold_lines_all` ([path, start, end] per hop), `hops` = 2 and `cross_file`; `gold_path` and
`gold_lines` are the first hop. Every hop file is in the smallest corpus.

Confusable subjects. v1 required the bare subject to be defined once in the largest corpus.
v2 also keeps subjects defined in 2+ files (the question names the full module path, so it is
unambiguous; the qualname must still be unique in its own file) and records `confusers` =
{rho: number of OTHER definitions of the bare name of the same kind, functions or module
constants, in C-rho}: more material, more look-alikes. Locate questions keep the
unique-definition rule (their answer is the one defining module).

Episodes. Each has Q questions (default 20) from at least 4 gold files. The first half holds the
probed anchors, the multi-hop questions (half of them, rounded up) and half the unanswerables;
the second half the fold probes, the other multi-hop questions and unanswerables, and any
anchors left over. A fold probe is a different question about the same 60-line chunk
(corpus.chunk_lines) as an earlier anchor, at least `probe_gap` positions later (default 8).
The default mix per 20 is 9 anchors, 6 probes, 3 multi-hop and 2 near-miss unanswerables; when
the corpus has fewer multi-hop questions left, anchors top them up. Leak rules: no question
repeats in any episode; within an episode no question text names the module a locate question
answers (`mod.f` gives `mod` away), and no question asks about the constant a multi-hop
question resolves to. A gold file may serve several episodes. Everything is seeded.

Output: runs/corpus/<id>/questions-v2.jsonl (its first line {"metadata": ...}); v1's
questions.jsonl is never touched. Row ids are "v2eNNqMM".

Session facts (BUILD-V2c, question set "v2s"). Facts that live only in the conversation, so
after compaction or folding they must be recalled or recovered through the arm's own mechanism.
A NOTE is a preamble of the user message of question k (row["note"]): "Deployment note: in this
deployment, the retry limit of service orion-relay is 48213." Some notes are SUPERSEDED later,
on another question: "Update: the retry limit of service orion-relay is now 70531." A SESSION
PROBE (template "session") at k + gap, gap from SESSION_GAP (4-12) counted from the first note,
asks "What is the current retry limit of service orion-relay?"; its gold is the latest value
as of the probe (every update of a probed note comes before its probe). An unanswerable probe
asks about a service/attribute pair never noted in the episode. Each episode draws its services
from one confusable family (orion-relay, orion-relayd, orion-proxy) and its attributes from one
or more (retry limit, retry delay, retry budget); values are 4-5 digit integers or short tokens
that occur nowhere in the corpus. The mix keys notes, supersessions, session and
session_unanswerable (default per 20 corpus questions: 6, 2, 4, 2) add to the corpus mix; the
probes are interleaved with the corpus questions (whose positions, ids and fold-probe gaps
follow), and notes go on corpus questions, one per question. The corpus part draws exactly as
v2 does for the same seed. Output: questions-v2s.jsonl; row ids "v2sNNqMM"; a note is shown to
every arm as "[note:<pos>]" then its text, so a session answer cites "note:<pos>".

Questions v3 (BUILD-V3 item 3): the v2s generator unchanged, under its own version label "v3",
row ids "v3eNNqMM" and output file questions-v3.jsonl, so v2s files are never overwritten. For
the same seed and mix it draws exactly the v2s rows. The grid uses --seed 2 (24 episodes on
corpus ca8e01d67009c), so no grid episode is a pilot episode.

Questions v4 (BUILD-V4 item 8): the same generator again under the label "v4", row ids "v4eNNqMM"
and file questions-v4.jsonl, built on the nested extension corpus (eval/extend.py) with a new seed
(3, the CLI's default for v4), so the v4 grid asks fresh draws, with confusers counted at every
level including the extension's, and near-miss names and session values absent from all of it.

Fresh items (BUILD-V4B decision 2). Questions v4 never reuse an item questions v3 asked in a run:
the REFERENCE is questions v3 (the base corpus's questions-v3.jsonl) episodes 1-10 (the v3 grid),
11-12 (the v4 pilot) and 19 (the v3 smoke). An item OVERLAPS a reference item if its target
identifiers (every backticked name of a corpus question; the service and attribute of a session
probe) and its gold answer are the same, or its question text is (lower-cased, whitespace
collapsed). The filter is applied before episodes are drawn, so every episode is complete:
overlapping corpus candidates leave the pools, a near-miss unanswerable that overlaps is drawn
again from the next anchor, and an episode's session probes ask only about service/attribute
pairs whose probe text no reference item asked (its unprobed notes may use any pair). If the
filtered pools support fewer than MIN_FRESH_EPISODES episodes, they get EXTRA targets from every
.py file of the smallest corpus (so every answerable item is answerable at every level), only for
names no questions v3 item asks about in any episode, and in the v3 templates' exact wording (the
gate parses them, the grader grades them): `extra_candidates`. Extras come after the v3
generator's candidates of each file, so they fill what it cannot. Every v4 corpus row records its
`origin`; the metadata records the rule, the reference (file, sha256, episodes), the excluded
counts and whether extras were used. `overlap_report` checks a built file against the reference.
"""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

from ..anchor.python import module_parts
from ..ids import sha256_hex
from . import pyscan
from .corpus import RUNS_CORPUS, chunk_lines, corpus_texts
from .perturb import char_offsets, char_span, perturbable
from .questions import Candidate, QuestionError, _texts, _unanswerable, defined_counts, file_candidates

VERSION = "v2"
FILENAME = "questions-v2.jsonl"
QUESTIONS = 20
PROBE_GAP = 8
MIN_FILES = 4
MULTIHOP = ("default_ref", "constant_ref")
SESSION_VERSION = "v2s"
SESSION_FILENAME = "questions-v2s.jsonl"
SESSION_GAP = (4, 12)
V3_VERSION = "v3"
V3_FILENAME = "questions-v3.jsonl"
V4_VERSION = "v4"
V4_FILENAME = "questions-v4.jsonl"
V4_SEED = 3
FILENAMES = {VERSION: FILENAME, SESSION_VERSION: SESSION_FILENAME, V3_VERSION: V3_FILENAME, V4_VERSION: V4_FILENAME}
QID_PREFIX = {SESSION_VERSION: "v2s", V3_VERSION: "v3e", V4_VERSION: "v4e"}  # the session-fact versions' row ids
SESSION_BASES = ("orion", "vega", "lyra", "rigel", "castor", "deneb", "altair", "mira", "nashira", "electra", "talitha", "alcor", "merak",
                 "phecda", "kochab", "sadr", "wezen", "zosma", "algol", "mintaka")  # fmt: skip
SESSION_SUFFIXES = (("-relay", "-relayd", "-proxy"), ("-gateway", "-gatewayd", "-router"), ("-store", "-stored", "-cache"), ("-broker", "-brokerd", "-bridge"))
SESSION_ATTRIBUTES = (
    (("retry limit", "retry delay", "retry budget"), "int"),
    (("queue depth", "queue size", "queue limit"), "int"),
    (("shard key", "shard tag", "shard label"), "token"),
    (("auth realm", "auth scope", "auth zone"), "token"),
)
FUNCTION_KIND = {"default": "function", "raises": "function", "locate": "function", "default_ref": "function", "constant": "constant", "constant_ref": "constant"}
_PRIORITY = {"default": 0, "constant": 0, "raises": 1, "locate": 2}
V3_REFERENCE = {"grid": tuple(range(1, 11)), "pilot": (11, 12), "smoke": (19,)}  # the questions v3 episodes a run asked (BUILD-V4B decision 2)
MIN_FRESH_EPISODES = 10
OVERLAP_RULE = ("an item overlaps a reference item if its target identifiers (every backticked name of a corpus question; the service and "
                "attribute of a session probe) and its gold answer are the same, or its question text is the same after lower-casing and "
                "collapsing whitespace")  # fmt: skip
EXTRA_RULE = ("extras, only when the filtered pools support fewer than 10 episodes: from every .py file of the smallest corpus, for names "
              "no questions v3 item asks about, `locate` for an unrenamed top-level function (6+ characters, defined once in the largest "
              "corpus) of a file with renamed functions, and `constant` for an upper-case module constant (leading underscores allowed) bound once to a literal the "
              "perturbation did not rewrite (its gold is the source literal; no upstream answer)")  # fmt: skip
ORIGINS = ("generator", "locate_unrenamed", "constant_unperturbed")  # a v4 corpus row's `origin`: the v3 generator's candidate, or an extra


@dataclass(frozen=True)
class Mix:
    """How many of each slot an episode holds: the corpus questions (anchors, probes,
    multi-hop, unanswerable) and, for v2s, the session facts (notes, supersessions, answerable
    and unanswerable session probes)."""

    anchors: int
    probes: int
    multihop: int
    unanswerable: int
    notes: int = 0
    supersessions: int = 0
    session: int = 0
    session_unanswerable: int = 0

    @property
    def total(self) -> int:
        """The corpus questions of an episode (session probes come on top)."""
        return self.anchors + self.probes + self.multihop + self.unanswerable

    @property
    def has_session(self) -> bool:
        return any((self.notes, self.supersessions, self.session, self.session_unanswerable))

    @classmethod
    def default(cls, questions: int = QUESTIONS) -> Mix:
        """9 anchors, 6 probes, 3 multi-hop and 2 unanswerable per 20, scaled to `questions`."""
        probes, multihop, unanswerable = round(questions * 3 / 10), round(questions * 3 / 20), max(1, round(questions / 10))
        return cls(questions - probes - multihop - unanswerable, probes, multihop, unanswerable)

    @classmethod
    def default_session(cls, questions: int = QUESTIONS) -> Mix:
        """`default` plus 6 notes, 2 supersessions, 4 answerable and 2 unanswerable session
        probes per 20 corpus questions, scaled to `questions`."""
        session = max(1, round(questions / 5))
        return replace(cls.default(questions), notes=round(questions * 3 / 10), supersessions=max(1, round(questions / 10)), session=session,
                       session_unanswerable=max(1, round(questions / 10)))  # fmt: skip

    @classmethod
    def parse(cls, text: str) -> Mix:
        """"anchors=9,probes=6,multihop=3,unanswerable=2" (every key, each once), optionally
        with notes=..,supersessions=..,session=..,session_unanswerable=.. (v2s)."""
        parts = dict(p.split("=", 1) for p in text.split(",") if p.strip())
        keys = {"anchors", "probes", "multihop", "unanswerable"}
        optional = {"notes", "supersessions", "session", "session_unanswerable"}
        if not keys <= set(parts) <= keys | optional or not all(v.strip().isdigit() for v in parts.values()):
            raise QuestionError(f"--mix needs exactly {sorted(keys)} as key=count, and optionally {sorted(optional)}")
        return cls(**{k: int(v) for k, v in parts.items()})

    def check(self, questions: int) -> None:
        if self.total != questions:
            raise QuestionError(f"the mix {self} holds {self.total} questions, not {questions}")
        if self.probes > self.anchors or self.anchors < 1:
            raise QuestionError("every probe needs its own anchor, and an episode needs an anchor")
        if self.session > self.notes or self.supersessions > self.session or self.notes > questions:
            raise QuestionError("every session probe needs its own note, every supersession a probed note, and every note a corpus question")

    def to_dict(self) -> dict:
        corpus = {"anchors": self.anchors, "probes": self.probes, "multihop": self.multihop, "unanswerable": self.unanswerable}
        if not self.has_session:
            return corpus
        return {**corpus, "notes": self.notes, "supersessions": self.supersessions, "session": self.session, "session_unanswerable": self.session_unanswerable}


# Static one-hop resolution


@dataclass
class FileInfo:
    tree: ast.Module
    module: str
    bindings: Counter  # how often each name is bound at module level (if/try/with/for bodies included)
    assigns: dict  # name -> its single top-level Assign/AnnAssign (simple Name target)
    imports: dict  # name -> ("import", module) or ("from", module, attr), bound once at top level


def _module_bindings(body: list[ast.stmt], out: Counter) -> None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] += 1
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                out.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            out[node.target.id] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name.split(".")[0]) for a in node.names)
        if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            for name in ("body", "orelse", "finalbody"):
                _module_bindings(getattr(node, name, []) or [], out)
            for handler in getattr(node, "handlers", []) or []:
                _module_bindings(handler.body, out)
            if isinstance(node, ast.For):
                out.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))


class ModuleIndex:
    """The corpus's .py files by module path, parsed lazily, with their module-level bindings."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts
        self.paths = {pyscan.module_path(p): p for p in sorted(texts) if p.endswith(".py")}
        self._info: dict[str, FileInfo | None] = {}

    def info(self, path: str) -> FileInfo | None:
        if path not in self._info:
            tree = pyscan.parse(self.texts[path]) if path.endswith(".py") else None
            self._info[path] = None if tree is None else self._build(path, tree)
        return self._info[path]

    def _build(self, path: str, tree: ast.Module) -> FileInfo:
        bindings: Counter = Counter()
        _module_bindings(tree.body, bindings)
        assigns, imports = {}, {}
        parts = module_parts(path)
        package = parts if path.endswith("__init__.py") else parts[:-1]
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                assigns[node.targets[0].id] = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                assigns[node.target.id] = node
            elif isinstance(node, ast.Import):
                for a in node.names:
                    imports[a.asname or a.name.split(".")[0]] = ("import", a.name if a.asname else a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level - 1 > len(package):
                        continue
                    base = package[: len(package) - (node.level - 1)] + ([node.module] if node.module else [])
                    source = ".".join(base)
                else:
                    source = node.module or ""
                for a in node.names:
                    if a.name != "*":
                        imports[a.asname or a.name] = ("from", source, a.name)
        once = lambda d: {k: v for k, v in d.items() if bindings[k] == 1}  # noqa: E731
        return FileInfo(tree, ".".join(parts), bindings, once(assigns), once(imports))

    def constant(self, path: str, name: str) -> ast.stmt | None:
        """The single module-level literal assignment of upper-case `name` in `path`, or None."""
        info = self.info(path)
        stmt = info.assigns.get(name) if info is not None else None
        if stmt is None or not pyscan.UPPER.match(name) or pyscan.literal_value(stmt.value) is None:
            return None
        return stmt

    def resolve(self, path: str, expr: ast.expr) -> tuple[str, str, ast.stmt] | None:
        """(target path, constant name, its assignment) for a Name or dotted Attribute that
        names a literal module constant one hop away (see the module docstring), else None."""
        info = self.info(path)
        if info is None:
            return None
        chain = []
        while isinstance(expr, ast.Attribute):
            chain.append(expr.attr)
            expr = expr.value
        if not isinstance(expr, ast.Name):
            return None
        head, chain = expr.id, chain[::-1]
        if not chain:
            if head in info.assigns:
                stmt = self.constant(path, head)
                return (path, head, stmt) if stmt is not None else None
            bound = info.imports.get(head)
            if bound is None or bound[0] != "from" or f"{bound[1]}.{bound[2]}" in self.paths or bound[1] not in self.paths:
                return None
            target = self.paths[bound[1]]
            stmt = self.constant(target, bound[2])
            return (target, bound[2], stmt) if stmt is not None else None
        bound = info.imports.get(head)
        if bound is None:
            return None
        base = bound[1] if bound[0] == "import" else f"{bound[1]}.{bound[2]}"
        module = ".".join([base, *chain[:-1]])
        if module not in self.paths:
            return None
        target = self.paths[module]
        stmt = self.constant(target, chain[-1])
        return (target, chain[-1], stmt) if stmt is not None else None


def _class_names(tree: ast.Module, qualname: str) -> set[str] | None:
    """Names bound in the bodies of the classes enclosing method `qualname`; None if a class
    cannot be found at module level (then a bare-name default is not resolved)."""
    parts = qualname.split(".")[:-1]
    body, names = tree.body, set()
    for part in parts:
        found = next((n for n in ast.walk(ast.Module(body=body, type_ignores=[])) if isinstance(n, ast.ClassDef) and n.name == part), None)
        if found is None:
            return None
        bound: Counter = Counter()
        _module_bindings(found.body, bound)
        names |= set(bound)
        body = found.body
    return names


@dataclass(frozen=True)
class Hop:
    path: str
    name: str
    lines: tuple[int, int]


@dataclass(frozen=True)
class MultiHop:
    candidate: Candidate
    hop: Hop

    @property
    def cross_file(self) -> bool:
        return self.hop.path != self.candidate.path


def perturbed_record(records: dict):
    """record_of(path, stmt) for multihop_candidates: the perturbation record of the constant
    `stmt` assigns, if its current literal is the perturbed value; `records` maps (path, line)
    to constant records."""

    def record_of(path: str, stmt: ast.stmt) -> dict | None:
        record = records.get((path, stmt.value.lineno))
        if record is None:
            return None
        value = pyscan.literal_value(stmt.value)
        return record if value is not None and value[1] == ast.literal_eval(record["new"]) else None

    return record_of


def perturbable_record(index: ModuleIndex):
    """record_of(path, stmt) on UNPERTURBED text: a stand-in record (new = old = the source) for a
    constant whose literal the perturbation would rewrite. No perturbed value is used."""

    def record_of(path: str, stmt: ast.stmt) -> dict | None:
        text = index.texts[path]
        starts, lines = char_offsets(text)
        node = stmt.value
        start, end = char_span(starts, lines, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
        kind = pyscan.literal_value(node)
        return {"new": text[start:end], "old": text[start:end]} if kind and perturbable(kind[0], text[start:end]) else None

    return record_of


def multihop_candidates(path: str, index: ModuleIndex, record_of, small: set[str] | None) -> list[MultiHop]:
    """Every default_ref and constant_ref question of gold file `path` whose hop file is in
    `small` (the smallest corpus; None: anywhere); `record_of(path, stmt)` gives the target
    constant's perturbation record or None (perturbed_record, perturbable_record)."""
    info = index.info(path)
    if info is None:
        return []
    module = info.module
    out = []
    funcs = pyscan.functions(info.tree)
    qualnames = Counter(q for q, _ in funcs)
    for qualname, func in funcs:
        if "<locals>" in qualname or qualnames[qualname] != 1:
            continue
        shadow = _class_names(info.tree, qualname) if "." in qualname else set()
        lines = (min([func.lineno, *(d.lineno for d in func.decorator_list)]), func.end_lineno)
        for param, default in pyscan.parameter_defaults(func):
            if not isinstance(default, (ast.Name, ast.Attribute)) or (isinstance(default, ast.Name) and (shadow is None or default.id in shadow)):
                continue
            found = index.resolve(path, default)
            record = record_of(found[0], found[2]) if found else None
            if record is None or (small is not None and found[0] not in small):
                continue
            text = f"What value does parameter `{param}` of `{module}.{qualname}` default to? Resolve any named constant to its literal value."
            cand = Candidate("default_ref", path, qualname.rsplit(".", 1)[-1], text, record["new"], record["old"], lines, ("default", module, qualname, param))
            out.append(MultiHop(cand, Hop(found[0], found[1], (found[2].lineno, found[2].end_lineno))))
    for name, stmt in sorted(info.assigns.items(), key=lambda kv: kv[1].lineno):
        if not pyscan.UPPER.match(name) or not isinstance(stmt.value, (ast.Name, ast.Attribute)):
            continue
        found = index.resolve(path, stmt.value)
        record = record_of(found[0], found[2]) if found else None
        if record is None or (small is not None and found[0] not in small) or (found[0], found[1]) == (path, name):
            continue
        text = f"What is the value of the module constant `{module}.{name}`? Resolve any named constant to its literal value."
        cand = Candidate("constant_ref", path, name, text, record["new"], record["old"], (stmt.lineno, stmt.end_lineno), ("constant", module, name, None))
        out.append(MultiHop(cand, Hop(found[0], found[1], (found[2].lineno, found[2].end_lineno))))
    return out


def density_v2(path: str, text: str, index: ModuleIndex, defined: Counter, tokens: int) -> tuple[int, Fraction, list[str]]:
    """(multi-hop candidates, v2 question candidates per token, hop files of its cross-file
    multi-hop candidates) of an UNPERTURBED file: v1's density rule (questions.question_density) with confusable subjects,
    plus multi-hop candidates whose target the perturbation would rewrite. No perturbed value,
    gold answer or model output is used."""
    tree = pyscan.parse(text)
    if tree is None or tokens <= 0:
        return 0, Fraction(0), []
    starts, lines = char_offsets(text)
    records = []
    for site in pyscan.literal_sites(tree):
        start, end = char_span(starts, lines, *site.span)
        if perturbable(pyscan.literal_value(site.node)[0], text[start:end]):
            records.append({"kind": site.kind, "line": site.node.lineno, "param": site.param, "old": text[start:end], "new": text[start:end]})
    singles = file_candidates(path, text, records, defined, unique=False)
    multi = multihop_candidates(path, index, perturbable_record(index), None)
    return len(multi), Fraction(len(singles) + len(multi), tokens), list(dict.fromkeys(m.hop.path for m in multi if m.cross_file))


def definition_counts(texts: dict[str, str], paths) -> tuple[Counter, Counter]:
    """(function bare-name counts, module-constant name counts) over the .py files `paths`."""
    functions, constants = Counter(), Counter()
    for path in paths:
        tree = pyscan.parse(texts[path]) if path.endswith(".py") else None
        if tree is None:
            continue
        functions.update(q.rsplit(".", 1)[-1] for q, _ in pyscan.functions(tree))
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            constants.update(t.id for t in targets if isinstance(t, ast.Name))
    return functions, constants


# Questions v4: fresh items (BUILD-V4B decision 2; module docstring)

_BACKTICKED = re.compile(r"`([^`\n]+)`")
_PROBE = re.compile(r"\bthe current (?P<attribute>[a-z][a-z ]*?) of service (?P<service>[A-Za-z0-9_-]+)\?")
_CONSTANT = re.compile(r"^_*[A-Z][A-Z0-9_]*$")  # an upper-case module constant, private ones too


def target_key(text: str) -> tuple:
    """A question's target identifiers: ("session", service, attribute) for a session probe, else
    its backticked names, sorted."""
    m = _PROBE.search(text)
    if m is not None:
        return ("session", m["service"], m["attribute"])
    return tuple(sorted(set(_BACKTICKED.findall(text))))


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _answer_key(answer: object) -> str:
    return json.dumps(answer, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class Overlap:
    """The reference items' keys (OVERLAP_RULE) and every name any questions v3 item asks about
    (`subjects`: an extra target must be none of them)."""

    keys: frozenset
    texts: frozenset
    subjects: frozenset
    reference: dict

    def hit(self, text: str, answer: object) -> bool:
        return (target_key(text), _answer_key(answer)) in self.keys or normalized(text) in self.texts

    def fresh_probe(self, fact: Fact) -> bool:
        """No reference item asked this session probe (its text fixes service and attribute)."""
        return normalized(probe_text(fact)) not in self.texts


def reference_path(manifest: dict, runs_dir: str | Path) -> Path:
    """The questions v3 file the overlap filter reads: the corpus's own, else its base's."""
    for cid in dict.fromkeys([manifest["id"], manifest.get("base")]):
        if cid and (Path(runs_dir) / cid / V3_FILENAME).exists():
            return Path(runs_dir) / cid / V3_FILENAME
    raise QuestionError(f"questions v4 need questions v3 of corpus {manifest['id']} (or its base) as the overlap reference; none found")


def question_rows(data: bytes) -> list[dict]:
    return [r for r in (json.loads(line) for line in data.decode("utf-8").splitlines()) if "metadata" not in r]


def load_overlap(manifest: dict, runs_dir: str | Path = RUNS_CORPUS) -> Overlap:
    path = reference_path(manifest, runs_dir)
    data = path.read_bytes()
    rows = question_rows(data)
    episodes = sorted({e for eps in V3_REFERENCE.values() for e in eps})
    ref = [r for r in rows if r["episode"] in episodes]
    subjects: set[str] = set()
    for r in rows:  # every v3 episode, not just the reference's
        if r["template"] != "session":
            subjects |= set(_BACKTICKED.findall(r["text"]))
            if r["template"] == "locate" and r["answerable"]:
                subjects.add(f"{r['answer']}.{r['subject']}")
    present = {r["episode"] for r in rows}
    reference = {"file": f"{path.parent.name}/{path.name}", "sha256": sha256_hex(data), "episodes": {k: list(v) for k, v in V3_REFERENCE.items()},
                 "items": len(ref), "episodes_missing": [e for e in episodes if e not in present]}  # fmt: skip
    return Overlap(frozenset((target_key(r["text"]), _answer_key(r["answer"])) for r in ref), frozenset(normalized(r["text"]) for r in ref), frozenset(subjects), reference)


def _owner_names(c: Candidate) -> tuple[str, ...]:
    """The names a candidate asks about, as questions v3 items name them."""
    _, module, qualname, _ = c.near
    return (c.subject, f"{c.answer}.{c.subject}") if c.template == "locate" else (f"{module}.{qualname}",)


def extra_candidates(path: str, text: str, records: list[dict], defined: Counter, index: ModuleIndex) -> list[Candidate]:
    """Questions v4's extra targets of a perturbed smallest-corpus file (EXTRA_RULE), in the v3
    templates' exact wording; `origin` says which kind."""
    tree = pyscan.parse(text)
    if tree is None:
        return []
    module = pyscan.module_path(path)
    renamed = {r["new"] for r in records if r["kind"] == "rename"}
    perturbed = {r["line"] for r in records if r["kind"] == "constant"}
    funcs = pyscan.functions(tree)
    local = Counter(q for q, _ in funcs)
    out = []
    if renamed:  # the v3 generator asks locate here only about the renamed functions
        for qualname, func in funcs:
            if "." in qualname or qualname in renamed or len(qualname) < 6 or defined[qualname] != 1 or local[qualname] != 1:
                continue
            lines = (min([func.lineno, *(d.lineno for d in func.decorator_list)]), func.end_lineno)
            out.append(Candidate("locate", path, qualname, _texts("locate", module, qualname, None), module, None, lines, ("locate", module, qualname, None),
                                 "locate_unrenamed"))  # fmt: skip
    starts, lines_ = char_offsets(text)
    info = index.info(path)
    for name, stmt in sorted(info.assigns.items(), key=lambda kv: kv[1].lineno):  # bound once at module level
        if not _CONSTANT.match(name) or pyscan.literal_value(stmt.value) is None or stmt.value.lineno in perturbed:
            continue
        node = stmt.value
        start, end = char_span(starts, lines_, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
        out.append(Candidate("constant", path, name, _texts("constant", module, name, None), text[start:end], None, (stmt.lineno, stmt.end_lineno),
                             ("constant", module, name, None), "constant_unperturbed"))  # fmt: skip
    return out


@dataclass
class Pool:
    """What one gold file still has to give: (anchor, probe) pairs on one chunk, single anchors
    and multi-hop questions, each in seeded order."""

    pairs: list = field(default_factory=list)
    singles: list = field(default_factory=list)
    multi: list = field(default_factory=list)


def _chunk_pairs(cands: list[Candidate], path: str, text: str) -> tuple[list[tuple[Candidate, Candidate]], list[Candidate]]:
    """Pairs of different questions on the same chunk (locate only with locate), in the given
    order; the rest are single anchors."""
    chunks = chunk_lines(path, text)
    chunk_of = lambda c: next((k for k, (s, e) in enumerate(chunks) if s <= c.lines[0] <= e), -1)  # noqa: E731
    pairs, used = [], set()
    for a in cands:
        if id(a) in used:
            continue
        for p in cands:
            if id(p) in used or p is a or p.text == a.text or (p.template == "locate") != (a.template == "locate") or chunk_of(p) != chunk_of(a):
                continue
            pairs.append((a, p))
            used.update({id(a), id(p)})
            break
    return pairs, [c for c in cands if id(c) not in used]


@dataclass(frozen=True)
class Q:
    """A chosen question: its candidate, slot kind ("anchor", "probe", "multihop",
    "unanswerable"), the anchor it probes, and its second hop."""

    cand: Candidate | None
    kind: str
    row: dict | None = None  # an unanswerable's row
    anchor: Candidate | None = None
    hop: Hop | None = None

    @property
    def text(self) -> str:
        return self.cand.text if self.cand is not None else self.row["text"]


def _module_named(text: str, module: str) -> bool:
    return re.search(rf"`{re.escape(module)}\.", text) is not None


def leaks(chosen: list[Q]) -> bool:
    """A locate answer's module named in another question, or a multi-hop target asked about."""
    for q in chosen:
        if q.cand is not None and q.cand.template == "locate" and any(o is not q and _module_named(o.text, str(q.cand.answer)) for o in chosen):
            return True
        if q.hop is not None:
            for o in chosen:
                if o is q or o.cand is None:
                    continue
                if (o.cand.template in ("constant", "constant_ref") and (o.cand.path, o.cand.subject) == (q.hop.path, q.hop.name)) or (o.hop == q.hop):
                    return True
    return False


def layout(mix: Mix, questions: int, probe_gap: int, rng: random.Random) -> list[str]:
    """Slot kinds by position: "Ap" (a probed anchor), "A", "P", "M", "U" (see the module
    docstring); probes are matched in order to probed anchors at least `probe_gap` earlier."""
    half = questions // 2
    u1, m1 = (mix.unanswerable + 1) // 2, (mix.multihop + 1) // 2
    first = ["Ap"] * mix.probes + ["U"] * u1 + ["M"] * m1
    if len(first) > half:
        raise QuestionError(f"the mix {mix.to_dict()} does not fit: {len(first)} first-half slots in a half of {half}")
    left = {"A": mix.anchors - mix.probes, "M": mix.multihop - m1, "U": mix.unanswerable - u1}
    for kind in ("A", "M", "U"):  # fill the first half: anchors, then multi-hop, then unanswerables
        take = min(half - len(first), left[kind])
        first += [kind] * take
        left[kind] -= take
    second = ["P"] * mix.probes + ["M"] * left["M"] + ["U"] * left["U"] + ["A"] * left["A"]

    def ok(slots: list[str]) -> bool:
        anchors = [n for n, k in enumerate(slots) if k == "Ap"]
        probes = [n for n, k in enumerate(slots) if k == "P"]
        return all(p - a >= probe_gap for a, p in zip(anchors, probes))

    for _ in range(50):
        a, b = list(first), list(second)
        rng.shuffle(a)
        rng.shuffle(b)
        if ok(a + b):
            return a + b
    fallback = sorted(first, key=lambda k: k != "Ap") + sorted(second, key=lambda k: k == "P")
    if not ok(fallback):
        raise QuestionError(f"a probe gap of {probe_gap} cannot be met in episodes of {questions} questions")
    return fallback


@dataclass
class Build:
    rows: list[dict]
    candidates: Counter  # per template, before episodes are drawn
    confusers: dict  # rho -> Counter of confuser counts over the candidates
    pools: dict
    excluded: Counter = field(default_factory=Counter)  # questions v4: what the overlap filter kept out, by kind


def _confusers(cand: Candidate, counts: dict) -> dict:
    """{rho: other definitions of the subject's bare name, of its kind, in C-rho}."""
    kind = FUNCTION_KIND[cand.template]
    return {str(rho): max(0, (functions if kind == "function" else constants)[cand.subject] - 1) for rho, (functions, constants) in counts.items()}


def build_v2(manifest: dict, *, seed: int, episodes: int | None, questions: int = QUESTIONS, mix: Mix | None = None,
             probe_gap: int = PROBE_GAP, runs_dir: str | Path = RUNS_CORPUS, version: str = SESSION_VERSION, overlap: Overlap | None = None,
             extras: bool = False) -> Build:  # fmt: skip
    """Draw up to `episodes` episodes (None: as many as the corpus supports) in memory; a mix
    with session facts labels its rows `version` ("v2s" or "v3"). With `overlap` (questions v4),
    nothing that overlaps the reference is drawn, every .py file of the smallest corpus is a
    gold file, and `extras` adds extra_candidates (module docstring)."""
    mix = mix or Mix.default(questions)
    mix.check(questions)
    texts = corpus_texts(manifest)
    records_path = Path(runs_dir) / manifest["id"] / "perturbations.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()] if records_path.exists() else []
    by_line = {(r["path"], r["line"]): r for r in records if r["kind"] == "constant"}
    defined = defined_counts(texts)
    rhos = sorted(manifest["rhos"])
    member_rho = {f["path"]: f["rho"] for f in manifest["files"]}
    small = {p for p, r in member_rho.items() if r == rhos[0]}
    excluded = set(manifest.get("excluded", []))
    gold = [p for p in manifest["gold"] if p in small and p not in excluded]
    if overlap is not None:
        gold += sorted(p for p in small if p.endswith(".py") and p not in excluded and p not in set(gold))
    counts = {rho: definition_counts(texts, [p for p, r in member_rho.items() if r <= rho]) for rho in rhos}
    index = ModuleIndex(texts)
    rng = random.Random(f"questions-v2:{seed}")
    pools: dict[str, Pool] = {}
    tally: Counter = Counter()
    kept_out: Counter = Counter()
    confusers: dict = {str(rho): Counter() for rho in rhos}
    for path in gold:
        file_records = [r for r in records if r["path"] == path]
        singles = file_candidates(path, texts[path], file_records, defined, unique=False)
        multi = multihop_candidates(path, index, perturbed_record(by_line), small)
        if overlap is not None:
            singles = [replace(c, origin="generator") for c in singles]
            multi = [MultiHop(replace(m.candidate, origin="generator"), m.hop) for m in multi]
            if extras:
                extra = extra_candidates(path, texts[path], file_records, defined, index)
                fresh = [c for c in extra if not overlap.subjects.intersection(_owner_names(c))]
                kept_out["extra_named_by_v3"] += len(extra) - len(fresh)
                singles += fresh
            kept_out.update(f"overlap_{c.template}" for c in singles + [m.candidate for m in multi] if overlap.hit(c.text, c.answer))
            singles = [c for c in singles if not overlap.hit(c.text, c.answer)]
            multi = [m for m in multi if not overlap.hit(m.candidate.text, m.candidate.answer)]
        rng.shuffle(singles)
        singles.sort(key=lambda c: (c.origin not in ("", "generator"), _PRIORITY[c.template]))  # stable: perturbed literals first, they measure leakage; v4's extras last
        rng.shuffle(multi)
        multi.sort(key=lambda m: not m.cross_file)  # cross-file hops first
        pairs, rest = _chunk_pairs(singles, path, texts[path])
        if pairs or rest or multi:
            pools[path] = Pool(pairs, rest, multi)
        for c in singles + [m.candidate for m in multi]:
            tally[c.template] += 1
            for rho, n in _confusers(c, counts).items():
                confusers[rho][n] += 1
    files = sorted(pools)
    rng.shuffle(files)
    haystack = list(texts.values())
    used_texts: set[str] = set()
    rows: list[dict] = []
    episode = 0
    # session facts draw from their own seeded streams, so the corpus part is v2's for the seed
    bases = random.Random(f"questions-v2s:{seed}").sample(SESSION_BASES, len(SESSION_BASES)) if mix.has_session else []
    words = corpus_words(haystack) if mix.has_session else set()
    while episodes is None or episode < episodes:
        drawn = _draw(pools, files, episode, mix, rng, haystack, used_texts, overlap, kept_out)
        if drawn is None:
            break
        episode += 1
        made = _rows(episode, drawn, mix, questions, probe_gap, rng, counts)
        if mix.has_session:
            session_rng = random.Random(f"questions-v2s:{seed}:{episode}")
            made = with_session(made, episode, mix, session_rng, bases[(episode - 1) % len(bases)], words, haystack, counts, version, overlap, kept_out)
        rows += made
        used_texts |= {r["text"] for r in rows}
    return Build(rows, tally, {rho: dict(sorted(c.items())) for rho, c in confusers.items()}, pools, kept_out)


def _round_robin(files: list[str], start: int, take, need: int, chosen: list[Q]) -> list[Q] | None:
    """One item per file per turn from `take(path, chosen)` (it returns a Q list or None, and
    consumes nothing) until `need` items are taken; None if the files run out. Each turn visits
    the files with the fewest questions in the episode first (rotated by `start`), so an episode
    spreads over as many gold files as it can."""
    out: list[Q] = []
    rotated = files[start:] + files[:start]
    while len(out) < need:
        added = False
        held = Counter(q.cand.path for q in chosen + out if q.cand is not None)
        for path in sorted(rotated, key=lambda p: held[p]):
            if len(out) >= need:
                break
            item = take(path, chosen + out)
            if item:
                out += item
                added = True
        if not added:
            return None
    return out


def _draw(pools: dict[str, Pool], files: list[str], episode: int, mix: Mix, rng: random.Random, haystack: list[str], used: set[str],
          overlap: Overlap | None = None, kept_out: Counter | None = None):  # fmt: skip
    """The questions of one episode (consumed from the pools), or None (pools unchanged). With
    `overlap`, a near-miss unanswerable that overlaps the reference is not taken (counted in
    `kept_out`)."""
    if not files:
        return None
    start = episode % len(files)
    taken: dict[str, dict[str, list]] = {p: {"pairs": [], "singles": [], "multi": []} for p in files}

    def first(path, kind, chosen, make):
        for item in getattr(pools[path], kind):
            if any(item is t for t in taken[path][kind]):
                continue
            qs = make(item)
            if any(q.text in used or q.text in {c.text for c in chosen} for q in qs) or leaks(chosen + qs):
                continue
            taken[path][kind].append(item)
            return qs
        return None

    pair_q = lambda pr: [Q(pr[0], "anchor"), Q(pr[1], "probe", anchor=pr[0])]  # noqa: E731
    pairs = _round_robin(files, start, lambda p, ch: first(p, "pairs", ch, pair_q), 2 * mix.probes, [])
    if pairs is None:
        return None
    multi: list[Q] = []
    for n in range(mix.multihop, 0, -1):  # as many as the corpus supplies; anchors top up the rest
        for kinds in taken.values():
            kinds["multi"].clear()
        multi = _round_robin(files, start, lambda p, ch: first(p, "multi", ch, lambda m: [Q(m.candidate, "multihop", hop=m.hop)]), n, pairs) or []
        if multi:
            break
    need = mix.anchors - mix.probes + (mix.multihop - len(multi))
    take_single = lambda p, ch: first(p, "singles", ch, lambda c: [Q(c, "anchor")])  # noqa: E731
    if overlap is not None:  # questions v4: a file with no single left may give a pair's anchor (the pair is used up)
        take_single = lambda p, ch: first(p, "singles", ch, lambda c: [Q(c, "anchor")]) or first(p, "pairs", ch, lambda pr: [Q(pr[0], "anchor")])  # noqa: E731
    singles = _round_robin(files, start, take_single, need, pairs + multi)
    if singles is None:
        return None
    chosen = pairs + multi + singles
    if len({q.cand.path for q in chosen}) < MIN_FILES:
        return None
    anchors = [q.cand for q in chosen if q.kind == "anchor"]
    unans: list[Q] = []
    for a in anchors:
        if len(unans) == mix.unanswerable:
            break
        row = _unanswerable(a, rng, haystack)
        if row is None or row["text"] in used or any(row["text"] == q.text for q in chosen + unans):
            continue
        if overlap is not None and overlap.hit(row["text"], None):
            kept_out["overlap_near_miss_draw"] += 1
            continue
        q = Q(None, "unanswerable", row=row)
        if not leaks(chosen + unans + [q]):
            unans.append(q)
    if len(unans) < mix.unanswerable:
        return None
    for path, kinds in taken.items():  # consume
        for kind, items in kinds.items():
            ids = {id(i) for i in items}
            setattr(pools[path], kind, [i for i in getattr(pools[path], kind) if id(i) not in ids])
    return {"pairs": [(pairs[k].cand, pairs[k + 1].cand) for k in range(0, len(pairs), 2)], "multi": multi, "singles": [q.cand for q in singles], "unans": unans}


def _row_of(c: Candidate, hop: Hop | None, counts: dict) -> dict:
    hops = [[c.path, *c.lines]] + ([[hop.path, *hop.lines]] if hop is not None else [])
    return {"template": c.template, "text": c.text, "answer": c.answer, "upstream_answer": c.upstream, "gold_path": c.path, "gold_lines": list(c.lines),
            "gold_paths": list(dict.fromkeys(h[0] for h in hops)), "gold_lines_all": hops, "hops": len(hops), "cross_file": hop is not None and hop.path != c.path,
            "answerable": True, "subject": c.subject, "confusers": _confusers(c, counts), **({"origin": c.origin} if c.origin else {})}  # fmt: skip


def _rows(episode: int, drawn: dict, mix: Mix, questions: int, probe_gap: int, rng: random.Random, counts: dict) -> list[dict]:
    slots = layout(Mix(len(drawn["singles"]) + len(drawn["pairs"]), len(drawn["pairs"]), len(drawn["multi"]), len(drawn["unans"])), questions, probe_gap, rng)
    # the k-th "Ap" slot takes the k-th pair's anchor and the k-th "P" slot its probe
    queues = {"Ap": list(drawn["pairs"]), "P": list(drawn["pairs"]), "A": list(drawn["singles"]), "M": list(drawn["multi"]), "U": list(drawn["unans"])}
    rows, anchor_at = [], {}
    none = {"fold_probe": False, "probe_of": None, "probe_gap": None}
    for pos, kind in enumerate(slots):
        qid = f"v2e{episode:02d}q{pos:02d}"
        item = queues[kind].pop(0)
        if kind == "U":
            row = {**item.row, "gold_paths": [], "gold_lines_all": [], "hops": 0, "cross_file": False, "confusers": {str(rho): 0 for rho in counts}, **none}
        elif kind == "M":
            row = {**_row_of(item.cand, item.hop, counts), **none}
        elif kind == "Ap":
            row = {**_row_of(item[0], None, counts), **none}
            anchor_at[id(item)] = (pos, qid)
        elif kind == "P":
            anchor_pos, anchor_qid = anchor_at[id(item)]
            row = {**_row_of(item[1], None, counts), "fold_probe": True, "probe_of": anchor_qid, "probe_gap": pos - anchor_pos}
        else:
            row = {**_row_of(item, None, counts), **none}
        rows.append({"qid": qid, "episode": episode, "pos": pos, "version": VERSION, **row})
    return rows


# Session facts (BUILD-V2c)


@dataclass(frozen=True)
class Fact:
    service: str
    attribute: str
    kind: str  # "int" or "token"


def note_text(fact: Fact, value: str, update: bool = False) -> str:
    if update:
        return f"Update: the {fact.attribute} of service {fact.service} is now {value}."
    return f"Deployment note: in this deployment, the {fact.attribute} of service {fact.service} is {value}."


def probe_text(fact: Fact) -> str:
    return f"What is the current {fact.attribute} of service {fact.service}?"


def corpus_words(haystack: list[str]) -> set[str]:
    """Every alphanumeric token of the corpus, lower-cased: a session value is none of them."""
    return {t.lower() for text in haystack for t in re.findall(r"[A-Za-z0-9]+", text)}


_CONSONANTS = "bcdfghjkmnpqrstvwxz"


def _value(kind: str, rng: random.Random, words: set[str], used: set[str]) -> str:
    """A fresh value, in no corpus file and not yet used in the episode: an integer of 4 or 5
    digits, or a short token (two letters, two digits, a letter)."""
    while True:
        if kind == "int":
            value = str(rng.randrange(1000, 100000))
        else:
            value = "".join(rng.choice(_CONSONANTS) for _ in range(2)) + str(rng.randrange(10, 100)) + rng.choice(_CONSONANTS)
        if value not in words and value not in used:
            used.add(value)
            return value


def session_slots(mix: Mix, questions: int, rng: random.Random) -> dict:
    """Where an episode's session facts go, among questions + session probes positions: the
    answerable and unanswerable probe positions; for each answerable probe the position of its
    note (SESSION_GAP before it) and, for `supersessions` of them, of the update between the
    two; and the positions of the notes no probe asks about. Notes and updates sit on corpus
    questions, one per question; an unanswerable probe comes at least SESSION_GAP[0] after the
    first note. QuestionError if no layout is found."""
    lo, hi = SESSION_GAP
    total = questions + mix.session + mix.session_unanswerable
    for _ in range(1000):
        probes = sorted(rng.sample(range(lo, total), mix.session + mix.session_unanswerable))
        answerable = sorted(rng.sample(probes, mix.session))
        free = [p for p in range(total) if p not in probes]
        taken: set[int] = set()

        def pick(options: list[int]) -> int | None:
            options = [o for o in options if o not in taken]
            if not options:
                return None
            taken.add(choice := rng.choice(options))
            return choice

        note_at = {p: pick([n for n in free if lo <= p - n <= hi]) for p in answerable}
        if None in note_at.values():
            continue
        update_at = {p: pick([u for u in free if note_at[p] < u < p]) for p in sorted(rng.sample(answerable, mix.supersessions))}
        rest = [n for n in free if n not in taken]
        if None in update_at.values() or len(rest) < mix.notes - mix.session:
            continue
        unprobed = sorted(rng.sample(rest, mix.notes - mix.session))
        first = min([*note_at.values(), *unprobed], default=0)
        unanswerable = [p for p in probes if p not in answerable]
        if all(p - first >= lo for p in unanswerable):
            return {"total": total, "answerable": answerable, "unanswerable": unanswerable, "note_at": note_at, "update_at": update_at, "unprobed": unprobed}
    raise QuestionError(f"the session mix {mix.to_dict()} does not fit in episodes of {questions} questions with gaps {lo}-{hi}")


def session_facts(mix: Mix, base: str, rng: random.Random, haystack: list[str], overlap: Overlap | None = None) -> list[Fact]:
    """The confusable service/attribute pairs of one episode: three services of one family
    named after `base` (orion-relay, orion-relayd, orion-proxy; a family any of whose names
    occurs in the corpus is skipped) times enough attribute families for every note and
    unanswerable probe to have its own pair. With `overlap` (questions v4), the first family and
    run of attribute families (in seeded order) with a fresh probe for every probe slot."""
    families = [f for f in rng.sample(SESSION_SUFFIXES, len(SESSION_SUFFIXES)) if not any(base + sfx in text for sfx in f for text in haystack)]
    if not families:
        raise QuestionError(f"every service family of {base!r} occurs in the corpus")
    services = [base + sfx for sfx in families[0]]
    need = -(-(mix.notes + mix.session_unanswerable) // (3 * len(services)))
    if need > len(SESSION_ATTRIBUTES):
        raise QuestionError(f"the session mix {mix.to_dict()} needs more service/attribute pairs than there are")
    if overlap is None:
        return [Fact(svc, attr, kind) for attrs, kind in rng.sample(SESSION_ATTRIBUTES, need) for attr in attrs for svc in services]
    order = rng.sample(SESSION_ATTRIBUTES, len(SESSION_ATTRIBUTES))
    for family in families:
        for start in range(len(order) - need + 1):
            facts = [Fact(base + sfx, attr, kind) for attrs, kind in order[start : start + need] for attr in attrs for sfx in family]
            if sum(overlap.fresh_probe(f) for f in facts) >= mix.session + mix.session_unanswerable:
                return facts
    raise QuestionError(f"no service family of {base!r} has {mix.session + mix.session_unanswerable} session probes the reference did not ask")


def with_session(rows: list[dict], episode: int, mix: Mix, rng: random.Random, base: str, words: set[str], haystack: list[str], counts: dict,
                 version: str = SESSION_VERSION, overlap: Overlap | None = None, kept_out: Counter | None = None) -> list[dict]:  # fmt: skip
    """An episode's corpus rows with its session facts (module docstring): probes interleaved,
    notes attached to corpus questions, positions, ids and fold-probe gaps renumbered; rows
    labelled `version` ("v2s" or "v3"). With `overlap` (questions v4), probes ask only about
    pairs no reference item asked about."""
    slots = session_slots(mix, len(rows), rng)
    facts = session_facts(mix, base, rng, haystack, overlap)
    if overlap is None:
        noted = rng.sample(facts, mix.notes)
        unanswered = rng.sample([f for f in facts if f not in noted], mix.session_unanswerable)
    else:
        fresh = [f for f in facts if overlap.fresh_probe(f)]
        kept_out["overlap_session_pair"] += len(facts) - len(fresh)
        probed = rng.sample(fresh, mix.session)
        unanswered = rng.sample([f for f in fresh if f not in probed], mix.session_unanswerable)
        noted = probed + rng.sample([f for f in facts if f not in probed and f not in unanswered], mix.notes - mix.session)
    used: set[str] = set()
    literal = lambda fact, value: value if fact.kind == "int" else repr(value)  # noqa: E731
    notes: dict[int, dict] = {}
    probes: dict[int, tuple[Fact, str | None, list[int]]] = {}

    def note(pos: int, fact: Fact, update: bool = False) -> str:
        value = _value(fact.kind, rng, words, used)
        notes[pos] = {"text": note_text(fact, value, update), "kind": "update" if update else "note", "service": fact.service,
                      "attribute": fact.attribute, "value": literal(fact, value)}  # fmt: skip
        return value

    for p, fact in zip(slots["answerable"], noted):
        value, at = note(slots["note_at"][p], fact), [slots["note_at"][p]]
        if p in slots["update_at"]:
            value = note(slots["update_at"][p], fact, update=True)
            at.append(slots["update_at"][p])
        probes[p] = (fact, literal(fact, value), at)
    for n, fact in zip(slots["unprobed"], noted[mix.session :]):
        note(n, fact)
    for p, fact in zip(slots["unanswerable"], unanswered):
        probes[p] = (fact, None, [])
    qid = lambda pos: f"{QID_PREFIX[version]}{episode:02d}q{pos:02d}"  # noqa: E731
    corpus = iter(rows)
    new_pos: dict[str, int] = {}
    out = []
    for pos in range(slots["total"]):
        if pos in probes:
            fact, answer, at = probes[pos]
            look_alikes = sum(1 for n, v in notes.items() if n < pos and (v["service"], v["attribute"]) != (fact.service, fact.attribute)
                              and (v["service"] == fact.service or v["attribute"] == fact.attribute))  # fmt: skip
            out.append({"qid": qid(pos), "episode": episode, "pos": pos, "version": version, "template": "session", "text": probe_text(fact),
                        "answer": answer, "upstream_answer": None, "gold_path": None, "gold_lines": None, "gold_paths": [], "gold_lines_all": [],
                        "hops": 0, "cross_file": False, "answerable": answer is not None, "subject": fact.service, "attribute": fact.attribute,
                        "confusers": {str(rho): look_alikes for rho in counts}, "fold_probe": False, "probe_of": None, "probe_gap": None, "note": None,
                        "notes_of": at, "session_gap": pos - at[0] if at else None, "superseded": len(at) > 1})  # fmt: skip
            continue
        row = next(corpus)
        new_pos[row["qid"]] = pos
        out.append({**row, "qid": qid(pos), "pos": pos, "version": version, "note": notes.get(pos)})
    for row in out:
        if row["probe_of"] is not None:
            anchor = new_pos[row["probe_of"]]
            row["probe_of"], row["probe_gap"] = qid(anchor), row["pos"] - anchor
    return out


def metadata(rows: list[dict], manifest: dict, seed: int, mix: Mix, questions: int, probe_gap: int, version: str | None = None) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    session = [r for r in rows if r["template"] == "session"]
    notes = [r["note"] for r in rows if r.get("note")]
    return {
        "version": version or (SESSION_VERSION if mix.has_session else VERSION),
        "corpus_id": manifest["id"],
        "seed": seed,
        "episodes": len({r["episode"] for r in rows}),
        "questions_per_episode": questions,
        "mix": mix.to_dict(),
        "probe_gap": probe_gap,
        "questions": len(rows),
        "answerable": len(answerable),
        "fold_probes": sum(r["fold_probe"] for r in rows),
        "multihop": sum(r["hops"] == 2 for r in rows),
        "cross_file": sum(r["cross_file"] for r in rows),
        "unanswerable": len(rows) - len(answerable),
        "distinct_gold_files": len({r["gold_path"] for r in answerable if r["gold_path"] is not None}),
        "template_mix": dict(sorted(Counter(r["template"] for r in answerable).items())),
        "with_confusers": sum(any(v > 0 for v in r["confusers"].values()) for r in answerable),
        **({"session": {
            "notes": sum(n["kind"] == "note" for n in notes), "supersessions": sum(n["kind"] == "update" for n in notes),
            "probes": sum(r["answerable"] for r in session), "unanswerable": sum(not r["answerable"] for r in session),
            "superseded_probes": sum(r["superseded"] for r in session), "gap_range": list(SESSION_GAP),
            "gaps": dict(sorted(Counter(r["session_gap"] for r in session if r["session_gap"] is not None).items())),
        }} if mix.has_session else {}),
    }  # fmt: skip


def build_questions_v2(manifest: dict, *, seed: int, episodes: int, questions: int = QUESTIONS, mix: Mix | None = None, probe_gap: int = PROBE_GAP,
                       runs_dir: str | Path = RUNS_CORPUS, version: str | None = None) -> list[dict]:  # fmt: skip
    """Generate exactly `episodes` v2 episodes and write questions-v2.jsonl (questions-v2s.jsonl
    for a mix with session facts, or questions-v3.jsonl for `version` "v3", which needs one);
    QuestionError if the corpus cannot supply them (`capacity` says how many it can)."""
    mix = mix or Mix.default(questions)
    version = version or (SESSION_VERSION if mix.has_session else VERSION)
    if (version == VERSION) == mix.has_session or version not in FILENAMES:
        raise QuestionError(f"version {version} and a mix {'with' if mix.has_session else 'without'} session facts do not go together")
    if version == V4_VERSION:
        built, fresh = build_fresh(manifest, seed=seed, episodes=episodes, questions=questions, mix=mix, probe_gap=probe_gap, runs_dir=runs_dir)
    else:
        built, fresh = build_v2(manifest, seed=seed, episodes=episodes, questions=questions, mix=mix, probe_gap=probe_gap, runs_dir=runs_dir, version=version), None
    made = len({r["episode"] for r in built.rows})
    if made < episodes:
        raise QuestionError(f"the corpus supports only {made} of {episodes} v2 episodes of {questions} questions with mix {mix.to_dict()}; see questions build --dry-run")
    out = Path(runs_dir) / manifest["id"] / FILENAMES[version]
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata(built.rows, manifest, seed, mix, questions, probe_gap, version)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"metadata": meta if fresh is None else {**meta, "fresh": fresh}}, sort_keys=True) + "\n")
        for row in built.rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return built.rows


def build_fresh(manifest: dict, *, seed: int, episodes: int | None, questions: int = QUESTIONS, mix: Mix | None = None, probe_gap: int = PROBE_GAP,
                runs_dir: str | Path = RUNS_CORPUS) -> tuple[Build, dict]:  # fmt: skip
    """Questions v4 (module docstring, "Fresh items"): the build with the overlap filter, with
    extras only if the filter alone supports fewer than MIN_FRESH_EPISODES episodes; and what the
    file's metadata records about it."""
    overlap = load_overlap(manifest, runs_dir)
    kwargs = {"seed": seed, "questions": questions, "mix": mix, "probe_gap": probe_gap, "runs_dir": runs_dir, "version": V4_VERSION, "overlap": overlap}
    alone = len({r["episode"] for r in build_v2(manifest, episodes=None, **kwargs).rows})
    extras = alone < MIN_FRESH_EPISODES
    built = build_v2(manifest, episodes=episodes, extras=extras, **kwargs)
    available = len({r["episode"] for r in build_v2(manifest, episodes=None, extras=extras, **kwargs).rows}) if episodes is not None else len({r["episode"] for r in built.rows})
    origins = Counter(r.get("origin") for r in built.rows if r["answerable"] and r["template"] != "session")
    return built, {
        "rule": OVERLAP_RULE, "reference": overlap.reference, "min_fresh_episodes": MIN_FRESH_EPISODES, "episodes_filter_only": alone,
        "extras": extras, "extra_rule": EXTRA_RULE if extras else None, "episodes_available": available,
        "excluded": dict(sorted(built.excluded.items())), "origins": dict(sorted(origins.items())),
    }  # fmt: skip


def overlap_report(rows: list[dict], reference_rows: list[dict]) -> dict:
    """Each item of `rows` against questions v3 by episode group (V3_REFERENCE, and "other" for
    the episodes no run asked): items overlapping by OVERLAP_RULE, and items whose target
    identifiers a v3 item also asks about (with any answer; informative)."""
    groups = {e: g for g, eps in V3_REFERENCE.items() for e in eps}
    names = [*V3_REFERENCE, "other"]
    keys, texts, targets = ({g: set() for g in names} for _ in range(3))
    for r in reference_rows:
        g = groups.get(r["episode"], "other")
        keys[g].add((target_key(r["text"]), _answer_key(r["answer"])))
        texts[g].add(normalized(r["text"]))
        targets[g].add(target_key(r["text"]))
    overlap = {g: Counter() for g in names}
    reused = {g: Counter() for g in names}
    hit = 0
    for r in rows:
        kind = "session" if r["template"] == "session" else "corpus" if r["answerable"] else "near_miss"
        key, text = (target_key(r["text"]), _answer_key(r["answer"])), normalized(r["text"])
        hits = {g for g in names if key in keys[g] or text in texts[g]}
        for g in hits:
            overlap[g][kind] += 1
        for g in (g for g in names if key[0] in targets[g]):
            reused[g][kind] += 1
        hit += bool(hits - {"other"})
    return {"items": len(rows), "episodes": len({r["episode"] for r in rows}), "rule": OVERLAP_RULE, "overlapping_reference": hit,
            "overlap": {g: dict(sorted(c.items())) for g, c in overlap.items()}, "target_reused": {g: dict(sorted(c.items())) for g, c in reused.items()},
            "origins": dict(sorted(Counter(r.get("origin", "-") for r in rows if r["answerable"] and r["template"] != "session").items()))}  # fmt: skip


def render_overlap(report: dict, label: str) -> str:
    episodes = {g: f"{eps[0]}-{eps[-1]}" if len(eps) > 1 else str(eps[0]) for g, eps in V3_REFERENCE.items()}
    lines = [f"{label}: {report['items']} items in {report['episodes']} episodes; {report['overlapping_reference']} overlap questions v3 episodes "
             f"{', '.join(episodes.values())} ({report['rule']})"]  # fmt: skip
    for g, counts in report["overlap"].items():
        lines.append(f"  v3 {g} ({episodes.get(g, 'the rest')}): overlapping {counts or 0}; same target, any answer {report['target_reused'][g] or 0}")
    lines.append(f"  answerable corpus items by origin: {report['origins']}")
    return "\n".join(lines)


def capacity(manifest: dict, *, seed: int, questions: int = QUESTIONS, mix: Mix | None = None, probe_gap: int = PROBE_GAP, runs_dir: str | Path = RUNS_CORPUS,
             version: str | None = None) -> dict:  # fmt: skip
    """How many episodes of this size and mix the corpus supports (drawn until it runs out), the
    candidates per template and the confuser distribution. Writes nothing. Version v4 counts with
    the overlap filter (and extras if it needs them)."""
    mix = mix or Mix.default(questions)
    if version == V4_VERSION:
        built, fresh = build_fresh(manifest, seed=seed, episodes=None, questions=questions, mix=mix, probe_gap=probe_gap, runs_dir=runs_dir)
    else:
        built, fresh = build_v2(manifest, seed=seed, episodes=None, questions=questions, mix=mix, probe_gap=probe_gap, runs_dir=runs_dir), None
    rows = built.rows
    return {
        **({"fresh": fresh} if fresh is not None else {}),
        "episodes": len({r["episode"] for r in rows}),
        "questions_per_episode": questions,
        "mix": mix.to_dict(),
        "candidates": dict(sorted(built.candidates.items())),
        "drawn": dict(sorted(Counter(r["template"] for r in rows if r["answerable"]).items())),
        "unanswerable_drawn": sum(not r["answerable"] and r["template"] != "session" for r in rows),
        "multihop_drawn": sum(r["hops"] == 2 for r in rows),
        "confusers": built.confusers,
        **({"session_drawn": {"notes": sum(bool(r.get("note")) for r in rows), "probes": sum(r["template"] == "session" and r["answerable"] for r in rows),
                              "unanswerable": sum(r["template"] == "session" and not r["answerable"] for r in rows)}} if mix.has_session else {}),
    }  # fmt: skip


def render_capacity(c: dict) -> str:
    lines = [f"capacity: {c['episodes']} episodes of {c['questions_per_episode']} questions, mix {c['mix']}"]
    lines.append("candidates per template: " + ", ".join(f"{k} {v}" for k, v in c["candidates"].items()))
    lines.append("drawn per template (answerable): " + ", ".join(f"{k} {v}" for k, v in c["drawn"].items())
                 + f"; multi-hop {c['multihop_drawn']}; unanswerable {c['unanswerable_drawn']}")  # fmt: skip
    for rho, dist in c["confusers"].items():
        lines.append(f"confusers in C{rho} (other definitions -> candidates): " + ", ".join(f"{n}: {k}" for n, k in dist.items()))
    if "session_drawn" in c:
        d = c["session_drawn"]
        lines.append(f"session facts drawn: {d['notes']} notes (updates included), {d['probes']} answerable and {d['unanswerable']} unanswerable session probes")
    if "fresh" in c:
        f = c["fresh"]
        lines.append(f"fresh (v4): {f['episodes_filter_only']} episodes with the overlap filter alone; extras {'used' if f['extras'] else 'not needed'}; "
                     f"excluded {f['excluded']}; corpus items by origin {f['origins']}")  # fmt: skip
    return "\n".join(lines)


def load_questions_v2(manifest: dict, runs_dir: str | Path = RUNS_CORPUS, version: str = VERSION) -> list[dict]:
    """The rows of questions-v2.jsonl, or of questions-v2s.jsonl for `version` "v2s", or of
    questions-v3.jsonl for "v3"."""
    path = Path(runs_dir) / manifest["id"] / FILENAMES.get(version, FILENAME)
    if not path.exists():
        raise QuestionError(f"no {version} questions for corpus {manifest['id']}; run `questions build --version {version}` first")
    return [row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()) if "metadata" not in row]


__all__ = [
    "MIN_FRESH_EPISODES", "OVERLAP_RULE", "Overlap", "V3_REFERENCE", "V4_FILENAME", "build_fresh", "extra_candidates", "load_overlap", "overlap_report", "question_rows",
    "reference_path", "render_overlap", "target_key",
    "FILENAME", "SESSION_FILENAME", "SESSION_VERSION", "V3_FILENAME", "V3_VERSION", "Fact", "Mix", "ModuleIndex", "build_questions_v2", "build_v2", "capacity", "layout", "leaks",
    "load_questions_v2", "multihop_candidates", "note_text", "probe_text", "render_capacity", "session_slots", "with_session",
]  # fmt: skip
