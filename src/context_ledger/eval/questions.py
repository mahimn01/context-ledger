"""Questions and episodes (C.3), generated from the PERTURBED source of the gold files.

Templates:
    default       What is the default value of parameter `p` of `mod.f`?
    constant      What is the value of the module constant `mod.X`?
    raises        Which exception classes does `mod.f` raise directly (in its own body)?
    locate        Which module defines the function `g`?   (renamed functions first)
    unanswerable  any template above over a near-miss symbol: a real name with one edit, absent
                  from every corpus file by exact search

Filters: the subject is defined exactly once in the largest corpus built (C1000); its gold
file is in the smallest (C10); default and constant answers come from a perturbed literal, so
memorization cannot answer them (`upstream_answer` is the original, unperturbed value).

Episodes: 12 of 8 questions, each drawing on 2-3 gold files. Six are answerable: three anchors,
then three fold probes, each probe asking about the same gold span (or at least the same file)
as its anchor, so the agent read it while answering the first four questions. The first four
slots are the anchors plus one unanswerable question, which always sits in the first three;
the last four are the probes plus the other unanswerable. Totals: 72 answerable (36 probes)
and 24 unanswerable.

Output: runs/corpus/<id>/questions.jsonl, one JSON object per question:
{qid, episode, pos, template, text, answer, upstream_answer, gold_path, gold_lines, answerable,
fold_probe, probe_of, subject}.
"""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .corpus import RUNS_CORPUS, chunk_lines, corpus_texts
from . import pyscan
from .perturb import char_offsets, char_span, perturbable

EPISODES = 12
ANCHORS = 3
TEMPLATES = ("default", "constant", "raises", "locate")


class QuestionError(Exception):
    """The corpus cannot supply the questions asked for."""


@dataclass(frozen=True)
class Candidate:
    template: str
    path: str
    subject: str  # the bare name the filter checks
    text: str
    answer: object
    upstream: object
    lines: tuple[int, int]
    near: tuple  # facts for a near-miss variant: (template, module, qualname, param)
    origin: str = ""  # questions v4 only: which generator made it (questions_v2.ORIGINS); "" leaves v1-v3 rows as they were


def _texts(template: str, module: str, qualname: str, param: str | None) -> str:
    if template == "default":
        return f"What is the default value of parameter `{param}` of `{module}.{qualname}`?"
    if template == "constant":
        return f"What is the value of the module constant `{module}.{qualname}`?"
    if template == "raises":
        return f"Which exception classes does `{module}.{qualname}` raise directly (in its own body)?"
    return f"Which module defines the function `{qualname}`?"


def file_candidates(path: str, text: str, records: list[dict], defined: Counter, *, unique: bool = True) -> list[Candidate]:
    """Every question the perturbed gold file supports that passes the filters. With
    unique=False (questions v2) a default, constant or raises subject may be defined elsewhere
    too (a confusable subject: the question names its module), as long as its qualname is
    defined once in this file; locate keeps the unique-definition rule."""
    tree = pyscan.parse(text)
    if tree is None:
        return []
    module = pyscan.module_path(path)
    by_site = {(r["kind"], r["line"], r["param"]): r for r in records if r["kind"] in ("default", "constant")}
    renamed = {r["new"] for r in records if r["kind"] == "rename"}
    out: list[Candidate] = []
    funcs = pyscan.functions(tree)
    if unique:
        once = lambda qualname, bare: defined[bare] == 1  # noqa: E731
    else:
        local = Counter(q for q, _ in funcs) + Counter(pyscan.defined_names(tree)) - Counter(q.rsplit(".", 1)[-1] for q, _ in funcs)
        once = lambda qualname, bare: local[qualname] == 1  # noqa: E731
    for site in pyscan.literal_sites(tree):
        record = by_site.get((site.kind, site.node.lineno, site.param))
        if record is None:
            continue
        kind, value = pyscan.literal_value(site.node)
        if value != ast.literal_eval(record["new"]):
            continue
        subject = site.qualname.rsplit(".", 1)[-1]
        if not once(site.qualname, subject):
            continue
        if site.kind == "constant":
            stmt = next(n for n in tree.body if n.lineno <= site.node.lineno <= n.end_lineno)
            lines = (stmt.lineno, stmt.end_lineno)
        else:
            func = next(f for q, f in funcs if q == site.qualname)
            lines = (min([func.lineno, *(d.lineno for d in func.decorator_list)]), func.end_lineno)
        out.append(Candidate(site.kind, path, subject, _texts(site.kind, module, site.qualname, site.param), record["new"], record["old"], lines,
                             (site.kind, module, site.qualname, site.param)))  # fmt: skip
    for qualname, func in funcs:
        subject = qualname.rsplit(".", 1)[-1]
        if not once(qualname, subject) or "<locals>" in qualname:
            continue
        lines = (min([func.lineno, *(d.lineno for d in func.decorator_list)]), func.end_lineno)
        raises = sorted(pyscan.direct_raises(func))
        if raises:
            out.append(Candidate("raises", path, subject, _texts("raises", module, qualname, None), raises, None, lines, ("raises", module, qualname, None)))
        if defined[subject] == 1 and (subject in renamed or (not renamed and "." not in qualname and len(subject) >= 6)):
            out.append(Candidate("locate", path, subject, _texts("locate", module, subject, None), module, None, lines, ("locate", module, subject, None)))
    return out


def defined_counts(texts: dict[str, str]) -> Counter:
    """How many times each name is defined across the .py files of `texts` (pyscan.defined_names)."""
    out: Counter = Counter()
    for path, text in texts.items():
        tree = pyscan.parse(text) if path.endswith(".py") else None
        if tree is not None:
            out.update(pyscan.defined_names(tree))
    return out


def question_density(path: str, text: str, defined: Counter, tokens: int) -> Fraction:
    """Question candidates per token of an UNPERTURBED file: `file_candidates` with every
    literal whose form the perturbation rewrites standing in as its own record, so the count
    uses the templates and filters but never a perturbed value (no answer). Renames are not
    known yet, so locate counts top-level functions of six or more characters."""
    tree = pyscan.parse(text)
    if tree is None or tokens <= 0:
        return Fraction(0)
    starts, lines = char_offsets(text)
    records = []
    for site in pyscan.literal_sites(tree):
        start, end = char_span(starts, lines, *site.span)
        if perturbable(pyscan.literal_value(site.node)[0], text[start:end]):
            records.append({"kind": site.kind, "line": site.node.lineno, "param": site.param, "old": text[start:end], "new": text[start:end]})
    return Fraction(len(file_candidates(path, text, records, defined)), tokens)


_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def near_miss(name: str, rng: random.Random, haystack: list[str]) -> str | None:
    """`name` with one seeded letter edit (swap two adjacent letters, double or drop a letter)
    or a short suffix, that no corpus file contains."""
    letters = [i for i in range(1, len(name)) if name[i].isalpha()]
    for _ in range(200):
        kind = rng.choice(["swap", "double", "drop", "suffix"])
        i = rng.choice(letters) if letters else 0
        if kind == "swap" and i + 1 < len(name) and name[i + 1].isalpha() and name[i] != name[i + 1]:
            new = name[:i] + name[i + 1] + name[i] + name[i + 2 :]
        elif kind == "double" and letters:
            new = name[:i] + name[i] + name[i:]
        elif kind == "drop" and letters and len(name) > 4 and name[i - 1] != "_":
            new = name[:i] + name[i + 1 :]
        else:
            sep = "_" if "_" in name or name.isupper() else ""
            tail = "".join(rng.choice(_LETTERS) for _ in range(2))
            new = name + sep + (tail.upper() if name.isupper() else tail)
        if new != name and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", new) and "__" not in new.strip("_") and not any(new in text for text in haystack):
            return new
    return None


def _unanswerable(anchor: Candidate, rng: random.Random, haystack: list[str]) -> dict | None:
    template, module, qualname, param = anchor.near
    bare = qualname.rsplit(".", 1)[-1]
    fake = near_miss(bare, rng, haystack)
    if fake is None:
        return None
    fake_qualname = qualname[: len(qualname) - len(bare)] + fake
    return {"template": template, "text": _texts(template, module, fake_qualname, param), "answer": None, "upstream_answer": None,
            "gold_path": None, "gold_lines": None, "answerable": False, "subject": fake}  # fmt: skip


def _pairs(cands: list[Candidate], rng: random.Random, path: str, text: str) -> list[tuple[Candidate, Candidate]]:
    """(anchor, probe) pairs from one file: two different questions about the same 60-line
    chunk first, then about the same file; each candidate is used once. Default and constant
    questions (perturbed literals, alternating) are paired first, then raises, then locate.
    A locate question pairs only with a locate question: every other template names the
    file's module, which is a locate question's answer (see `_leak_free`)."""
    chunks = chunk_lines(path, text)

    def chunk_of(c: Candidate) -> int:
        return next((k for k, (s, e) in enumerate(chunks) if s <= c.lines[0] <= e), -1)

    shuffled = list(cands)
    rng.shuffle(shuffled)
    by_template = {t: [c for c in shuffled if c.template == t] for t in TEMPLATES}
    literals = [c for pair in zip(by_template["default"], by_template["constant"]) for c in pair]
    longer = by_template["default"] if len(by_template["default"]) > len(by_template["constant"]) else by_template["constant"]
    literals += longer[len(literals) // 2 :]
    pool = literals + by_template["raises"] + by_template["locate"]  # perturbed literals first: they measure leakage
    pairs, used = [], set()
    for same_chunk in (True, False):
        for a in pool:
            if id(a) in used:
                continue
            for p in pool:
                if id(p) in used or p is a or p.text == a.text or (p.template == "locate") != (a.template == "locate"):
                    continue
                if not same_chunk or chunk_of(a) == chunk_of(p):
                    pairs.append((a, p))
                    used.update({id(a), id(p)})
                    break
    return pairs


def _leak_free(pairs: list[tuple[Candidate, Candidate]]) -> list[tuple[Candidate, Candidate]]:
    """The pairs a file may contribute to one episode: only locate pairs, or none, by the kind of
    its first pair. Questions of every other template name the file's module (`mod.f`), which is
    the answer of its locate questions, so mixing them would let an earlier question in the
    conversation give a locate answer away."""
    return [p for p in pairs if (p[0].template == "locate") == (pairs[0][0].template == "locate")] if pairs else []


def build_questions(manifest: dict, *, seed: int, episodes: int = EPISODES, runs_dir: str | Path = RUNS_CORPUS) -> list[dict]:
    """Generate exactly `episodes` episodes and write questions.jsonl next to the manifest (its
    first line is {"metadata": ...}); returns the question rows. Raises QuestionError if the
    gold files cannot supply that many.

    First pass: each episode takes 2-3 gold files not used yet, in a seeded order, and three of
    their (anchor, probe) pairs (from each file only locate pairs, or none: `_leak_free`). Second pass, only if the files run out: a gold file may serve
    one more episode with pairs it has left, so at most two episodes share a file, no question
    repeats and every probe stays in its anchor's episode."""
    texts = corpus_texts(manifest)
    out_dir = Path(runs_dir) / manifest["id"]
    records_path = out_dir / "perturbations.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()] if records_path.exists() else []
    defined = defined_counts(texts)
    smallest = min(manifest["rhos"])
    in_small = {f["path"] for f in manifest["files"] if f["rho"] == smallest}
    excluded = set(manifest.get("excluded", []))
    gold = [p for p in manifest["gold"] if p in in_small and p not in excluded]
    rng = random.Random(f"questions:{seed}")
    by_file = {}
    for path in gold:
        cands = file_candidates(path, texts[path], [r for r in records if r["path"] == path], defined)
        pairs = _pairs(cands, rng, path, texts[path])
        if pairs:
            by_file[path] = pairs
    files = sorted(by_file)
    rng.shuffle(files)
    haystack = list(texts.values())
    rows: list[dict] = []
    episode = 0
    second: list[str] = []  # files that may serve a second episode, popped in order of first use
    uses: Counter = Counter()
    used_texts: set[str] = set()
    while episode < episodes and (files or second):
        queue = files if files else second
        take, pairs = [], []
        while queue and (len(take) < 2 or sum(map(len, pairs)) < ANCHORS) and len(take) < 3:
            path = queue.pop()
            take.append(path)
            pairs.append(_leak_free(by_file[path]))
        interleaved = [pair for group in zip(*[g + [None] * (ANCHORS - len(g)) for g in pairs]) for pair in group if pair is not None]
        chosen = interleaved[:ANCHORS]
        unans = []
        if len(chosen) == ANCHORS:
            for a, _ in chosen:
                q = _unanswerable(a, rng, haystack)
                if q is not None and q["text"] not in used_texts and len(unans) < 2:
                    unans.append(q)
        if len(chosen) < ANCHORS or len(unans) < 2:
            if queue is files:  # these files get their chance in the second pass
                second[:0] = take[::-1]
                continue
            break
        for a, b in chosen:
            by_file[a.path].remove((a, b))
        for path in take:
            uses[path] += 1
            if queue is files and by_file[path]:
                second.insert(0, path)
        episode += 1
        first = [("a", k) for k in range(ANCHORS)]
        first.insert(rng.randrange(0, 3), ("u", 0))
        last = [("p", k) for k in range(ANCHORS)]
        last.insert(rng.randrange(0, 4), ("u", 1))
        anchors_qid = {}
        for pos, (kind, k) in enumerate(first + last):
            qid = f"e{episode:02d}q{pos}"
            if kind == "u":
                row = dict(unans[k])
                row.update(fold_probe=False, probe_of=None)
            else:
                c = chosen[k][0] if kind == "a" else chosen[k][1]
                row = {"template": c.template, "text": c.text, "answer": c.answer, "upstream_answer": c.upstream, "gold_path": c.path,
                       "gold_lines": list(c.lines), "answerable": True, "subject": c.subject, "fold_probe": kind == "p",
                       "probe_of": anchors_qid.get(k) if kind == "p" else None}  # fmt: skip
                if kind == "a":
                    anchors_qid[k] = qid
            used_texts.add(row["text"])
            rows.append({"qid": qid, "episode": episode, "pos": pos, **row})
    if episode < episodes:
        raise QuestionError(f"built only {episode} of {episodes} episodes: the {len(by_file)} gold files in C{smallest} with question "
                            f"candidates ran out (each may serve at most two episodes)")  # fmt: skip
    meta = question_metadata(rows, manifest, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "questions.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"metadata": meta}, sort_keys=True) + "\n")
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return rows


def question_metadata(rows: list[dict], manifest: dict, seed: int) -> dict:
    """What questions.jsonl's first line reports: counts, distinct gold files (and those two
    episodes share), the template mix of the answerable questions, probes and unanswerables."""
    answerable = [r for r in rows if r["answerable"]]
    episodes_of: dict[str, set[int]] = {}
    for r in answerable:
        episodes_of.setdefault(r["gold_path"], set()).add(r["episode"])
    return {
        "corpus_id": manifest["id"],
        "seed": seed,
        "episodes": len({r["episode"] for r in rows}),
        "questions": len(rows),
        "answerable": len(answerable),
        "fold_probes": sum(r["fold_probe"] for r in rows),
        "unanswerable": len(rows) - len(answerable),
        "distinct_gold_files": len(episodes_of),
        "shared_gold_files": sorted(p for p, eps in episodes_of.items() if len(eps) > 1),
        "template_mix": dict(sorted(Counter(r["template"] for r in answerable).items())),
        "unanswerable_template_mix": dict(sorted(Counter(r["template"] for r in rows if not r["answerable"]).items())),
    }


def load_questions(manifest: dict, runs_dir: str | Path = RUNS_CORPUS) -> list[dict]:
    path = Path(runs_dir) / manifest["id"] / "questions.jsonl"
    if not path.exists():
        raise QuestionError(f"no questions for corpus {manifest['id']}; run `questions build` first")
    return [row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()) if "metadata" not in row]


def load_question_metadata(manifest: dict, runs_dir: str | Path = RUNS_CORPUS) -> dict | None:
    """The metadata line of questions.jsonl (None for files written before M4)."""
    path = Path(runs_dir) / manifest["id"] / "questions.jsonl"
    first = json.loads(path.read_text(encoding="utf-8").split("\n", 1)[0]) if path.exists() else {}
    return first.get("metadata")


def smoke_questions(rows: list[dict]) -> tuple[list[dict], dict | None]:
    """The smoke profile: the first 3 questions of episode 1 (one is unanswerable), and the fold
    probe of the first answerable one, for B4 and L."""
    first = sorted((r for r in rows if r["episode"] == 1), key=lambda r: r["pos"])
    three = first[:3]
    anchor = next((r for r in three if r["answerable"]), None)
    probe = next((r for r in first if anchor is not None and r.get("probe_of") == anchor["qid"]), None)
    return three, probe
