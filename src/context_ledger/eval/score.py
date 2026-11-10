"""Automatic scoring of a pilot run (D). No LLM judge.

Normalization. Literals go through ast.literal_eval where possible and compare by value (1e3
equals 1000.0; a string compares by value whatever its quotes; an answer that is not a
literal compares as stripped text). Module paths "Lib/a/b.py", "a/b.py" and "a.b" all become
"a.b", and "pkg/__init__.py" becomes "pkg". Exception sets compare by bare class name, scored
by set F1 (1.0 counts as correct).

Outcome. Answerable: correct, wrong or abstain. Unanswerable: correct_abstain or false_answer.

One scorer for every arm (M4). The ledger's verifier is part of arm L's mechanism and never
the grader: every arm's answer and citations are graded by the same functions here, whatever
L's verifier said.

Citations. Every arm exports its citations as "path:start-end" plus a quote (L's span refs are
mapped to the lines of their source, eval/tools.cited_lines). A cite passes if its quote occurs
verbatim (after whitespace folding) in the cited file's perturbed text at the cited lines;
`cite_passes` is the one check for all arms. A wrong answer or false answer is
hallucination_unsupported if any cite fails (or there is none) and hallucination_misread if all
pass.

Caught (L only, descriptive). Each answer L's egress check blocked (RefetchObligationError or
UngroundedClaimError) is recorded with its would-be grade (correct, wrong or false_answer). A
question is `caught` when a blocked answer would have been graded wrong or false_answer and
the final outcome is a correct answer or an abstention.

Leakage. An answer equal to the unperturbed value (`upstream_answer`) is parametric leakage.

Session probes (BUILD-V2c, template "session") are graded like every literal answer: the
answer value against the latest noted value, whatever was cited. A cite "note:<pos>" passes if
its quote occurs in the note of question <pos> of the same episode.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from pathlib import Path

from ..anchor.text import normalize_for_match
from .corpus import lines_of

_REF = re.compile(r"^(?P<path>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def literal(value: object) -> object:
    """The value a literal answer denotes: literal_eval of a string where it parses, else the
    stripped string; numbers and lists as they are."""
    if isinstance(value, str):
        text = value.strip()
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return text
    return value


def module_name(value: object) -> str:
    """"Lib/a/b.py", "a/b.py", "a.b" -> "a.b"; "pkg/__init__.py" -> "pkg"."""
    text = str(value).strip().strip("`'\"")
    if text.startswith("Lib/"):
        text = text[4:]
    if text.endswith(".py"):
        text = text[:-3]
    parts = [p for p in re.split(r"[/.]", text) if p]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def exception_names(value: object) -> set[str]:
    """Bare class names of an answer: a list, or a string of names separated by commas or 'and'."""
    value = literal(value)
    items = value if isinstance(value, (list, tuple, set)) else re.split(r",|\band\b|\s+", str(value))
    return {str(v).strip().strip("`'\"").rsplit(".", 1)[-1] for v in items if str(v).strip().strip("`'\"")}


def set_f1(got: set, want: set) -> tuple[int, int]:
    """F1 of two sets as an exact (numerator, denominator) pair: 2|A n B| / (|A| + |B|)."""
    total = len(got) + len(want)
    return (2 * len(got & want), total) if total else (1, 1)


def same_answer(template: str, got: object, want: object) -> bool:
    if template == "raises":
        num, den = set_f1(exception_names(got), exception_names(want))
        return num == den
    if template == "locate":
        return module_name(got) == module_name(want)
    a, b = literal(got), literal(want)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b and type(a) in (int, float, str) and type(b) in (int, float, str) and (isinstance(a, str) == isinstance(b, str))


def cite_passes(citation: dict, texts: dict[str, str], notes: dict[str, str] | None = None) -> bool:
    """A cite: its quote occurs verbatim (whitespace folded) at the cited lines, or in the cited
    note of `notes` ("note:<pos>" -> that question's note text in the episode)."""
    ref, quote = str(citation.get("ref", "")), citation.get("quote")
    if ref.startswith("note:"):
        return bool(quote) and ref in (notes or {}) and normalize_for_match(quote) in normalize_for_match(notes[ref])
    m = _REF.match(ref)
    if m is None or not quote or m.group("path") not in texts:
        return False
    span = lines_of(texts[m.group("path")], int(m.group("start")), int(m.group("end")))
    return normalize_for_match(quote) in normalize_for_match(span)


def would_be_grade(question: dict, answer: object) -> str:
    """The grade an answer would get: correct or wrong on an answerable question, false_answer
    on an unanswerable one."""
    if not question["answerable"]:
        return "false_answer"
    return "correct" if same_answer(question["template"], answer, question["answer"]) else "wrong"


def score_result(result: dict, question: dict, texts: dict[str, str], notes: dict[str, str] | None = None) -> dict:
    """One scored row: the outcome, any hallucination, a caught hallucination, leakage
    (`notes`: the episode's notes, for note cites)."""
    template = question["template"]
    answered = not result["abstained"] and result["answer"] is not None
    if question["answerable"]:
        outcome = "abstain" if not answered else "correct" if same_answer(template, result["answer"], question["answer"]) else "wrong"
    else:
        outcome = "false_answer" if answered else "correct_abstain"
    hallucination = None
    if outcome in ("wrong", "false_answer"):
        cites = result["citations"]
        hallucination = "misread" if cites and all(cite_passes(c, texts, notes) for c in cites) else "unsupported"
    blocked = [{"answer": a, "grade": would_be_grade(question, a)} for a in result.get("blocked_answers", [])]
    caught = any(b["grade"] in ("wrong", "false_answer") for b in blocked) and outcome in ("correct", "abstain", "correct_abstain")
    upstream = question.get("upstream_answer")
    leaked = answered and upstream is not None and same_answer(template, result["answer"], upstream)
    return {
        "arm": result["arm"],
        "rho": result["rho"],
        "model": result["model"],
        "episode": result["episode"],
        "qid": result["qid"],
        "template": template,
        "answerable": question["answerable"],
        "fold_probe": question["fold_probe"],
        "pos": question["pos"],
        "has_upstream": upstream is not None,
        "outcome": outcome,
        "hallucination": hallucination,
        "caught": caught,
        "blocked": blocked,
        "leaked": leaked,
        "bound_hit": result["bound_hit"],
        "calls": result["calls"],
        "cost_pusd": result["cost_pusd"],
        "error": result.get("error") is not None,
    }


def score(results: Iterable[dict], questions: Iterable[dict], texts: dict[str, str]) -> list[dict]:
    """Score every result against its question."""
    questions = list(questions)
    by_qid = {q["qid"]: q for q in questions}
    notes: dict[int, dict[str, str]] = {}
    for q in questions:
        if q.get("note"):
            notes.setdefault(q["episode"], {})[f"note:{q['pos']}"] = q["note"]["text"]
    return [score_result(r, by_qid[r["qid"]], texts, notes.get(r["episode"])) for r in results]


def score_run(run_dir: Path, questions: Iterable[dict], texts: dict[str, str]) -> list[dict]:
    """Score run_dir (and any grid subdirectories) and write each scores.jsonl; returns all rows."""
    questions = list(questions)
    out = []
    for results_path in sorted({run_dir / "results.jsonl", *run_dir.glob("*/results.jsonl")}):
        if not results_path.exists():
            continue
        results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        rows = score(results, questions, texts)
        with open(results_path.parent / "scores.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        out += rows
    return out
