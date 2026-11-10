"""A scripted, deterministic stand-in for an LLM, used by the demo. No network.

It answers a question only from the text of the request it is given:
- the target span's text is verbatim in the request: answer with the expected value, cite
  the span, and quote the line that contains the value;
- only a fold sentence about the span is present: "guess" a wrong value (a simulated
  hallucination) and cite that fold sentence;
- neither: abstain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from context_ledger.model import Citation
from context_ledger.session import CallResult


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    target_span_id: str | None
    expected_value: str


@dataclass(frozen=True)
class Answer:
    kind: str  # "answer", "guess" or "abstain"
    text: str
    citations: tuple[Citation, ...]


def _wrong(value: str) -> str:
    return str(int(value) * 2) if value.isdigit() else value + "_guess"


def _fold_ref_citing(contents: list[str], span_id: str) -> str | None:
    """The "fold:<id>#<i>" ref of a fold sentence in the request that cites `span_id`."""
    for content in contents:
        lines = content.split("\n")
        if not (lines[0].startswith("[fold ") and lines[0].endswith("]")):
            continue
        fold_id = lines[0][len("[fold ") : -1]
        for i, line in enumerate(lines[1:]):
            cites = line.rsplit(" [", 1)[-1].rstrip("]").split(", ")
            if span_id in cites:
                return f"fold:{fold_id}#{i}"
    return None


class MockModel:
    """`span_text` is the scripted oracle: the text a span id stands for, or None if unknown."""

    def __init__(self, span_text: Callable[[str], str | None]) -> None:
        self.span_text = span_text

    def answer(self, result: CallResult, q: Question) -> Answer:
        contents = list(result.request["system"]) + [m["content"] for m in result.request["messages"]]
        text = self.span_text(q.target_span_id) if q.target_span_id else None
        if text is not None and any(text in content for content in contents):
            line = next(line.strip() for line in text.split("\n") if q.expected_value in line)
            quote = line if len(line) >= 20 else text
            return Answer("answer", f"The value is {q.expected_value}.", (Citation(q.target_span_id, quote),))
        fold_ref = _fold_ref_citing(contents, q.target_span_id) if q.target_span_id else None
        if fold_ref is not None:
            return Answer("guess", f"The value is {_wrong(q.expected_value)}.", (Citation(fold_ref),))
        return Answer("abstain", f"I cannot find the answer to: {q.question}", ())
