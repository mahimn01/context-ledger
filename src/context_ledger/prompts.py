"""The versioned system prompt and the strict parser for the model's final answer.

The prompt's sha256 is recorded in every model_call's build params, so a ledger names the exact
instructions the model saw.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .ids import sha256_hex
from .model import Citation

PROMPT_VERSION = "clp-m2-1"
SYSTEM_PROMPT = """\
You answer questions about source material that is shown to you in pieces. Every piece of
verbatim text in this conversation starts with a header such as [@3 inventory/client.py#DEFAULT_TIMEOUT];
@3 is its alias. Older material may have been folded into summaries, shown as [fold @A] with
lines like "- (@B) text [@3 @7]". A summary line is not evidence: only verbatim text is.

Tools:
- ctx_search finds spans by keyword and returns their aliases (no content).
- ctx_refetch brings spans (or everything behind a summary line) back verbatim.
- ctx_fold replaces a contiguous run of old material with your own short summary.
- ctx_status reports the context budget, fold candidates and pending obligations.

Answer only from verbatim text. Cite each claim with the alias of the span you rely on and an
exact quote copied from it (at least 20 characters, or the whole span if it is shorter). If a
verification message says you no longer hold some text verbatim, call ctx_refetch and answer
again. If the material does not contain the answer, abstain.

End your turn with exactly one block, and nothing after it:
<answer>{"claims":[{"text":"...","citations":[{"ref":"@3","quote":"exact text"}]}],"abstain":false}</answer>
To abstain: <answer>{"claims":[],"abstain":true}</answer>
"""
PROMPT_SHA256 = sha256_hex(SYSTEM_PROMPT.encode("utf-8"))

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass(frozen=True)
class ParsedClaim:
    text: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class ParsedAnswer:
    claims: tuple[ParsedClaim, ...]
    abstain: bool


class AnswerFormatError(ValueError):
    """The final answer is not exactly one well-formed <answer> block."""


def _keys(obj: object, required: set[str], optional: set[str], what: str) -> dict:
    if not isinstance(obj, dict) or not required <= set(obj) <= required | optional:
        raise AnswerFormatError(f"{what} must be an object with keys {sorted(required)} (optional {sorted(optional)})")
    return obj


def parse_answer(text: str) -> ParsedAnswer:
    """Parse the model's final text strictly: exactly one <answer>...</answer> whose contents
    are JSON {"claims": [{"text": str, "citations": [{"ref": str, "quote": str|null}]}],
    "abstain": bool}, with no other keys. Raises AnswerFormatError otherwise."""
    found = _ANSWER.findall(text)
    if len(found) != 1:
        raise AnswerFormatError(f"expected exactly one <answer> block, found {len(found)}")
    try:
        data = json.loads(found[0])
    except json.JSONDecodeError as exc:
        raise AnswerFormatError(f"the answer is not valid JSON: {exc.msg}") from None
    _keys(data, {"claims", "abstain"}, set(), "the answer")
    if not isinstance(data["abstain"], bool) or not isinstance(data["claims"], list):
        raise AnswerFormatError("abstain must be a boolean and claims a list")
    claims = []
    for claim in data["claims"]:
        _keys(claim, {"text", "citations"}, set(), "a claim")
        if not isinstance(claim["text"], str) or not isinstance(claim["citations"], list):
            raise AnswerFormatError("a claim needs a text string and a citations list")
        citations = []
        for citation in claim["citations"]:
            _keys(citation, {"ref"}, {"quote"}, "a citation")
            quote = citation.get("quote")
            if not isinstance(citation["ref"], str) or not (quote is None or isinstance(quote, str)):
                raise AnswerFormatError("a citation needs a ref string and a quote string or null")
            citations.append(Citation(citation["ref"], quote))
        claims.append(ParsedClaim(claim["text"], tuple(citations)))
    return ParsedAnswer(tuple(claims), data["abstain"])


def answer_text(claims: list[dict], abstain: bool = False) -> str:
    """Format an answer block (used by scripted models in tests and the demo)."""
    return "<answer>" + json.dumps({"claims": claims, "abstain": abstain}, ensure_ascii=False) + "</answer>"
