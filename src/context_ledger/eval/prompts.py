"""The pilot's versioned prompts and their strict final-answer parsers.

The system prompt is identical across the three arms except for the paragraph describing the
tools that arm has. Each arm's exact prompt text has its own sha256, recorded in the run
manifest (and, for the ledger arm, in every model_call's build params).

Versions (PROMPTS maps each to its frame, parser and answer formatter):
    clp-m3-pilot-1   M3: one <answer>{"answer", "claims", "abstain"}</answer> JSON block. Kept
                     to reproduce the 2026-10-31 smoke runs.
    clp-m4-pilot-v2  M4: the answer stated plainly on one line, `ANSWER: <value>` or
                     `ANSWER: UNANSWERABLE`, then the claims as a JSON list in one
                     <claims>...</claims> block. Each claim states its subject and value plainly
                     (`mod.f` parameter `p` defaults to `3`) and quotes the verbatim defining
                     line(s). Same answer-format text for B3, B4 and L. Kept to reproduce the
                     M4 grid.
    clp-v2-pilot-1   BUILD-V2, the default: clp-m4-pilot-v2 with one sentence added to L's tool
                     paragraph only ("Before answering UNANSWERABLE, check ctx_search and
                     ctx_refetch for folded material."). The frame and the answer-format text
                     are identical across arms and to clp-m4-pilot-v2.
    clp-v2-pilot-2   BUILD-V2b, selectable with --prompt: clp-v2-pilot-1 with a neutral
                     abstention rule, in every arm alike: "If you are not sure, abstain. Never
                     guess." is deleted, and "If the text shown does not answer the question, or
                     you are not sure, write" becomes "If the corpus does not contain the answer,
                     write". L keeps its extra tool-paragraph sentence.
    clp-v2-pilot-3   BUILD-V2c, selectable with --prompt: clp-v2-pilot-2 plus one neutral
                     sentence, in every arm alike, for the session facts of questions v2s:
                     "Notes from the user can be updated later, and the latest value is the one
                     that counts." (after "Only text shown to you here counts as evidence.").
    clp-v3-allowed   BUILD-V3, the abstain-allowed condition: an alias of clp-v2-pilot-3, the same
                     text and sha256 in every arm.
    clp-v3-forced    BUILD-V3, the forced condition: clp-v2-pilot-3 with the abstention sentence
                     ("If the corpus does not contain the answer, write ANSWER: UNANSWERABLE and
                     <claims>[]</claims>.") replaced, in every arm alike, by FORCED_SENTENCE. The
                     answer format and the parser are unchanged (an UNANSWERABLE still parses, as
                     an abstention). `eval run --condition` picks one of the two.
    clp-v4-allowed   BUILD-V4: clp-v3-allowed and clp-v3-forced with tool paragraphs for the two v4
    clp-v4-forced    arms added (TOOL_TEXT_V4: L4's address tools and B5's). The frames and the
                     paragraphs of B3, B4 and L are byte-identical to v3's, so those arms' prompts
                     and sha256s are v3's; only the tool descriptions differ. A run with a v4 arm
                     uses these (CONDITION_PROMPTS_V4).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..ids import sha256_hex
from ..model import Citation
from ..prompts import AnswerFormatError, ParsedClaim, _keys

PILOT_PROMPT_V1 = "clp-m3-pilot-1"
PILOT_PROMPT_V2 = "clp-m4-pilot-v2"
PILOT_PROMPT_V2_1 = "clp-v2-pilot-1"
PILOT_PROMPT_V2_2 = "clp-v2-pilot-2"
PILOT_PROMPT_V2_3 = "clp-v2-pilot-3"
PILOT_PROMPT_V3_ALLOWED = "clp-v3-allowed"
PILOT_PROMPT_V3_FORCED = "clp-v3-forced"
CONDITION_PROMPTS = {"allowed": PILOT_PROMPT_V3_ALLOWED, "forced": PILOT_PROMPT_V3_FORCED}
PILOT_PROMPT_V4_ALLOWED = "clp-v4-allowed"
PILOT_PROMPT_V4_FORCED = "clp-v4-forced"
CONDITION_PROMPTS_V4 = {"allowed": PILOT_PROMPT_V4_ALLOWED, "forced": PILOT_PROMPT_V4_FORCED}
FORCED_PROMPTS = (PILOT_PROMPT_V3_FORCED, PILOT_PROMPT_V4_FORCED)
PILOT_PROMPT_VERSION = PILOT_PROMPT_V2_1  # the default for new runs
COMPACTION_PROMPT_VERSION = "clp-m3-compact-1"

_FRAME = """\
You answer questions about the source tree of CPython 3.15. It is a modified copy: names and
values differ from the real CPython, so never answer from memory. Only text shown to you here
counts as evidence.

{tools}

Every piece of source text shown to you starts with a header in square brackets. Cite the ref
from that header: the @alias if the header has one ([@3 Lib/x.py#f] gives @3), otherwise the
path and line range ([Lib/x.py:10-70] gives Lib/x.py:10-70). Quote the exact text you rely on,
copied verbatim (at least 20 characters, or the whole piece if it is shorter).

If you are not sure, abstain. Never guess.

End your turn with exactly one block, and nothing after it:
<answer>{{"answer": ..., "claims": [{{"text": "...", "citations": [{{"ref": "...", "quote": "exact text"}}]}}], "abstain": false}}</answer>
"answer" is the bare answer: a Python literal for a value (41, 2.5 or 'utf-8'), a list of
exception class names (["ValueError"]), or a dotted module name ("email.utils").
To abstain: <answer>{{"answer": null, "claims": [], "abstain": true}}</answer>
"""

TOOL_TEXT = {
    "B3": "No tools: the source chunks most relevant to the question are included in the message.",
    "B4": (
        "Tools: corpus_search finds source chunks by keywords and returns path:start-end line ranges with one\n"
        "line of each; corpus_read returns lines of a file (at most 200 lines per call). When the\n"
        "conversation grows long, its older part is replaced by a summary."
    ),
    "L": (
        "Tools: corpus_search finds source chunks by keywords and returns path:start-end line ranges with one\n"
        "line of each; corpus_read brings lines of a file into the conversation as verbatim spans, each\n"
        "with an alias. Older material may be folded into summaries, shown as [fold @A] with lines like\n"
        '"- (@B) text [@3 @7]"; a summary line is not evidence, only verbatim text is. ctx_refetch brings\n'
        "spans (or everything behind a summary line) back verbatim; ctx_search finds spans you have seen;\n"
        "ctx_fold replaces a run of old material with your own summary; ctx_status reports the budget.\n"
        "If a verification message says you no longer hold some text verbatim, call ctx_refetch and answer again."
    ),
}
# clp-v2-pilot-1: L's tool paragraph gains one sentence; B3 and B4 are unchanged.
TOOL_TEXT_V2_1 = {**TOOL_TEXT, "L": TOOL_TEXT["L"] + "\nBefore answering UNANSWERABLE, check ctx_search and ctx_refetch for folded material."}

# clp-v4-*: the v3 paragraphs unchanged, plus L4's and B5's (BUILD-V4 item 7).
_LOOKUP_TEXT = (
    "lookup finds the exact address of a name (path:start-end#hash): a dotted name such as\n"
    "mod.Class.method or mod.CONSTANT, a bare function name, or a file path; a name it does not\n"
    "find exactly is reported with the nearest DIFFERENT identifiers, which are never the one asked."
)
TOOL_TEXT_V4 = {
    **TOOL_TEXT_V2_1,
    "L4": (
        f"Tools: {_LOOKUP_TEXT}\n"
        "read brings the lines of an address into the conversation as verbatim spans, each with an\n"
        "alias; outline lists what a directory, file or class contains, with addresses; search finds\n"
        "source chunks by keywords, at most twice per question (prefer lookup). With each question the\n"
        "harness looks up the name it asks about and shows that text with its address, or says that\n"
        "the name has no exact address. Older material may be folded into summaries, shown as [fold @A] with lines like\n"
        '"- (@B) text [@3 @7]"; a summary line is not evidence, only verbatim text is. ctx_refetch brings\n'
        "spans (or everything behind a summary line) back verbatim; ctx_search finds spans you have seen;\n"
        "ctx_fold replaces a run of old material with your own summary; ctx_status reports the budget.\n"
        "If a verification message says you no longer hold some text verbatim, call ctx_refetch and answer again;\n"
        "if a repair message shows you the text the question asks about, cite that text and answer again.\n"
        "Before answering UNANSWERABLE, check ctx_search and ctx_refetch for folded material."
    ),
    "B5": (
        f"Tools: {_LOOKUP_TEXT}\n"
        "read returns the lines of an address (at most 200 lines per call); outline lists what a\n"
        "directory, file or class contains, with addresses. With each question the harness looks up\n"
        "the name it asks about and shows that text with its address, or says that the name has no\n"
        "exact address; the source chunks most relevant to the question are included in the message too."
    ),
}


_FRAME_V2 = """\
You answer questions about the source tree of CPython 3.15. It is a modified copy: names and
values differ from the real CPython, so never answer from memory. Only text shown to you here
counts as evidence.

{tools}

Every piece of source text shown to you starts with a header in square brackets. Cite the ref
from that header: the @alias if the header has one ([@3 Lib/x.py#f] gives @3), otherwise the
path and line range ([Lib/x.py:10-70] gives Lib/x.py:10-70).

If you are not sure, abstain. Never guess.

End your turn with exactly these two parts, and nothing after them:
ANSWER: <the answer, plainly, on this one line>
<claims>[{{"text": "...", "citations": [{{"ref": "...", "quote": "..."}}]}}]</claims>

The ANSWER line holds only the bare answer: a Python literal for a value (41, 2.5 or 'utf-8'),
exception class names separated by commas (ValueError, KeyError), or a dotted module name
(email.utils). If the text shown does not answer the question, or you are not sure, write
ANSWER: UNANSWERABLE and <claims>[]</claims>.
Each claim states its subject and the value plainly, for example
"`email.utils.formatdate` parameter `usegmt` defaults to `False`", and cites the text it rests on
with a quote that is the verbatim defining line or lines, copied exactly (at least 20
characters, or the whole piece if it is shorter).
"""

# clp-v2-pilot-2: the same frame without the "not sure" rule (BUILD-V2b item 4).
_FRAME_V2_2 = _FRAME_V2.replace("If you are not sure, abstain. Never guess.\n\n", "").replace(
    "If the text shown does not answer the question, or you are not sure, write", "If the corpus does not contain the answer, write"
)

# clp-v2-pilot-3: clp-v2-pilot-2 plus the session-facts sentence (BUILD-V2c).
NOTES_SENTENCE = "Notes from the user can be updated later, and the latest value is the one that counts."
_FRAME_V2_3 = _FRAME_V2_2.replace("counts as evidence.\n", f"counts as evidence. {NOTES_SENTENCE}\n", 1)

# clp-v3-forced: clp-v2-pilot-3 with its abstention sentence replaced (BUILD-V3 item 2).
ABSTENTION_SENTENCE = "If the corpus does not contain the answer, write\nANSWER: UNANSWERABLE and <claims>[]</claims>."
FORCED_SENTENCE = (
    "Always give your best answer. Never write UNANSWERABLE: if you are unsure, give your best guess, still in the required format "
    "with claims and verbatim quotes."
)
_FRAME_V3_FORCED = _FRAME_V2_3.replace(ABSTENTION_SENTENCE, FORCED_SENTENCE, 1)


def system_prompt(arm: str, version: str = PILOT_PROMPT_VERSION) -> str:
    """The exact system prompt of arm B3, B4 or L under prompt `version`."""
    return PROMPTS[version].frame.format(tools=PROMPTS[version].tools[arm])


def prompt_sha256(arm: str, version: str = PILOT_PROMPT_VERSION) -> str:
    return sha256_hex(system_prompt(arm, version).encode("utf-8"))


COMPACTION_PROMPT = """\
You compress a working conversation so that it can continue in less space. Summarize the
conversation the user gives you: the questions asked and the answers given, every file path and
line range read, and every symbol name and literal value found, copied exactly. Plain text, at
most 250 words. Do not answer any question yourself.
"""
COMPACTION_SHA256 = sha256_hex(COMPACTION_PROMPT.encode("utf-8"))

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass(frozen=True)
class PilotAnswer:
    """A parsed final answer: the bare `answer` value, the claims with citations, abstain."""

    answer: object
    claims: tuple[ParsedClaim, ...]
    abstain: bool


def parse_pilot_answer(text: str) -> PilotAnswer:
    """Parse strictly: exactly one <answer> block holding JSON with the keys answer, claims and
    abstain (claims as in M2: text and citations of ref and quote). AnswerFormatError otherwise."""
    found = _ANSWER.findall(text)
    if len(found) != 1:
        raise AnswerFormatError(f"expected exactly one <answer> block, found {len(found)}")
    try:
        data = json.loads(found[0])
    except json.JSONDecodeError as exc:
        raise AnswerFormatError(f"the answer is not valid JSON: {exc.msg}") from None
    _keys(data, {"answer", "claims", "abstain"}, set(), "the answer")
    if not isinstance(data["abstain"], bool) or not isinstance(data["claims"], list):
        raise AnswerFormatError("abstain must be a boolean and claims a list")
    return PilotAnswer(data["answer"], _claims_list(data["claims"]), data["abstain"])


def pilot_answer_text(answer: object, claims: list[dict], abstain: bool = False) -> str:
    """Format a v1 pilot answer block (the scripted fake model uses it)."""
    return "<answer>" + json.dumps({"answer": answer, "claims": claims, "abstain": abstain}, ensure_ascii=False) + "</answer>"


UNANSWERABLE = "UNANSWERABLE"
_ANSWER_LINE = re.compile(r"^[ \t]*ANSWER:[ \t]*(.*?)[ \t]*$", re.MULTILINE)
_CLAIMS = re.compile(r"<claims>(.*?)</claims>", re.DOTALL)


def _claims_list(items: object) -> tuple[ParsedClaim, ...]:
    if not isinstance(items, list):
        raise AnswerFormatError("claims must be a JSON list")
    claims = []
    for claim in items:
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
    return tuple(claims)


def parse_pilot_answer_v2(text: str) -> PilotAnswer:
    """Parse a v2 answer strictly: exactly one `ANSWER:` line and at most one <claims> block
    holding a JSON list of claims (text and citations of ref and quote); the block may be left
    out only when the answer is UNANSWERABLE, which abstains (any claims are then ignored). The
    answer is the line's text with one pair of surrounding backticks removed, as a string: the
    scorer reads literals, exception names and module paths from it. AnswerFormatError
    otherwise."""
    lines = _ANSWER_LINE.findall(text)
    if len(lines) != 1:
        raise AnswerFormatError(f"expected exactly one ANSWER: line, found {len(lines)}")
    value = lines[0].strip()
    if len(value) > 1 and value[0] == value[-1] == "`":
        value = value[1:-1].strip()
    if not value:
        raise AnswerFormatError("the ANSWER: line is empty")
    blocks = _CLAIMS.findall(text)
    if len(blocks) > 1:
        raise AnswerFormatError(f"expected at most one <claims> block, found {len(blocks)}")
    abstain = value.upper() == UNANSWERABLE
    if not blocks:
        if not abstain:
            raise AnswerFormatError("an answer needs a <claims> block after the ANSWER: line")
        return PilotAnswer(None, (), True)
    if text.index("<claims>") < _ANSWER_LINE.search(text).start():
        raise AnswerFormatError("the ANSWER: line must come before the <claims> block")
    try:
        data = json.loads(blocks[0])
    except json.JSONDecodeError as exc:
        raise AnswerFormatError(f"the claims are not valid JSON: {exc.msg}") from None
    claims = _claims_list(data)
    return PilotAnswer(None, (), True) if abstain else PilotAnswer(value, claims, False)


def pilot_answer_text_v2(answer: object, claims: list[dict], abstain: bool = False) -> str:
    """Format a v2 answer (the scripted fake model uses it): a list is written comma-separated,
    anything else as its text."""
    if abstain:
        shown, claims = UNANSWERABLE, []
    elif isinstance(answer, list):
        shown = ", ".join(str(a) for a in answer)
    else:
        shown = str(answer)
    return f"ANSWER: {shown}\n<claims>" + json.dumps(claims, ensure_ascii=False) + "</claims>"


@dataclass(frozen=True)
class PromptVersion:
    frame: str
    parse: Callable[[str], PilotAnswer]
    answer_text: Callable[..., str]
    reformat: str  # what a reformat request asks for
    tools: dict[str, str] = field(default_factory=lambda: TOOL_TEXT)  # each arm's tool paragraph


_REFORMAT_V2 = "End your turn with exactly one ANSWER: line followed by one <claims>[...]</claims> block."
PROMPTS = {
    PILOT_PROMPT_V1: PromptVersion(_FRAME, parse_pilot_answer, pilot_answer_text, "End your turn with exactly one <answer>{...}</answer> block."),
    PILOT_PROMPT_V2: PromptVersion(_FRAME_V2, parse_pilot_answer_v2, pilot_answer_text_v2, _REFORMAT_V2),
    PILOT_PROMPT_V2_1: PromptVersion(_FRAME_V2, parse_pilot_answer_v2, pilot_answer_text_v2, _REFORMAT_V2, TOOL_TEXT_V2_1),
    PILOT_PROMPT_V2_2: PromptVersion(_FRAME_V2_2, parse_pilot_answer_v2, pilot_answer_text_v2, _REFORMAT_V2, TOOL_TEXT_V2_1),
    PILOT_PROMPT_V2_3: PromptVersion(_FRAME_V2_3, parse_pilot_answer_v2, pilot_answer_text_v2, _REFORMAT_V2, TOOL_TEXT_V2_1),
    PILOT_PROMPT_V3_FORCED: PromptVersion(_FRAME_V3_FORCED, parse_pilot_answer_v2, pilot_answer_text_v2, _REFORMAT_V2, TOOL_TEXT_V2_1),
}
PROMPTS[PILOT_PROMPT_V3_ALLOWED] = PROMPTS[PILOT_PROMPT_V2_3]  # an alias: the same text, so the same sha256
PROMPTS[PILOT_PROMPT_V4_ALLOWED] = PromptVersion(_FRAME_V2_3, parse_pilot_answer_v2, pilot_answer_text_v2, _REFORMAT_V2, TOOL_TEXT_V4)
PROMPTS[PILOT_PROMPT_V4_FORCED] = PromptVersion(_FRAME_V3_FORCED, parse_pilot_answer_v2, pilot_answer_text_v2, _REFORMAT_V2, TOOL_TEXT_V4)


def prompt_arms(version: str) -> list[str]:
    """The arms prompt `version` has a tool paragraph for (B3, B4 and L; and B5 and L4 for v4)."""
    return [arm for arm in ("B3", "B4", "L", "B5", "L4") if arm in PROMPTS[version].tools]
