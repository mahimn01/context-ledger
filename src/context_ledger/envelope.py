"""The envelope model (BUILD-V2 item 4): what a prompt costs beyond the text it carries.

Definition (v2). The window W bounds the whole input prompt as count_tokens counts it, minus
the arm's fixed tool definitions (a constant per arm, measured with and without `tools` and
recorded in run.json). So W holds the system prompt, every message, every content-block
envelope and the thinking blocks kept in history. M4 counted only the text of the pieces, so
the tool_use/tool_result envelopes went uncounted (L's p95 actual prompt was about 9.2k at
W = 6000).

The measure of a request (system blocks and messages; never the tools):

    base                                   once per request
    + message                              per message
    + text + tokens(text)                  per text block (system, message and tool_result
                                           inner text blocks alike)
    + tool_use + tokens(name + " " + canonical_json(input), json)    per tool_use block
    + tool_result                          per tool_result block (its inner text blocks count
                                           as text blocks)
    + thinking + tokens(thinking text)     per thinking block (redacted_thinking: thinking only)

with tokens(...) the estimator's content count (calib-v3: the calib-v2 point slope per piece).
A text block is classed by its header: "[@3 <path>#<label>]" (an L span) and "[<path>:a-b]"
(a corpus read) take their source's class (tokens.source_class), anything else is "other" in a
message and "json" inside a tool_result. B3's "<context>" block, the concatenation of several
read chunks, is counted chunk by chunk, each by its own header.

The constants are integers fitted by `calibrate --envelope` (eval/calibrate.py) against the
free count_tokens endpoint. tool_use and tool_result always come in pairs in a valid request,
so only their sum is identifiable: the fit attributes it to tool_use (tool_result is 0).
`prompt_tokens` measures any request; the ledger arm measures its window segment by segment
(render.window_items), and the two agree exactly on the request the builder renders.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

from .ids import canonical_json, sha256_hex
from .tokens import CALIBRATION_DIR, source_class

KINDS = ("base", "message", "text", "tool_use", "tool_result", "thinking")
_SPAN_HEADER = re.compile(r"\[@[0-9A-Z]+ ([^\]\n#]+)#[^\]\n]*\]")
_READ_HEADER = re.compile(r"\[([^\]\n:]+):\d+-\d+\]")
_READ_LINE = re.compile(r"^\[[^\]\n:]+:\d+-\d+\]$", re.M)
CONTEXT_OPEN = "<context>\n"


class EnvelopeError(ValueError):
    """An envelope payload or file is malformed."""


class EnvelopeMissing(EnvelopeError):
    """No envelope calibration exists for the model (run `calibrate --live --envelope`)."""


@dataclass(frozen=True)
class Envelope:
    """Structural token constants, all ints >= 0 (see the module docstring)."""

    base: int
    message: int
    text: int
    tool_use: int
    tool_result: int
    thinking: int

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise EnvelopeError(f"envelope.{f.name} must be an int >= 0, got {value!r}")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in KINDS}

    @classmethod
    def from_dict(cls, d: object) -> Envelope:
        if not isinstance(d, dict) or set(d) != set(KINDS):
            raise EnvelopeError(f"an envelope has exactly the keys {list(KINDS)}")
        return cls(**{k: d[k] for k in KINDS})

    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))


# Offline fakes only (BUILD-V2: "offline fakes may use a default"); a live run refuses to start
# without a fitted envelope file. Plausible magnitudes, not measurements.
OFFLINE_ENVELOPE = Envelope(base=4, message=4, text=3, tool_use=12, tool_result=0, thinking=4)


def text_class(text: str, default: str) -> str:
    """The class of a text block by its header line (see the module docstring)."""
    first = text.split("\n", 1)[0]
    m = _SPAN_HEADER.fullmatch(first) or _READ_HEADER.fullmatch(first)
    return source_class(m.group(1)) if m else default


def text_pieces(text: str, default: str) -> list[tuple[str, str]]:
    """(piece, class) of a text block: one piece, classed by `text_class`, except a B3
    "<context>" block, which is cut before each read-header line and each piece classed by
    its own header (the leading "<context>" line is `default`)."""
    if not text.startswith(CONTEXT_OPEN):
        return [(text, text_class(text, default))]
    cuts = [0, *(m.start() for m in _READ_LINE.finditer(text) if m.start() > 0)]
    pieces = [text[a:b] for a, b in zip(cuts, [*cuts[1:], len(text)])]
    return [(p, text_class(p, default)) for p in pieces if p]


def text_tokens(text: str, counter, default: str) -> int:
    """Content tokens of a text block (`text_pieces`, each counted by `counter`)."""
    return sum(counter.count(piece, cls) for piece, cls in text_pieces(text, default))


def tool_use_text(block: dict) -> str:
    """What a tool_use block's content is counted as: its name and canonical JSON input."""
    return block["name"] + " " + canonical_json(block.get("input", {})).decode("utf-8")


def inner_blocks(block: dict) -> list[dict]:
    """The content blocks of a tool_result (a string content is one text block)."""
    content = block.get("content")
    if content is None:
        return []
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def block_tokens(block: dict, counter, env: Envelope, *, inner: bool = False) -> int:
    """Tokens of one content block, envelope included."""
    kind = block.get("type")
    if kind == "text":
        return env.text + text_tokens(block["text"], counter, "json" if inner else "other")
    if kind == "tool_use":
        return env.tool_use + counter.count(tool_use_text(block), "json")
    if kind == "tool_result":
        return env.tool_result + sum(block_tokens(b, counter, env, inner=True) for b in inner_blocks(block))
    if kind == "thinking":
        return env.thinking + counter.count(block.get("thinking", ""), "other")
    if kind == "redacted_thinking":
        return env.thinking
    return counter.count(canonical_json(block).decode("utf-8"), "json")


def prompt_tokens(request: dict, counter, env: Envelope) -> int:
    """The W measure of a request: base, its system blocks and its messages (never its tools)."""
    total = env.base + sum(block_tokens(b, counter, env) for b in request.get("system", []))
    for message in request.get("messages", []):
        content = message["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        total += env.message + sum(block_tokens(b, counter, env) for b in blocks)
    return total




@dataclass(frozen=True)
class EnvelopeCalibration:
    """A loaded envelope file: the constants, each arm's tool-definition tokens (count_tokens
    with minus without `tools`) and the sha256 of its tool list, the calib-v1 calibration whose
    slopes the fit used for content, and the file's own sha256."""

    envelope: Envelope
    tool_definitions: dict
    tools_sha256: dict
    calibration_sha256: str
    sha256: str


def envelope_calibration(data: dict, *, allow_fake: bool = False) -> EnvelopeCalibration:
    """The EnvelopeCalibration of an envelope file's contents (refusing a fake-counter fit
    unless `allow_fake`)."""
    if not isinstance(data, dict) or data.get("version") != "envelope-v1":
        raise EnvelopeError("not an envelope-v1 calibration file")
    if data.get("counter") != "count_tokens" and not allow_fake:
        raise EnvelopeMissing(f"envelope was fitted with the {data.get('counter')!r} counter, not count_tokens")
    defs = data.get("tool_definitions")
    if not isinstance(defs, dict) or not all(isinstance(v, dict) and isinstance(v.get("tokens"), int) for v in defs.values()):
        raise EnvelopeError("tool_definitions must map each arm to {tokens, tools_sha256}")
    return EnvelopeCalibration(
        Envelope.from_dict(data.get("envelope")),
        {arm: v["tokens"] for arm, v in defs.items()},
        {arm: v.get("tools_sha256") for arm, v in defs.items()},
        str(data.get("calibration_sha256")),
        sha256_hex(canonical_json(data)),
    )


def envelope_path(model: str, created: str, directory=None):
    """calibration/<model>-envelope-<YYYYMMDD>.json (a separate file from calib-v1's)."""
    return Path(directory if directory is not None else CALIBRATION_DIR) / f"{model}-envelope-{created.replace('-', '')}.json"


def load_envelope(model: str, directory=None, *, allow_fake: bool = False) -> EnvelopeCalibration:
    """The newest calibration/<model>-envelope-<YYYYMMDD>.json. Raises EnvelopeMissing, naming
    where it looked, if there is none."""
    folder = Path(directory) if directory is not None else CALIBRATION_DIR
    found = sorted(folder.glob(f"{model}-envelope-[0-9]*.json")) if folder.is_dir() else []
    if not found:
        raise EnvelopeMissing(f"no envelope calibration for {model} in {folder}; run `calibrate --live --counter count_tokens --envelope --model {model}` first")
    return envelope_calibration(json.loads(found[-1].read_text(encoding="utf-8")), allow_fake=allow_fake)


__all__ = [
    "CONTEXT_OPEN",
    "Envelope",
    "EnvelopeCalibration",
    "EnvelopeError",
    "EnvelopeMissing",
    "KINDS",
    "OFFLINE_ENVELOPE",
    "block_tokens",
    "envelope_calibration",
    "envelope_path",
    "inner_blocks",
    "load_envelope",
    "prompt_tokens",
    "text_class",
    "text_pieces",
    "text_tokens",
    "tool_use_text",
]
