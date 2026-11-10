"""What the model actually sees: segment rendering and rendered-token accounting.

Blobs hold only verbatim text (so content addressing is unchanged); headers are derived from
the ledger. A span segment whose span has an alias renders as "[@3 <uri>#<label>]" plus a
newline plus its verbatim text; every other segment renders as its blob text. All M2 budget
and payoff arithmetic counts tokens of these rendered strings, with the estimator in force
(tokens.py; bytes4-v0 unless the ledger has a `tokenizer_set` event).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from .envelope import block_tokens
from .model import PyAnchor, Segment
from .store.blobs import BlobStore
from .tokens import BYTES4_V0, Bytes4Estimator, envelope_of, estimator_for, segment_class
from .window import WindowState


class TokenCounter(Protocol):
    """Counts tokens of a rendered string of a segment class ("python", "prose", "json",
    "other"). `name` is recorded in policies so P9 can re-run. The estimators of tokens.py are
    TokenCounters."""

    name: str

    def count(self, text: str, cls: str = "other") -> int: ...


ApproxTokenCounter = Bytes4Estimator  # ceil(UTF-8 bytes / 4), policy name "approx-bytes4"
APPROX = BYTES4_V0
_COUNTERS: dict[str, TokenCounter] = {APPROX.name: APPROX}


def counter_for(name: str) -> TokenCounter:
    """The coefficient-free token counter recorded as `name` in a policy; ValueError if unknown
    (calib-v1 carries coefficients, so it comes from the ledger's estimator in force instead)."""
    try:
        return _COUNTERS[name]
    except KeyError:
        raise ValueError(f"unknown token counter {name!r}") from None


def span_label(state: WindowState, span_id: str) -> str:
    """"<uri>#<qualname>" for a Python span, "<uri>#<start>-<end>" for a text span."""
    span = state.spans[span_id]
    uri = state.sources[span.source_id].uri
    anchor = span.anchor
    return f"{uri}#{anchor.qualname}" if isinstance(anchor, PyAnchor) else f"{uri}#{anchor.start}-{anchor.end}"


def render_segment(state: WindowState, segment: Segment, blobs: BlobStore) -> str:
    """The exact text the request carries for `segment`: an alias header line for a span
    segment whose span has an alias, then the blob text."""
    text = blobs.get(segment.text_hash).decode("utf-8")
    if segment.ref_kind == "span":
        alias = state.aliases.alias_of(segment.ref)
        if alias is not None:
            return f"[{alias} {span_label(state, segment.ref)}]\n{text}"
    return text


def rendered_segment_tokens(state: WindowState, segment: Segment, blobs: BlobStore, counter: TokenCounter | None = None) -> int:
    """Tokens of the rendered segment, by `counter` or else the estimator in force in `state`."""
    return (counter or estimator_for(state)).count(render_segment(state, segment, blobs), segment_class(state, segment))


def rendered_window_tokens(state: WindowState, blobs: BlobStore, counter: TokenCounter | None = None) -> int:
    """Sum of rendered tokens over the window (per segment, as each is sent). Under calib-v3
    (an estimator with an envelope) it is the whole request measure: the envelope base plus
    each segment's `window_costs`, which equals envelope.prompt_tokens of the request the
    builder renders."""
    counter = counter or estimator_for(state)
    env = envelope_of(counter)
    if env is None:
        return sum(rendered_segment_tokens(state, seg, blobs, counter) for seg in state.segments)
    return env.base + sum(window_costs(window_items(state, blobs), counter, env))


@dataclass(frozen=True)
class Item:
    """What the window measure needs of a segment: its role, tool_use_id, content type and
    rendered text (a pending fold's summary has no blob yet, so the text is carried)."""

    role: str
    tool_use_id: str | None
    content_type: str
    text: str

    @property
    def side(self) -> str | None:
        """The message role it renders into, None for a system segment."""
        return None if self.role == "system" else "assistant" if self.role == "assistant" else "user"


def window_items(state: WindowState, blobs: BlobStore) -> list[Item]:
    return [Item(seg.role, seg.tool_use_id, seg.content_type, render_segment(state, seg, blobs)) for seg in state.segments]


def fold_item(summary_text: str) -> Item:
    """The item of a fold segment (a user text block)."""
    return Item("user", None, "text", summary_text)


def window_costs(items: list[Item], counter, env) -> list[int]:
    """Per item, the tokens it adds to the request (envelope.py): a system text block; or, for
    a message segment, a message envelope if it opens a message (its side differs from the
    previous non-system item's), a tool_result envelope if it opens a tool_result (the
    previous non-system item is not a result of the same tool use), and its blocks."""
    out, prev = [], None
    for item in items:
        if item.side is None:
            out.append(block_tokens({"type": "text", "text": item.text}, counter, env))
            continue
        cost = env.message if prev is None or prev.side != item.side else 0
        if item.tool_use_id is not None:
            if prev is None or prev.tool_use_id != item.tool_use_id:
                cost += env.tool_result
            cost += block_tokens({"type": "text", "text": item.text}, counter, env, inner=True)
        elif item.content_type == "blocks":
            cost += sum(block_tokens(b, counter, env) for b in json.loads(item.text))
        else:
            cost += block_tokens({"type": "text", "text": item.text}, counter, env)
        out.append(cost)
        prev = item
    return out


def render_request(state: WindowState, blobs: BlobStore) -> dict:
    """The provider-neutral request a bare `Session.call()` records: pinned system texts, then
    one message per non-system segment (assistant stays assistant; user, tool and fold
    segments are user content), each with its rendered text, or its content blocks."""
    system, messages = [], []
    for seg in state.segments:
        text = render_segment(state, seg, blobs)
        if seg.role == "system":
            system.append(text)
        else:
            content = json.loads(text) if seg.content_type == "blocks" else text
            messages.append({"role": "assistant" if seg.role == "assistant" else "user", "content": content})
    return {"system": system, "messages": messages}


def content_blocks(segment: Segment, blobs: BlobStore) -> list[dict]:
    """The content blocks of a "blocks" segment (its blob is their canonical JSON)."""
    return json.loads(blobs.get(segment.text_hash).decode("utf-8"))


def empty_text_block(block: dict) -> bool:
    """A text block whose text is empty or whitespace only: the API refuses one in a message
    (and a cache_control on one), so the request builder never sends it (BUILD-V3FIX)."""
    return block.get("type") == "text" and not str(block.get("text", "")).strip()


def tool_use_ids(segment: Segment, blobs: BlobStore) -> list[str]:
    """Ids of the tool_use blocks of an assistant "blocks" segment, in order."""
    if segment.content_type != "blocks":
        return []
    return [b["id"] for b in content_blocks(segment, blobs) if b.get("type") == "tool_use"]


def tool_blocks(state: WindowState, blobs: BlobStore) -> list[tuple[int, int]]:
    """(start, end) of each position's tool block: an assistant segment with tool_use blocks and
    every tool-result segment answering it form one block, from the tool use to its last
    result; any other position is a block of its own. A fold, an evict or a backstop run must
    contain a whole block or none of it."""
    n = len(state.segments)
    spans = [(p, p + 1) for p in range(n)]
    used_at: dict[str, int] = {}
    for p, seg in enumerate(state.segments):
        for tool_use_id in tool_use_ids(seg, blobs):
            used_at[tool_use_id] = p
    ends: dict[int, int] = {}
    for p, seg in enumerate(state.segments):
        if seg.tool_use_id in used_at:
            start = used_at[seg.tool_use_id]
            ends[start] = max(ends.get(start, start + 1), p + 1)
    for start, end in ends.items():
        for p in range(start, end):
            spans[p] = (min(start, spans[p][0]), max(end, spans[p][1]))
    return spans


def splits_tool_block(state: WindowState, blobs: BlobStore, removed: set[str], insert_at: int | None = None) -> bool:
    """True if removing the segment ids `removed` (and inserting one segment at index
    `insert_at` of what remains) would separate a tool use from any of its results."""
    segments = state.segments
    for start, end in set(tool_blocks(state, blobs)):
        inside = {seg.segment_id in removed for seg in segments[start:end]}
        if len(inside) > 1:
            return True
    if insert_at is None:
        return False
    remaining = [p for p, seg in enumerate(segments) if seg.segment_id not in removed]
    if 0 < insert_at < len(remaining):
        before, after = remaining[insert_at - 1], remaining[insert_at]
        blocks = tool_blocks(state, blobs)
        return blocks[before] == blocks[after] and blocks[before][1] - blocks[before][0] > 1
    return False
