"""Pure, deterministic Anthropic Messages request builder. No SDK import: it emits plain dicts.

The window becomes `system` text blocks (pinned system segments) and messages (everything else,
adjacent same-role segments merged). Segments with a tool_use_id become tool_result blocks in
the user message right after the assistant message holding that tool use. Up to four
`cache_control` breakpoints are placed, in priority order:

    BP1  the last system block (tools + system); the last tool definition if there is no system
    BP2  the last block before the first fold-candidate position (the frozen prefix the
         backstop can never change)
    BP3  the last block of the window as it was at the previous model call (incremental
         conversation caching)
    BP4  the last block of this request

A breakpoint whose prefix has fewer tokens than the model's `min_cache_tokens` is skipped.
Prefix token counts are the estimator in force (tokens.py) applied to each block's canonical
JSON, classed by the segment it renders (tool definitions are json); under bytes4-v0 that is
approx_tokens, the same measure the FakeAnthropicServer simulates with. They are an
approximation, not Anthropic's tokenizer.

Empty text blocks (BUILD-V3FIX, harness patch v3.1-empty-text-cache-control). The API refuses a
message text block that is empty or whitespace only ("text content blocks must be non-empty"),
and a breakpoint on one ("cache_control cannot be set for empty text blocks"). So a message's
empty text blocks are dropped before the breakpoints are placed; a message that would be left
with no block keeps one text block reading EMPTY_TEXT_PLACEHOLDER, "(empty)", in place of its
last empty one (B4 already sends an empty reply that way). A breakpoint meant for a segment
that rendered into nothing goes on the nearest earlier block. A window with no empty text
block renders exactly as before.

`thinking`, if given, is sent as the request's thinking config (Haiku 5.5: disabled; Sonnet
5.5: between_tools, see PILOT_THINKING). Sampling parameters, server-side context editing,
service_tier and inference_geo are never sent.

Thinking blocks (BUILD-V4FIX, harness patch v4.1-thinking-binding). Since 2026-11-02 the API
binds a thinking block to the content before it when it was created: sent back after content
that was rewritten since (L's backstop or ctx_fold folds earlier turns into a fold line), it is
refused with a 400 ("... The block is bound to a different conversation. Remove the block ...").
So the client records, for each thinking or redacted_thinking block of a response, the digest of
the messages before it as the next request renders them (`thinking_bindings`, kept in the
model_call event as "thinking_bound" and replayed into WindowState.thinking_bound), and the
builder sends such a block back only while the messages before it, as this request renders
them, still have that digest; otherwise the block is left out (never altered). An assistant
message left with no block keeps EMPTY_TEXT_PLACEHOLDER. A window whose ledger never recorded a
binding (thinking disabled, as on Haiku, or a ledger written before the patch) renders exactly
as before.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..ids import canonical_json, sha256_hex
from ..pricing import PricingTable
from ..render import empty_text_block, render_segment
from ..store.blobs import BlobStore
from ..tokens import estimator_for, segment_class
from ..window import WindowState, approx_tokens

FORBIDDEN_PARAMS = frozenset({"temperature", "top_p", "top_k"})
NEVER_SENT = FORBIDDEN_PARAMS | {"context_management", "inference_geo", "service_tier"}
# The lowest thinking setting each pilot model accepts (A.1): Haiku 5.5 can disable thinking;
# Sonnet 5.5 rejects "disabled" and its lowest setting still returns progress-update thinking
# blocks between tool calls, which must be passed back unchanged.
PILOT_THINKING = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}
TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_BREAKPOINTS = 4
EMPTY_TEXT_PLACEHOLDER = "(empty)"  # BUILD-V3FIX: a message whose every text block is empty is sent as this one block
THINKING_TYPES = ("thinking", "redacted_thinking")


class InvalidWindow(Exception):
    """The window does not render to a valid Messages request."""


def cache_control(ttl: str) -> dict:
    """The breakpoint marker for a cache TTL: 5m is the API default, 1h is explicit."""
    if ttl not in ("5m", "1h"):
        raise ValueError(f"unknown cache ttl {ttl!r}")
    return {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": "1h"}


def strip_cache_control(block: dict) -> dict:
    return {k: v for k, v in block.items() if k != "cache_control"}


def linearize(request: dict) -> list[tuple[str, dict]]:
    """The request's blocks in cache order, tools then system then message content, each as
    (where, block) with where "tools", "system", "user" or "assistant"."""
    out = [("tools", t) for t in request.get("tools", [])]
    out += [("system", b) for b in request.get("system", [])]
    for message in request.get("messages", []):
        out += [(message["role"], b) for b in message["content"]]
    return out


def linear_paths(request: dict) -> list[tuple]:
    """The path of each block of `linearize(request)`: ("tools", i), ("system", i) or
    ("messages", m, b)."""
    paths: list[tuple] = [("tools", i) for i in range(len(request.get("tools", [])))]
    paths += [("system", i) for i in range(len(request.get("system", [])))]
    return paths + [("messages", m, b) for m, msg in enumerate(request.get("messages", [])) for b in range(len(msg["content"]))]


def block_tokens(block: dict) -> int:
    """approx_tokens of the block's canonical JSON, without its cache_control marker."""
    return approx_tokens(canonical_json(strip_cache_control(block)).decode("utf-8"))


def empty_text_problems(request: dict) -> list[str]:
    """The API's 400 invalid_request_error for each empty text block of the system or the
    messages, in its own words: a breakpoint on one ("messages.6.content.0.text: cache_control
    cannot be set for empty text blocks"), else the block itself (empty, or whitespace only)."""
    problems = []
    for path, (where, block) in zip(linear_paths(request), linearize(request)):
        if where == "tools" or not empty_text_block(block):
            continue
        field = "messages" if path[0] == "messages" else "system"
        if "cache_control" in block:
            at = f"messages.{path[1]}.content.{path[2]}" if field == "messages" else f"system.{path[1]}"
            problems.append(f"{at}.text: cache_control cannot be set for empty text blocks")
        else:
            problems.append(f"{field}: text content blocks must {'contain non-whitespace text' if block.get('text') else 'be non-empty'}")
    return problems


def request_problems(request: dict) -> list[str]:
    """Structural problems of a Messages request (empty if valid): forbidden sampling params
    and other parameters the harness never sends, more than four breakpoints, bad tool names,
    an empty text block (with or without a breakpoint: `empty_text_problems`), a first message
    that is not user, a tool_result that does not answer a tool_use of the preceding assistant
    message or is not first in its message, and a tool_use left unanswered before the final
    message."""
    problems = [f"forbidden parameter {p}" for p in sorted(NEVER_SENT & set(request))]
    problems += empty_text_problems(request)
    marks = sum("cache_control" in block for _, block in linearize(request))
    if marks > MAX_BREAKPOINTS:
        problems.append(f"{marks} cache breakpoints, at most {MAX_BREAKPOINTS} allowed")
    problems += [f"bad tool name {t.get('name')!r}" for t in request.get("tools", []) if not TOOL_NAME.match(str(t.get("name", "")))]
    messages = request.get("messages", [])
    if not messages or messages[0]["role"] != "user":
        problems.append("the first message must be a user message")
    for n, message in enumerate(messages):
        content = message["content"]
        if message["role"] == "user":
            results = [b for b in content if b.get("type") == "tool_result"]
            if content[: len(results)] != results:
                problems.append(f"message {n}: tool_result blocks must come first")
            asked = _tool_uses(messages[n - 1]) if n and messages[n - 1]["role"] == "assistant" else set()
            for block in results:
                if block["tool_use_id"] not in asked:
                    problems.append(f"message {n}: tool_result {block['tool_use_id']} answers no tool_use of the previous message")
        elif n + 1 < len(messages):
            answered = {b["tool_use_id"] for b in messages[n + 1]["content"] if b.get("type") == "tool_result"}
            if messages[n + 1]["role"] != "user" or not _tool_uses(message) <= answered:
                problems.append(f"message {n}: a tool_use is not answered by the next message")
    return problems


def _tool_uses(message: dict) -> set[str]:
    return {b["id"] for b in message["content"] if b.get("type") == "tool_use"}


def thinking_key(block: dict) -> str:
    """The sha256 of a thinking block's canonical JSON (cache_control stripped): its signature
    and its text, so an altered block never has a known key."""
    return sha256_hex(canonical_json(strip_cache_control(block)))


def thinking_prefix_digest(messages: Sequence[dict], m: int, b: int) -> str:
    """The sha256 of everything before messages[m].content[b]: the earlier messages and the
    first b blocks of message m, with their roles, cache_control stripped."""
    prefix = [{"role": msg["role"], "content": [strip_cache_control(x) for x in (msg["content"] if n < m else msg["content"][:b])]}
              for n, msg in enumerate(messages[: m + 1])]  # fmt: skip
    return sha256_hex(canonical_json(prefix))


def thinking_bindings(request: dict, content: Sequence[dict]) -> dict[str, str]:
    """{thinking_key: thinking_prefix_digest} of each thinking block of the response `content`
    to `request`, placed as the next request renders it: in a new assistant message after the
    request's messages, or after the last one if that is the assistant's, empty text blocks
    dropped. Empty if the response has no thinking block."""
    blocks = [b for b in content if not empty_text_block(b)]
    if not any(b.get("type") in THINKING_TYPES for b in blocks):
        return {}
    messages = list(request.get("messages", []))
    if messages and messages[-1]["role"] == "assistant":
        offset = len(messages[-1]["content"])
        messages[-1] = {"role": "assistant", "content": [*messages[-1]["content"], *blocks]}
    else:
        offset = 0
        messages.append({"role": "assistant", "content": blocks})
    m = len(messages) - 1
    return {thinking_key(b): thinking_prefix_digest(messages, m, offset + i) for i, b in enumerate(blocks) if b.get("type") in THINKING_TYPES}


def _drop_unbound_thinking(messages: list[dict], paths: list[tuple[tuple, ...]], bound: dict[str, str]) -> list[tuple[tuple, ...]]:
    """Remove, in place and in order, every thinking block whose binding is unknown or whose
    prefix (as this request now renders it, earlier removals included) is not the one it was
    created after; a message left with no block keeps EMPTY_TEXT_PLACEHOLDER in place of its
    last removed one. Returns the paths renumbered, as _drop_empty_text does."""
    moved: dict[tuple, tuple | None] = {}
    for m, message in enumerate(messages):
        if message["role"] != "assistant" or not any(b.get("type") in THINKING_TYPES for b in message["content"]):
            continue
        kept: list[dict] = []
        renumbered: dict[tuple, tuple | None] = {}
        for old, block in enumerate(message["content"]):
            if block.get("type") in THINKING_TYPES and bound.get(thinking_key(block)) != thinking_prefix_digest([*messages[:m], {"role": "assistant", "content": kept}], m, len(kept)):
                renumbered[("messages", m, old)] = None
                last_dropped = old
                continue
            renumbered[("messages", m, old)] = ("messages", m, len(kept))
            kept.append(block)
        if len(kept) == len(message["content"]):
            continue
        if not kept:
            kept = [{"type": "text", "text": EMPTY_TEXT_PLACEHOLDER}]
            renumbered[("messages", m, last_dropped)] = ("messages", m, 0)
        message["content"] = kept
        moved |= renumbered
    if not moved:
        return paths
    return [tuple(moved.get(p, p) for p in own if moved.get(p, p) is not None) for own in paths]


@dataclass(frozen=True)
class BuildMeta:
    """`breakpoints`: one dict per placed breakpoint, {bp, position, block, prefix_tokens}, in
    block order (position is the window position, None for a tool definition). `paths`: for
    each window position, the top-level block paths it renders into, ("system", i) or
    ("messages", m, b)."""

    breakpoints: tuple[dict, ...]
    paths: tuple[tuple[tuple, ...], ...]


def build_request(
    state: WindowState,
    blobs: BlobStore,
    *,
    model: str,
    max_tokens: int,
    tools: Sequence[dict],
    ttl: str,
    pricing: PricingTable,
    frozen: str | None = None,
    prev_last: str | None = None,
    thinking: dict | None = None,
) -> tuple[dict, BuildMeta]:
    """Render the window as {"model", "max_tokens", "system", "tools", "messages"} (plus
    "thinking" if given) and BuildMeta.

    `frozen` is the segment id of the first fold-candidate position (BP2 goes on the last block
    before it); `prev_last` the last segment of the previous model call's window (BP3). Tools
    are sorted by name so their JSON, part of the cached prefix, is byte-stable. Empty text
    blocks are dropped, and so is a thinking block no longer bound to what precedes it (module
    docstring). Raises InvalidWindow if the result is not a valid request (see
    `request_problems`)."""
    system: list[dict] = []
    messages: list[dict] = []
    paths: list[tuple[tuple, ...]] = []
    for seg in state.segments:
        text = render_segment(state, seg, blobs)
        if seg.role == "system":
            system.append({"type": "text", "text": text})
            paths.append((("system", len(system) - 1),))
            continue
        role = "assistant" if seg.role == "assistant" else "user"
        if seg.tool_use_id is not None and role == "assistant":
            raise InvalidWindow(f"segment {seg.segment_id}: an assistant segment cannot be a tool result")
        if not messages or messages[-1]["role"] != role:
            messages.append({"role": role, "content": []})
        m, content = len(messages) - 1, messages[-1]["content"]
        if seg.tool_use_id is not None:
            last = content[-1] if content else None
            if last is None or last["type"] != "tool_result" or last["tool_use_id"] != seg.tool_use_id:
                last = {"type": "tool_result", "tool_use_id": seg.tool_use_id, "content": []}
                content.append(last)
            last["content"].append({"type": "text", "text": text})
            if seg.is_error:
                last["is_error"] = True
            paths.append((("messages", m, len(content) - 1),))
        elif seg.content_type == "blocks":
            start = len(content)
            content.extend(json.loads(text))
            paths.append(tuple(("messages", m, b) for b in range(start, len(content))))
        else:
            content.append({"type": "text", "text": text})
            paths.append((("messages", m, len(content) - 1),))
    paths = _drop_empty_text(messages, paths)
    if state.thinking_bound is not None:  # BUILD-V4FIX: only a ledger that recorded bindings (module docstring)
        paths = _drop_unbound_thinking(messages, paths, state.thinking_bound)
    ordered_tools = [json.loads(canonical_json(t)) for t in sorted(tools, key=lambda t: t["name"])]
    request = {"model": model, "max_tokens": max_tokens, "system": system, "tools": ordered_tools, "messages": messages}
    if thinking is not None:
        request["thinking"] = json.loads(canonical_json(thinking))
    problems = request_problems(request)  # no empty text block is left, so no breakpoint can go on one
    if problems:
        raise InvalidWindow("; ".join(problems))
    breakpoints = _place_breakpoints(state, request, paths, ttl, pricing.price(model).min_cache_tokens, frozen, prev_last)
    return request, BuildMeta(breakpoints, tuple(paths))


def _drop_empty_text(messages: list[dict], paths: list[tuple[tuple, ...]]) -> list[tuple[tuple, ...]]:
    """Remove every empty text block of the messages, in place; a message left with no block
    keeps EMPTY_TEXT_PLACEHOLDER in place of its last empty one. Returns the paths renumbered
    (a dropped block's path is gone, so its segment may own none)."""
    moved: dict[tuple, tuple | None] = {}
    for m, message in enumerate(messages):
        content = message["content"]
        empty = [b for b, block in enumerate(content) if empty_text_block(block)]
        if empty and len(empty) == len(content):
            content[empty.pop()] = {"type": "text", "text": EMPTY_TEXT_PLACEHOLDER}
        if not empty:
            continue
        kept = [b for b in range(len(content)) if b not in empty]
        moved |= {("messages", m, b): None for b in empty} | {("messages", m, old): ("messages", m, new) for new, old in enumerate(kept)}
        message["content"] = [content[b] for b in kept]
    if not moved:
        return paths
    return [tuple(moved.get(p, p) for p in own if moved.get(p, p) is not None) for own in paths]


def _place_breakpoints(state, request, paths, ttl, min_tokens, frozen, prev_last) -> tuple[dict, ...]:
    """Choose BP1..BP4 (deduplicated, at most four), mark their blocks, and describe them."""
    ids = [seg.segment_id for seg in state.segments]
    linear = linear_paths(request)
    index = {path: n for n, path in enumerate(linear)}
    owner = {path: p for p, own in enumerate(paths) for path in own}
    estimator = estimator_for(state)
    prefix, total = [], 0
    for path, (_, block) in zip(linear, linearize(request)):
        position = owner.get(path)
        cls = "json" if position is None else segment_class(state, state.segments[position])
        total += estimator.count(canonical_json(strip_cache_control(block)).decode("utf-8"), cls)
        prefix.append(total)

    def last_before(position: int) -> tuple | None:
        return next((paths[q][-1] for q in range(position - 1, -1, -1) if paths[q]), None)

    wanted: list[tuple[str, tuple | None]] = [
        ("BP1", ("system", len(request["system"]) - 1) if request["system"] else ("tools", len(request["tools"]) - 1) if request["tools"] else None),
        ("BP2", last_before(ids.index(frozen)) if frozen in ids else None),
        ("BP3", last_before(ids.index(prev_last) + 1) if prev_last in ids else None),
        ("BP4", linear[-1] if request["messages"] else None),
    ]
    chosen: dict[tuple, str] = {}
    for name, path in wanted:
        if path is not None and path not in chosen and prefix[index[path]] >= min_tokens:
            chosen[path] = name
    out = []
    for path in sorted(chosen, key=index.__getitem__):
        _block_at(request, path)["cache_control"] = cache_control(ttl)
        out.append({"bp": chosen[path], "position": owner.get(path), "block": index[path], "prefix_tokens": prefix[index[path]]})
    return tuple(out)


def _block_at(request: dict, path: tuple) -> dict:
    if path[0] == "messages":
        return request["messages"][path[1]]["content"][path[2]]
    return request[path[0]][path[1]]
