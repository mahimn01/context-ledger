"""Normalizing Messages API responses and their usage for the ledger (no SDK import)."""

from __future__ import annotations

from typing import Any

USAGE_KEYS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
    "cache_creation_5m",
    "cache_creation_1h",
)
HANDLED_STOP_REASONS = frozenset({"end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal", "pause_turn"})


class UnsupportedResponse(Exception):
    """The response holds something the ledger cannot record (e.g. a server tool block)."""


def _count(raw: dict, key: str) -> int:
    value = raw.get(key)
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise UnsupportedResponse(f"usage.{key} must be an int >= 0, got {value!r}")
    return value


def normalize_usage(raw: dict | None, ttl: str) -> dict:
    """The six int usage fields the ledger records; missing or None values count as 0.

    With the API's `cache_creation` breakdown, the 5m and 1h writes are taken from it;
    without it, every cache write is attributed to the TTL the request used."""
    raw = raw or {}
    created = _count(raw, "cache_creation_input_tokens")
    breakdown = raw.get("cache_creation")
    if isinstance(breakdown, dict):
        five, hour = _count(breakdown, "ephemeral_5m_input_tokens"), _count(breakdown, "ephemeral_1h_input_tokens")
    else:
        five, hour = (created, 0) if ttl == "5m" else (0, created)
    return {
        "input_tokens": _count(raw, "input_tokens"),
        "cache_creation_input_tokens": created,
        "cache_read_input_tokens": _count(raw, "cache_read_input_tokens"),
        "output_tokens": _count(raw, "output_tokens"),
        "cache_creation_5m": five,
        "cache_creation_1h": hour,
    }


def plain(value: Any) -> Any:
    """Canonical-JSON-safe copy: dicts, lists, str, int, bool, None. An integral float becomes
    an int and any other float its repr string, because the ledger holds no floats."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float):
        return int(value) if value.is_integer() else repr(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise UnsupportedResponse(f"cannot record a {type(value).__name__} value")


def response_dict(response: Any) -> dict:
    """A Messages response as a plain dict, from an SDK object (to_dict / model_dump) or a dict."""
    if isinstance(response, dict):
        return response
    for method in ("to_dict", "model_dump"):
        if hasattr(response, method):
            return getattr(response, method)()
    raise UnsupportedResponse(f"cannot read a response of type {type(response).__name__}")


def content_blocks(content: list) -> list[dict]:
    """The response's text, tool_use, thinking and redacted_thinking blocks with exactly the
    fields the API needs back. Thinking blocks are opaque: kept verbatim (signature included)
    inside the assistant message, sent back unchanged, never folded apart from their turn and
    never summarized. Other block types (server tools) are unsupported."""
    blocks = []
    for block in content or []:
        block = block if isinstance(block, dict) else response_dict(block)
        kind = block.get("type")
        if kind == "thinking":
            blocks.append({"type": "thinking", "thinking": str(block["thinking"]), "signature": str(block["signature"])})
        elif kind == "redacted_thinking":
            blocks.append({"type": "redacted_thinking", "data": str(block["data"])})
        elif kind == "text":
            blocks.append({"type": "text", "text": str(block["text"])})
        elif kind == "tool_use":
            blocks.append({"type": "tool_use", "id": str(block["id"]), "name": str(block["name"]), "input": plain(block.get("input") or {})})
        else:
            raise UnsupportedResponse(f"content block type {kind!r} is not supported")
    return blocks


def handling(stop_reason: str | None) -> str:
    """How the loop treats a stop reason: the six handled values as themselves;
    model_context_window_exceeded (listed by the installed SDK) like max_tokens; anything
    else like end_turn. The raw value is what the ledger records."""
    if stop_reason in HANDLED_STOP_REASONS:
        return stop_reason
    return "max_tokens" if stop_reason == "model_context_window_exceeded" else "end_turn"
