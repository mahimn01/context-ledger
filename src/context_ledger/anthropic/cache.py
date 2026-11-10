"""Predicted prompt-cache reads, the same way for every arm (M4).

The prediction for a request is the longest cached prefix it can read: a prefix ending at a
block where an earlier request of the same conversation (arm and episode) placed a breakpoint
that was written or read, byte-identical (the same sha256 over the marker-stripped blocks in
cache order) and within the TTL, at most `lookback` (20) blocks before one of this request's
breakpoints, as the API's cache lookup does. Prefixes below the model's minimum cacheable
length are never cached. The value is the estimated tokens of that prefix and never more than
the prompt's input total (the counted prompt when known). A request whose prefix changed (B3's
per-question context) reads only what is still identical: its system prompt if that alone is
long enough, else nothing.

After predicting, the request's own breakpoint prefixes (written or read) are recorded with
the current time. Block tokens are the estimator applied to each block's canonical JSON: tool
definitions, tool_use and tool_result blocks as "json", everything else as "other".
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..ids import canonical_json, sha256_hex
from .request import linearize, strip_cache_control

TTL_SECONDS = {"5m": 300, "1h": 3600}
LOOKBACK = 20


def block_tokens(request: dict, estimator) -> list[int]:
    """Estimated tokens of each block of `linearize(request)`, markers stripped."""
    out = []
    for where, block in linearize(request):
        cls = "json" if where == "tools" or block.get("type") in ("tool_use", "tool_result") else "other"
        out.append(estimator.count(canonical_json(strip_cache_control(block)).decode("utf-8"), cls))
    return out


def prefix_digests(request: dict) -> list[str]:
    """sha256 of the marker-stripped prefix through each block, in cache order."""
    out, digest = [], ""
    for where, block in linearize(request):
        digest = sha256_hex((digest + where).encode() + canonical_json(strip_cache_control(block)))
        out.append(digest)
    return out


class CachePredictor:
    """Predicts each request's cache read from the requests before it (see the module docstring).
    `clock` returns seconds; tests pass a fake one."""

    def __init__(self, estimator, min_tokens: int, *, clock: Callable[[], float] = time.monotonic, lookback: int = LOOKBACK) -> None:
        self.estimator, self.min_tokens, self.clock, self.lookback = estimator, min_tokens, clock, lookback
        self.cached: dict[str, tuple[float, int]] = {}  # prefix digest -> (last written or read, ttl seconds)

    def predict(self, request: dict, prompt_total: int | None = None) -> int:
        """Predicted cache_read_input_tokens of `request`; records its breakpoints afterwards."""
        now = self.clock()
        blocks = linearize(request)
        totals, running = [], 0
        for n in block_tokens(request, self.estimator):
            running += n
            totals.append(running)
        digests = prefix_digests(request)
        marks = [n for n, (_, block) in enumerate(blocks) if "cache_control" in block]
        live = {d for d, (at, ttl) in self.cached.items() if now - at <= ttl}
        readable = [
            q for q, d in enumerate(digests)
            if d in live and totals[q] >= self.min_tokens and any(0 <= b - q <= self.lookback for b in marks)
        ]  # fmt: skip
        best = max(readable, key=totals.__getitem__, default=None)
        if best is not None:  # a read refreshes the entry's TTL
            self.cached[digests[best]] = (now, self.cached[digests[best]][1])
        for b in marks:
            if totals[b] >= self.min_tokens:
                self.cached[digests[b]] = (now, TTL_SECONDS[blocks[b][1]["cache_control"].get("ttl", "5m")])
        cap = prompt_total if prompt_total is not None else (totals[-1] if totals else 0)
        return min(0 if best is None else totals[best], cap)
