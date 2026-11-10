"""Predicted versus actual fold savings in the ledger arm (F.2).

For each fold in an L ledger, with the next model call after it:
- predicted delta: the fold's freed_tokens, the net savings the payoff gate used: the estimator
  in force (calib-v2 in pilot runs) on the rendered victims minus the rendered fold segment;
- predicted rewrite cost D: for a backstop-v1 fold, (L - P_i - delta) tokens re-written at
  p_w - p_r, from its decision record (zero when the cache was cold); model folds record none;
- actual delta: count_tokens of the counterfactual request (the next call's request rebuilt
  from the ledger with the fold left out) minus the next call's actual prompt tokens
  (input + cache read + cache write). If the ledger cannot be replayed without the fold
  because a later fold before the same call builds on it (BUILD-V2b: several folds per call),
  it is count_tokens of the request just before the fold minus just after it (the
  counterfactual prompt is then the one just before);
- actual D: the next call's cache-write tokens times the cache-write price.
count_tokens is free; offline it is the fake server's deterministic count.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..anthropic.request import InvalidWindow, build_request, strip_cache_control
from ..ids import canonical_json
from ..model import Event
from ..pricing import PricingTable
from ..window import ReplayError, replay

COUNT_KEYS = ("model", "system", "tools", "messages", "thinking")


def _unmarked(request: dict) -> dict:
    """The request without cache_control markers (they do not change a token count)."""
    out = {k: request[k] for k in COUNT_KEYS if k in request}
    out["system"] = [strip_cache_control(b) for b in out.get("system", [])]
    out["tools"] = [strip_cache_control(t) for t in out.get("tools", [])]
    out["messages"] = [{**m, "content": [strip_cache_control(b) for b in m["content"]]} for m in out.get("messages", [])]
    return out


def fold_reconciliation(events: Sequence[Event], blobs, count, episode: int) -> list[dict]:
    """One row per fold that a model call followed. `count(body) -> int | None` is count_tokens.
    The counterfactual is None if the ledger cannot be replayed without the fold, nor the
    requests just before and after it be built."""
    rows = []
    for i, event in enumerate(events):
        if event.type != "fold":
            continue
        j = next((k for k in range(i + 1, len(events)) if events[k].type == "model_call"), None)
        if j is None:
            continue
        call = events[j].payload
        call_index = next(k for k in range(j - 1, i, -1) if events[k].type == "call")
        build = call["build"]
        pricing = PricingTable.from_dict(replay(events[: i + 1]).pricing)
        tools = json.loads(blobs.get(build["tools_hash"]).decode("utf-8"))

        def counted_at(kept: Sequence[Event]) -> int | None:
            request, _ = build_request(
                replay(kept), blobs, model=call["model"], max_tokens=build["max_tokens"], tools=tools, ttl=build["ttl"], pricing=pricing,
                thinking=build.get("thinking"),
            )  # fmt: skip
            return count(_unmarked(request))

        usage = call["usage"]
        actual_prompt = usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"]
        try:
            counted, after = counted_at([e for k, e in enumerate(events[: call_index + 1]) if k != i]), actual_prompt
        except (ReplayError, InvalidWindow):
            try:
                counted, after = counted_at(events[:i]), counted_at(events[: i + 1])
            except (ReplayError, InvalidWindow):
                counted = after = None
        fold = event.payload["fold"]
        policy = fold.get("policy") or {}
        price = pricing.price(call["model"])
        p_w, p_r = price.cache_write(build["ttl"]), price.cache_read
        predicted_d = None
        if policy.get("id") == "backstop-v1":
            rewrite = 0 if policy["cache_cold"] else policy["L"] - policy["P_i"] - policy["delta"]
            predicted_d = rewrite * (p_w - p_r)
        rows.append({
            "episode": episode,
            "fold_id": fold["fold_id"],
            "trigger": fold["trigger"],
            "zone": policy.get("zone"),
            "predicted_delta": fold["freed_tokens"],
            "predicted_rewrite_pusd": predicted_d,
            "counterfactual_prompt": counted,
            "actual_prompt": actual_prompt,
            "actual_delta": None if counted is None or after is None else counted - after,
            "next_call_cache_write": usage["cache_creation_input_tokens"],
            "actual_rewrite_pusd": usage["cache_creation_5m"] * price.cache_write_5m + usage["cache_creation_1h"] * price.cache_write_1h,
            "next_call_id": call["call_id"],
        })  # fmt: skip
        canonical_json(rows[-1])
    return rows
