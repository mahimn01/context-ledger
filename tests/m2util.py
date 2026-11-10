"""Shared helpers for the M2 tests: tamper-and-rechain, violation sets, sample sources."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from context_ledger.fold import BackstopPolicyV1
from context_ledger.invariants import check
from context_ledger.pricing import PricingTable
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore, rechain

MODULE_PATH = "inventory/client.py"
MODULE = '''"""HTTP client settings and retry helpers for the inventory service."""

import functools
import time

DEFAULT_TIMEOUT = 37.0
MAX_RETRIES = 4
USER_AGENT = "inventory-client/2.1"


def retrying(fn):
    """Retry `fn` up to MAX_RETRIES times on ConnectionError."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except ConnectionError:
                time.sleep(backoff_delay(attempt))
        return fn(*args, **kwargs)

    return wrapper


def backoff_delay(attempt):
    """Seconds to wait before retry number `attempt`."""
    return min(2 ** attempt, 300)


class InventoryClient:
    """Thin client for the inventory HTTP API."""

    def __init__(self, base_url, timeout=DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def url_for(self, sku):
        return f"{self.base_url}/items/{sku}"

    def restock_threshold(self, sku):
        """Reorder when stock falls below this many units."""
        return 12 if sku.startswith("A") else 25
'''


def props(events, blobs) -> set[str]:
    return {v.prop for v in check(events, blobs)}


def index_of(events, type_: str, pred=lambda p: True, last: bool = False) -> int:
    hits = [i for i, e in enumerate(events) if e.type == type_ and pred(e.payload)]
    return hits[-1] if last else hits[0]


def mutate(events, index, fn, rechained: bool = True):
    """Copy `events`, apply `fn` to a deep copy of one payload, and re-chain unless told not to."""
    payload = json.loads(json.dumps(events[index].payload))
    fn(payload)
    out = list(events)
    out[index] = replace(out[index], payload=payload)
    return rechain(out) if rechained else out


PRICING_PATH = Path(__file__).resolve().parents[1] / "pricing" / "anthropic-2026-10.json"


def pricing() -> PricingTable:
    return PricingTable.load(PRICING_PATH)


def v1_session(budget: int = 100_000, ledger: Ledger | None = None, blobs=None, **policy) -> Session:
    """A backstop-v1 (aliased) session over an in-memory ledger with the shipped price table."""
    return Session(
        ledger if ledger is not None else Ledger(":memory:"),
        blobs if blobs is not None else MemoryBlobStore(),
        budget,
        BackstopPolicyV1(budget_tokens=budget, **policy),
        pricing=pricing(),
    )


def reply(*content: dict, stop_reason: str | None = None, **extra):
    """A scripted Messages response (a callable of the request), stop reason inferred from content."""
    stop = stop_reason or ("tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn")
    return lambda request: {"content": list(content), "stop_reason": stop, **extra}


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def tool_use(id_: str, name: str, **input) -> dict:
    return {"type": "tool_use", "id": id_, "name": name, "input": input}


def sdk_client(server, asynchronous: bool = False):
    """The real Anthropic SDK client, wired to the fake server through httpx2.MockTransport."""
    import anthropic

    if asynchronous:
        return anthropic.AsyncAnthropic(
            api_key="test-key-not-real", base_url="http://anthropic.test", max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(transport=server.as_async_transport()),
        )  # fmt: skip
    return anthropic.Anthropic(
        api_key="test-key-not-real", base_url="http://anthropic.test", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=server.as_sync_transport()),
    )  # fmt: skip
