"""GuardedClient: the only client an arm, the compaction summarizer, the calibrator or the
planner ever receives. It exposes `messages.create(**body)` and `messages.count_tokens(**body)`
and nothing else; every create goes through the SpendGuard.

Before each create: count the prompt with the free count_tokens endpoint on the identical body
(model, system, tools, messages and thinking, without max_tokens), bound it
(ceil(count x 1.05) + 64, or the conservative fallback if counting fails), and reserve the
worst case. The guard refuses (BudgetRefused) before any network call if that does not fit.

After it: success settles the API-reported cost (GuardInvariantBroken halts the run if it is
above the reservation); an HTTP error with a status of 400-599 and a response body is void
(failed requests are not billed); anything else (a timeout, a dropped connection) is assumed
spent at the worst case. 429, 5xx and ambiguous failures are retried, each retry a new
reservation, at most twice, with backoff.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from ..anthropic.usage import normalize_usage, response_dict
from .redact import api_error_record
from .spend import SpendGuard, prompt_upper_fallback, prompt_upper_from_count, request_sha256

log = logging.getLogger("context_ledger.live")
COUNT_KEYS = ("model", "system", "tools", "messages", "thinking")
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})


def ttl_of(body: dict) -> str:
    """"1h" if any block of the request asks for a one-hour cache write, else "5m"."""

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            marker = value.get("cache_control")
            if isinstance(marker, dict) and marker.get("ttl") == "1h":
                return True
            return any(walk(v) for v in value.values())
        return isinstance(value, list) and any(walk(v) for v in value)

    return "1h" if walk(body) else "5m"


def http_status(exc: BaseException) -> int | None:
    """The HTTP status of an API error that came with a response body, else None."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status <= 599 and getattr(exc, "body", None) is not None:
        return status
    return None


class _Messages:
    def __init__(self, owner: GuardedClient) -> None:
        self._owner = owner

    def create(self, **body: Any) -> Any:
        return self._owner._create(body)

    def count_tokens(self, **body: Any) -> Any:
        return self._owner._raw.messages.count_tokens(**body)


class GuardedClient:
    """Wraps a raw SDK-like client (the real one or the fake). `estimator` is the calib-v1
    estimator for the counting fallback (None: bytes / 2). `last_prompt_count` is the count of
    the latest successful create (None if counting failed), for the predicted-vs-actual record."""

    def __init__(
        self,
        raw: Any,
        guard: SpendGuard,
        *,
        estimator: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff: Sequence[float] = (2.0, 8.0),
        max_retries: int = 2,
    ) -> None:
        self._raw, self.guard, self.estimator = raw, guard, estimator
        self._sleep, self._backoff, self._max_retries = sleep, tuple(backoff), max_retries
        self.messages = _Messages(self)
        self.last_prompt_count: int | None = None
        self.errors: list[dict] = []  # redacted API error records

    def _count(self, body: dict) -> int | None:
        try:
            result = self._raw.messages.count_tokens(**{k: body[k] for k in COUNT_KEYS if k in body})
            value = result["input_tokens"] if isinstance(result, dict) else result.input_tokens
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        except Exception as exc:  # counting is best effort; the fallback bound is conservative
            log.warning("count_tokens failed, using the fallback bound: %s", api_error_record(exc))
            return None

    def _create(self, body: dict) -> Any:
        model, max_tokens, ttl, digest = body["model"], body["max_tokens"], ttl_of(body), request_sha256(body)
        for attempt in range(self._max_retries + 1):
            count = self._count(body)
            upper = prompt_upper_from_count(count) if count is not None else prompt_upper_fallback(body, self.estimator)
            reservation = self.guard.reserve(model=model, request_sha256=digest, prompt_tokens_upper=upper, max_tokens=max_tokens)
            try:
                response = self._raw.messages.create(**body)
            except Exception as exc:
                status = http_status(exc)
                record = api_error_record(exc)
                self.errors.append(record)
                if status is not None:
                    self.guard.void(reservation, f"http_{status}")
                else:
                    self.guard.assume_spent(reservation, f"ambiguous_{record['error_type']}")
                log.warning("API call failed (attempt %d): %s", attempt + 1, record)
                if attempt < self._max_retries and (status is None or status in RETRY_STATUSES):
                    self._sleep(self._backoff[min(attempt, len(self._backoff) - 1)])
                    continue
                raise
            usage = normalize_usage(response_dict(response).get("usage"), ttl)
            self.guard.settle(reservation, usage, ttl)
            self.last_prompt_count = count
            return response
        raise AssertionError("unreachable")  # the loop always returns or raises
