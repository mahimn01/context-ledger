"""Configurable model price table and exact call costs.

Prices are integer micro-USD per million tokens ($2.50/MTok is 2_500_000), so a token count
times a price is an exact integer number of pico-USD (1e-12 USD). Nothing here is a float.
The table is data, loaded by path; the ledger records it (`pricing_set`) so every cost and
payoff decision replays without the external file.

A model may have a long-context tier: when a request's prompt (input + cache read + cache
write tokens) exceeds its threshold, the whole request, output included, is priced at the
higher tier. `VERIFIED_PRICING_PATH` is the table checked against the published prices on
2026-10-31; it is the default for every live and pilot path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from .ids import canonical_json, sha256_hex

PICO_PER_USD = 10**12
VERIFIED_PRICING_PATH = Path(__file__).resolve().parents[2] / "pricing" / "anthropic-2026-10-31.json"
USAGE_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
BREAKDOWN_FIELDS = ("cache_creation_5m", "cache_creation_1h")


class PricingError(Exception):
    """A price table is malformed, or a cost cannot be priced (unknown model, tier exceeded)."""


def _int(value: object, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PricingError(f"{what} must be an int >= 0, got {value!r}")
    return value


_PRICE_KEYS = ("input", "output", "cache_write_5m", "cache_write_1h", "cache_read", "min_cache_tokens")
_RATE_KEYS = ("input", "output", "cache_write_5m", "cache_write_1h", "cache_read")


@dataclass(frozen=True)
class Rates:
    """The five per-token prices of one tier, in micro-USD per MTok."""

    input: int
    output: int
    cache_write_5m: int
    cache_write_1h: int
    cache_read: int

    def cache_write(self, ttl: str) -> int:
        if ttl not in ("5m", "1h"):
            raise PricingError(f"unknown cache ttl {ttl!r}")
        return self.cache_write_5m if ttl == "5m" else self.cache_write_1h


@dataclass(frozen=True)
class LongContextPrice:
    """The higher tier of a model: it prices a whole request whose prompt (input + cache read +
    cache write tokens) exceeds `threshold_prompt_tokens`."""

    threshold_prompt_tokens: int
    input: int
    output: int
    cache_write_5m: int
    cache_write_1h: int
    cache_read: int

    def __post_init__(self) -> None:
        for key in ("threshold_prompt_tokens", *_RATE_KEYS):
            _int(getattr(self, key), f"long_context.{key}")
        if self.threshold_prompt_tokens < 1:
            raise PricingError("long_context.threshold_prompt_tokens must be >= 1")

    def rates(self) -> Rates:
        return Rates(*(getattr(self, key) for key in _RATE_KEYS))

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in ("threshold_prompt_tokens", *_RATE_KEYS)}

    @classmethod
    def from_dict(cls, d: dict) -> LongContextPrice:
        expected = {"threshold_prompt_tokens", *_RATE_KEYS}
        if not isinstance(d, dict) or set(d) != expected:
            raise PricingError(f"long_context keys must be exactly {sorted(expected)}")
        return cls(**d)


@dataclass(frozen=True)
class ModelPrice:
    """Prices of one model in micro-USD per MTok. `min_cache_tokens` (shortest cacheable prefix)
    is used only to skip pointless breakpoints. `max_prompt_tokens_for_tier`, if set, is the
    largest prompt these prices cover. `long_context`, if set, is the tier for prompts above
    its threshold (it and its key are absent from the dict when None, so older tables hash
    as before)."""

    model: str
    input: int
    output: int
    cache_write_5m: int
    cache_write_1h: int
    cache_read: int
    min_cache_tokens: int
    max_prompt_tokens_for_tier: int | None = None
    long_context: LongContextPrice | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise PricingError("model must be a non-empty string")
        for key in _PRICE_KEYS:
            _int(getattr(self, key), f"{self.model}.{key}")
        if self.max_prompt_tokens_for_tier is not None:
            _int(self.max_prompt_tokens_for_tier, f"{self.model}.max_prompt_tokens_for_tier")
        if self.long_context is not None:
            if not isinstance(self.long_context, LongContextPrice):
                raise PricingError(f"{self.model}.long_context must be a LongContextPrice")
            limit = self.max_prompt_tokens_for_tier
            if limit is not None and limit <= self.long_context.threshold_prompt_tokens:
                raise PricingError(f"{self.model}: max_prompt_tokens_for_tier must lie above the long-context threshold")

    def rates_for(self, prompt_tokens: int) -> Rates:
        """The tier that prices a request with this many prompt tokens. Raises PricingError above
        `max_prompt_tokens_for_tier`; never guesses a tier the table does not list."""
        limit = self.max_prompt_tokens_for_tier
        if limit is not None and prompt_tokens > limit:
            raise PricingError(f"{self.model}: a {prompt_tokens}-token prompt is above the {limit}-token price tier")
        if self.long_context is not None and prompt_tokens > self.long_context.threshold_prompt_tokens:
            return self.long_context.rates()
        return Rates(*(getattr(self, key) for key in _RATE_KEYS))

    def cache_write(self, ttl: str) -> int:
        if ttl not in ("5m", "1h"):
            raise PricingError(f"unknown cache ttl {ttl!r}")
        return self.cache_write_5m if ttl == "5m" else self.cache_write_1h

    def to_dict(self) -> dict:
        d = {**{key: getattr(self, key) for key in _PRICE_KEYS}, "max_prompt_tokens_for_tier": self.max_prompt_tokens_for_tier}
        return {**d, "long_context": self.long_context.to_dict()} if self.long_context is not None else d

    @classmethod
    def from_dict(cls, model: str, d: dict) -> ModelPrice:
        expected = {*_PRICE_KEYS, "max_prompt_tokens_for_tier"}
        if not isinstance(d, dict) or not expected <= set(d) <= expected | {"long_context"}:
            raise PricingError(f"{model}: price keys must be exactly {sorted(expected)}, plus optionally long_context")
        long_context = LongContextPrice.from_dict(d["long_context"]) if "long_context" in d else None
        return cls(model, **{k: v for k, v in d.items() if k != "long_context"}, long_context=long_context)


@dataclass(frozen=True)
class PricingTable:
    """A versioned price table. `verified` says whether the numbers were checked against the
    provider's published prices; `source` says where they came from and `sources` lists the
    pages they were checked against (omitted from the dict when empty)."""

    version: str
    as_of: str
    source: str
    verified: bool
    models: dict[str, ModelPrice] = field(hash=False)
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for key in ("version", "as_of", "source"):
            if not isinstance(getattr(self, key), str):
                raise PricingError(f"{key} must be a string")
        if not isinstance(self.verified, bool):
            raise PricingError("verified must be a bool")
        if not isinstance(self.models, dict) or not self.models:
            raise PricingError("models must be a non-empty dict")
        if any(not isinstance(p, ModelPrice) or p.model != name for name, p in self.models.items()):
            raise PricingError("models must map each model id to its ModelPrice")
        object.__setattr__(self, "sources", tuple(self.sources))
        if not all(isinstance(url, str) and url for url in self.sources):
            raise PricingError("sources must be non-empty strings")

    def to_dict(self) -> dict:
        d = {
            "version": self.version,
            "as_of": self.as_of,
            "source": self.source,
            "verified": self.verified,
            "models": {name: self.models[name].to_dict() for name in sorted(self.models)},
        }
        return {**d, "sources": list(self.sources)} if self.sources else d

    @classmethod
    def from_dict(cls, d: dict) -> PricingTable:
        """Strict: exactly the keys version, as_of, source, verified, models (plus optionally
        sources, a list of strings); ints >= 0."""
        required = {"version", "as_of", "source", "verified", "models"}
        if not isinstance(d, dict) or not required <= set(d) <= required | {"sources"}:
            raise PricingError("a price table has exactly the keys as_of, models, source, verified, version (and optionally sources)")
        if not isinstance(d["models"], dict):
            raise PricingError("models must be a dict")
        if not isinstance(d.get("sources", []), list):
            raise PricingError("sources must be a list")
        models = {name: ModelPrice.from_dict(name, prices) for name, prices in d["models"].items()}
        return cls(d["version"], d["as_of"], d["source"], d["verified"], models, tuple(d.get("sources", [])))

    @classmethod
    def load(cls, path: str | Path, *, live: bool = False) -> PricingTable:
        """Read a JSON price table; floats anywhere in the file are rejected. With `live`, an
        unverified table is refused: no paid call is ever priced with guessed numbers."""

        def no_floats(text: str) -> object:
            raise PricingError(f"price tables hold integers only, found {text}")

        table = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8"), parse_float=no_floats))
        if live and not table.verified:
            raise PricingError(f"price table {table.version} is not verified; live and pilot paths need a verified table")
        return table

    @classmethod
    def verified_default(cls) -> PricingTable:
        """The verified table at VERIFIED_PRICING_PATH, loaded with the live check."""
        return cls.load(VERIFIED_PRICING_PATH, live=True)

    def pricing_hash(self) -> str:
        """sha256 of the canonical JSON of `to_dict()`."""
        return sha256_hex(canonical_json(self.to_dict()))

    def price(self, model: str) -> ModelPrice:
        try:
            return self.models[model]
        except KeyError:
            raise PricingError(f"model {model!r} is not in price table {self.version}") from None

    def cost_pusd(self, model: str, usage: dict, ttl: str = "5m") -> int:
        """Exact pico-USD cost of one call.

        `usage` has the int fields input_tokens, cache_creation_input_tokens,
        cache_read_input_tokens and output_tokens, and optionally the cache-write breakdown
        cache_creation_5m and cache_creation_1h. With the breakdown, the two TTLs are priced
        separately (any remainder of cache_creation_input_tokens at `ttl`); without it, all
        cache writes are priced at the TTL the request used. The whole request, output
        included, is priced at the tier its prompt (input + cache write + cache read) selects.
        Raises PricingError for an unknown model, a malformed usage, or a prompt above the
        model's tier limit."""
        counts = {key: _int(usage.get(key, 0), f"usage.{key}") for key in USAGE_FIELDS}
        prompt = counts["input_tokens"] + counts["cache_creation_input_tokens"] + counts["cache_read_input_tokens"]
        price = self.price(model).rates_for(prompt)
        writes = {"5m": 0, "1h": 0}
        if any(key in usage for key in BREAKDOWN_FIELDS):
            writes = {"5m": _int(usage.get("cache_creation_5m", 0), "usage.cache_creation_5m"),
                      "1h": _int(usage.get("cache_creation_1h", 0), "usage.cache_creation_1h")}  # fmt: skip
        rest = counts["cache_creation_input_tokens"] - writes["5m"] - writes["1h"]
        if rest < 0:
            raise PricingError("the cache-write breakdown exceeds cache_creation_input_tokens")
        price.cache_write(ttl)  # rejects an unknown ttl
        writes[ttl] += rest
        return (
            counts["input_tokens"] * price.input
            + writes["5m"] * price.cache_write_5m
            + writes["1h"] * price.cache_write_1h
            + counts["cache_read_input_tokens"] * price.cache_read
            + counts["output_tokens"] * price.output
        )

    def worst_case_pusd(self, model: str, prompt_tokens_upper: int, max_tokens: int) -> int:
        """An upper bound on the cost of a request whose prompt has at most `prompt_tokens_upper`
        tokens and whose output at most `max_tokens`: every prompt token at the dearest input
        price (uncached, 5m or 1h write) plus every output token, both at the tier
        `prompt_tokens_upper` selects. A smaller prompt never selects a dearer tier, so any
        usage within those bounds costs at most this."""
        prompt_tokens_upper = _int(prompt_tokens_upper, "prompt_tokens_upper")
        max_tokens = _int(max_tokens, "max_tokens")
        rates = self.price(model).rates_for(prompt_tokens_upper)
        return prompt_tokens_upper * max(rates.input, rates.cache_write_5m, rates.cache_write_1h) + max_tokens * rates.output


def usd(pusd: int | Fraction, places: int = 6) -> str:
    """Display only: pico-USD as "$0.000123" (rounded half-even to `places` decimals)."""
    value = Fraction(pusd) / PICO_PER_USD
    amount = (Decimal(value.numerator) / Decimal(value.denominator)).quantize(Decimal(1).scaleb(-places))
    return f"-${-amount}" if amount < 0 else f"${amount}"
