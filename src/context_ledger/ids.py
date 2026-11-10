"""Canonical JSON and deterministic content-derived identifiers.

Nothing here uses randomness or the clock, so a ledger built from the same inputs is
byte-for-byte reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from typing import Any

GENESIS_HASH = "0" * 64


def _check_json_value(obj: Any, path: str) -> None:
    """Raise TypeError unless `obj` is built only from dict (str keys), list, tuple, str,
    int, bool and None. Floats are rejected because their text form is not canonical."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _check_json_value(item, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: dict key {key!r} is not a str")
            _check_json_value(value, f"{path}.{key}")
        return
    raise TypeError(f"{path}: {type(obj).__name__} is not allowed in canonical JSON")


def canonical_json(obj: Any) -> bytes:
    """Serialize `obj` to canonical UTF-8 JSON bytes (sorted keys, no whitespace).

    Raises TypeError, before serializing anything, if any value anywhere in `obj` is not a
    dict with str keys, list, tuple, str, int, bool or None (floats included)."""
    _check_json_value(obj, "$")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the sha256 of `data` as 64 lowercase hex characters."""
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    """Return sha256_hex of the UTF-8 encoding of `text`."""
    return sha256_hex(text.encode("utf-8"))


def span_id(source_hash: str, anchor: dict) -> str:
    """Return H(canonical_json({"source": source_hash, "anchor": anchor})).

    The id depends only on the source content and the anchor, never on how the source was
    chunked or when the span was registered."""
    return sha256_hex(canonical_json({"source": source_hash, "anchor": anchor}))


def new_id(prefix: str, *parts: str) -> str:
    """Return the deterministic id `prefix + "_" + sha256_hex(canonical_json(list(parts)))[:16]`."""
    return f"{prefix}_{sha256_hex(canonical_json(list(parts)))[:16]}"


def frac_str(value: Fraction) -> str:
    """Canonical string of a rational: "num/den", reduced, den > 0 ("7/1" for integers)."""
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def parse_frac(text: str) -> Fraction:
    """Inverse of frac_str; ValueError unless `text` is exactly a reduced "num/den" with den > 0."""
    if not isinstance(text, str) or _FRAC.match(text) is None:
        raise ValueError(f"not a canonical fraction: {text!r}")
    value = Fraction(*map(int, text.split("/")))
    if frac_str(value) != text:
        raise ValueError(f"fraction {text!r} is not reduced")
    return value


_FRAC = re.compile(r"^-?(0|[1-9][0-9]*)/[1-9][0-9]*$")
