from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from context_ledger.ids import canonical_json, new_id, sha256_hex, span_id, text_hash

json_scalars = st.none() | st.booleans() | st.integers() | st.text()
json_values = st.recursive(
    json_scalars,
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=4),
    max_leaves=20,
)


def test_key_order_independence():
    a = {"b": 1, "a": [1, 2, {"z": None, "y": True}]}
    b = {"a": [1, 2, {"y": True, "z": None}], "b": 1}
    assert canonical_json(a) == canonical_json(b) == b'{"a":[1,2,{"y":true,"z":null}],"b":1}'


@given(st.dictionaries(st.text(max_size=8), json_scalars, max_size=8))
def test_key_order_independence_property(d):
    reversed_d = dict(reversed(list(d.items())))
    assert canonical_json(d) == canonical_json(reversed_d)


@pytest.mark.parametrize("bad", [1.5, {"a": 0.0}, [1, [2, 3.0]], {"x": {"y": float("nan")}}, b"bytes", {1: "int key"}, {"s": {1, 2}}])
def test_rejects_floats_and_other_types(bad):
    with pytest.raises(TypeError):
        canonical_json(bad)


def test_tuple_serializes_like_list():
    assert canonical_json((1, "a")) == canonical_json([1, "a"])


@given(json_values)
def test_unicode_round_trip(value):
    assert json.loads(canonical_json(value).decode("utf-8")) == value


def test_unicode_is_not_escaped():
    data = canonical_json({"k": "héllo 中文 ✓"})
    assert "héllo 中文 ✓".encode("utf-8") in data
    assert json.loads(data) == {"k": "héllo 中文 ✓"}


def test_hashes():
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert text_hash("é") == sha256_hex("é".encode("utf-8"))
    h = text_hash("abc")
    assert len(h) == 64 and h == h.lower()


def test_span_id_depends_on_source_and_anchor_only():
    anchor = {"kind": "text", "start": 0, "end": 3, "exact": "abc", "prefix": "", "suffix": ""}
    assert span_id("h1", anchor) == span_id("h1", dict(reversed(list(anchor.items()))))
    assert span_id("h1", anchor) != span_id("h2", anchor)
    assert span_id("h1", anchor) == sha256_hex(canonical_json({"source": "h1", "anchor": anchor}))


def test_new_id_is_deterministic():
    assert new_id("seg", "a", "b") == new_id("seg", "a", "b")
    assert new_id("seg", "a", "b") != new_id("seg", "b", "a")
    assert new_id("seg", "ab") != new_id("seg", "a", "b")
    ident = new_id("call", "x")
    assert ident.startswith("call_") and len(ident) == len("call_") + 16
    assert ident == "call_" + sha256_hex(canonical_json(["x"]))[:16]
