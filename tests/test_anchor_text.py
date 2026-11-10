from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from context_ledger.anchor.text import (
    chunk_views,
    make_text_anchor,
    normalize_for_match,
    normalize_newlines,
    resolve_text,
)
from context_ledger.ids import span_id

ALPHABET = st.characters(blacklist_categories=("Cs",), blacklist_characters="\r§")
texts = st.text(alphabet=ALPHABET, max_size=300)
MARK = "§"  # appears exactly once in marked texts, so `exact` has exactly one occurrence


@st.composite
def text_and_range(draw):
    text = draw(texts.filter(bool))
    start = draw(st.integers(0, len(text) - 1))
    end = draw(st.integers(start + 1, len(text)))
    return text, start, end


@st.composite
def marked_text(draw):
    """(text, start, end) where text[start:end] contains the only MARK in the text."""
    left, right = draw(texts), draw(texts)
    a, b = draw(st.text(alphabet=ALPHABET, max_size=10)), draw(st.text(alphabet=ALPHABET, max_size=10))
    text = left + a + MARK + b + right
    return text, len(left), len(left) + len(a) + 1 + len(b)


def test_normalize_newlines():
    assert normalize_newlines("a\r\nb\rc\n") == "a\nb\nc\n"


def test_normalize_for_match():
    assert normalize_for_match("  ＡＢＣ\t１２３ \n\n x  ") == "ABC 123 x"


@given(text_and_range())
def test_fresh_anchor_resolves_ok(case):
    text, start, end = case
    anchor = make_text_anchor(text, start, end)
    assert anchor.exact == text[start:end]
    assert len(anchor.prefix) <= 32 and len(anchor.suffix) <= 32
    assert resolve_text(anchor, text).status == "OK"


@given(marked_text(), st.text(alphabet=ALPHABET, min_size=1, max_size=50), st.data())
def test_insertion_before_range_moves(case, insertion, data):
    text, start, end = case
    anchor = make_text_anchor(text, start, end)
    pos = data.draw(st.integers(0, start))
    new_text = text[:pos] + insertion + text[pos:]
    res = resolve_text(anchor, new_text)
    assert res.status == "MOVED"
    assert res.anchor.exact == anchor.exact
    assert (res.anchor.start, res.anchor.end) == (start + len(insertion), end + len(insertion))
    assert new_text[res.anchor.start : res.anchor.end] == anchor.exact


@given(marked_text(), st.text(alphabet=ALPHABET, min_size=1, max_size=50), st.data())
def test_insertion_after_range_stays_ok(case, insertion, data):
    text, start, end = case
    anchor = make_text_anchor(text, start, end)
    pos = data.draw(st.integers(end, len(text)))
    res = resolve_text(anchor, text[:pos] + insertion + text[pos:])
    assert res.status == "OK" and res.anchor.exact == anchor.exact


@given(marked_text())
def test_deleting_exact_text_is_stale(case):
    text, start, end = case
    anchor = make_text_anchor(text, start, end)
    assert resolve_text(anchor, text[:start] + text[end:]).status == "STALE"


def test_ambiguous_without_unique_context_is_stale():
    anchor = make_text_anchor("x foo y", 2, 5)
    assert resolve_text(anchor, "foo foo").status == "STALE"
    assert resolve_text(anchor, "foo foo").note == "ambiguous"


def test_duplicate_resolved_by_context():
    anchor = make_text_anchor("x foo y", 2, 5)
    res = resolve_text(anchor, "foo... x foo y")
    assert res.status == "MOVED" and (res.anchor.start, res.anchor.end) == (9, 12)


def test_make_text_anchor_rejects_empty_and_out_of_range():
    for start, end in [(2, 2), (3, 1), (-1, 2), (0, 99)]:
        with pytest.raises(ValueError):
            make_text_anchor("hello", start, end)


@given(texts.filter(bool), st.integers(1, 64), st.data())
def test_chunking_does_not_change_span_ids(text, size, data):
    overlap = data.draw(st.integers(0, size - 1))
    views = chunk_views(text, size, overlap)
    assert views[0][0] == 0 and views[-1][1] == len(text)
    for (s1, e1), (s2, e2) in zip(views, views[1:]):
        assert e1 - s2 == overlap and e1 - s1 == size
    source_hash = "a" * 64
    for chunk_start, chunk_end in views:
        chunk = text[chunk_start:chunk_end]
        rel_start = data.draw(st.integers(0, len(chunk) - 1))
        rel_end = data.draw(st.integers(rel_start + 1, len(chunk)))
        start, end = chunk_start + rel_start, chunk_start + rel_end
        direct = make_text_anchor(text, start, end)
        assert direct.exact == chunk[rel_start:rel_end]
        # Every chunk that contains the range (overlaps included) yields the same span id.
        containing = [(cs, ce) for cs, ce in views if cs <= start and end <= ce]
        ids = {
            span_id(source_hash, make_text_anchor(text, cs + (start - cs), cs + (end - cs)).to_dict())
            for cs, ce in containing
        }
        assert ids == {span_id(source_hash, direct.to_dict())}


def test_chunk_views_validates_arguments():
    assert chunk_views("", 4, 1) == []
    assert chunk_views("abcdefg", 3, 1) == [(0, 3), (2, 5), (4, 7)]
    for size, overlap in [(0, 0), (3, 3), (3, -1)]:
        with pytest.raises(ValueError):
            chunk_views("abc", size, overlap)
