from __future__ import annotations

import pytest

from context_ledger.anchor.python import python_anchors, resolve_python

SRC = '''\
import functools

MAX_RETRIES = 3
NAME: str = "svc"
A = B = 7
x, y = 1, 2


def retry(fn):
    @functools.wraps(fn)
    def wrapper(*args):
        return fn(*args)
    return wrapper


@retry
def fetch(url, timeout=30):
    return url


async def poll(n):
    return n


class Outer:
    class Inner:
        def deep(self):
            return 1

    def method(self, value):
        def helper(y):
            return y
        return helper(value)
'''


def by_name(text: str) -> dict:
    return {a.qualname: a for a in python_anchors(text, "pkg/mod.py", commit="abc123")}


def slice_of(text: str, anchor) -> str:
    return text[anchor.start : anchor.end]


def test_qualnames_and_kinds():
    anchors = by_name(SRC)
    assert {q: a.node_kind for q, a in anchors.items()} == {
        "MAX_RETRIES": "assign",
        "NAME": "assign",
        "A": "assign",
        "B": "assign",
        "retry": "function",
        "retry.<locals>.wrapper": "function",
        "fetch": "function",
        "poll": "function",
        "Outer": "class",
        "Outer.Inner": "class",
        "Outer.Inner.deep": "function",
        "Outer.method": "function",
        "Outer.method.<locals>.helper": "function",
    }
    assert all(a.path == "pkg/mod.py" and a.commit == "abc123" for a in anchors.values())


def test_ranges_cover_decorators_and_bodies():
    anchors = by_name(SRC)
    assert slice_of(SRC, anchors["fetch"]) == "@retry\ndef fetch(url, timeout=30):\n    return url"
    assert slice_of(SRC, anchors["retry.<locals>.wrapper"]).startswith("@functools.wraps(fn)\n    def wrapper")
    assert slice_of(SRC, anchors["poll"]) == "async def poll(n):\n    return n"
    assert slice_of(SRC, anchors["MAX_RETRIES"]) == "MAX_RETRIES = 3"
    assert slice_of(SRC, anchors["NAME"]) == 'NAME: str = "svc"'
    assert slice_of(SRC, anchors["A"]) == slice_of(SRC, anchors["B"]) == "A = B = 7"
    assert slice_of(SRC, anchors["Outer"]).endswith("return helper(value)")


def test_non_ascii_before_definition_on_same_line():
    text = 's = "héllo 日本"; LIMIT = 5\nT = "日本語"\n'
    anchors = by_name(text)
    assert slice_of(text, anchors["s"]) == 's = "héllo 日本"'
    assert slice_of(text, anchors["LIMIT"]) == "LIMIT = 5"
    assert slice_of(text, anchors["T"]) == 'T = "日本語"'


def test_comments_and_blank_lines_keep_hash():
    old = by_name(SRC)["fetch"]
    edited = SRC.replace("@retry\ndef fetch", "# a comment\n\n\n@retry\ndef fetch").replace(
        "    return url", "    # inner comment\n\n    return url  # trailing"
    )
    new = by_name(edited)["fetch"]
    assert new.ast_hash == old.ast_hash
    res = resolve_python(old, edited)
    assert res.status in {"OK", "MOVED"}
    assert slice_of(edited, res.anchor).endswith("return url")


def test_reformat_keeps_hash():
    old = by_name(SRC)["Outer.method"]
    edited = SRC.replace("def method(self, value):", "def method(\n        self,\n        value,\n    ):")
    res = resolve_python(old, edited)
    assert res.status in {"OK", "MOVED"} and res.anchor.ast_hash == old.ast_hash


def test_unchanged_resolves_ok():
    old = by_name(SRC)["poll"]
    edited = SRC + "\n\ndef added():\n    pass\n"
    assert resolve_python(old, edited).status == "OK"


@pytest.mark.parametrize(
    ("qualname", "before", "after"),
    [("fetch", "timeout=30", "timeout=60"), ("MAX_RETRIES", "MAX_RETRIES = 3", "MAX_RETRIES = 4")],
)
def test_changed_value_is_stale(qualname, before, after):
    old = by_name(SRC)[qualname]
    assert resolve_python(old, SRC.replace(before, after)).status == "STALE"


def test_inserting_function_above_moves():
    old = by_name(SRC)["fetch"]
    edited = SRC.replace("@retry\ndef fetch", "def inserted():\n    return 0\n\n\n@retry\ndef fetch")
    res = resolve_python(old, edited)
    assert res.status == "MOVED"
    assert res.anchor.start == old.start + len("def inserted():\n    return 0\n\n\n")
    assert slice_of(edited, res.anchor) == slice_of(SRC, old)


def test_rename_with_identical_body_moves_renamed():
    old = by_name(SRC)["fetch"]
    edited = SRC.replace("def fetch(", "def download(")
    res = resolve_python(old, edited)
    assert (res.status, res.note, res.anchor.qualname) == ("MOVED", "renamed", "download")


def test_deleted_function_dangles():
    old = by_name(SRC)["poll"]
    edited = SRC.replace("async def poll(n):\n    return n\n", "")
    assert resolve_python(old, edited).status == "DANGLING"


def test_syntax_error_falls_back_to_text():
    old = by_name(SRC)["fetch"]
    old_text = slice_of(SRC, old)
    broken_after = SRC + "\ndef broken(:\n"
    res = resolve_python(old, broken_after, old_text=old_text)
    assert res.status == "OK" and res.note.startswith("fallback")
    broken_before = "x = (\n" + SRC
    res = resolve_python(old, broken_before, old_text=old_text)
    assert res.status == "MOVED" and res.note == "fallback"
    assert slice_of(broken_before, res.anchor) == old_text
    assert resolve_python(old, broken_after) == type(res)("DANGLING", old, "syntax_error")


def test_unparseable_source_raises_in_python_anchors():
    with pytest.raises(SyntaxError):
        python_anchors("def (:", "bad.py")
