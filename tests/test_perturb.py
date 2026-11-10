"""C.2: deterministic constant and identifier perturbation."""

from __future__ import annotations

import ast
import random
import re

import pytest
from hypothesis import given, settings, strategies as st

from context_ledger.eval import perturb as perturb_mod
from context_ledger.eval import pyscan
from context_ledger.eval.perturb import apply_edits, choose_renames, literal_edits, new_float, new_int, new_str, perturb_corpus
from pilot_fixture import fixture_files


def is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


@settings(max_examples=300, deadline=None)
@given(value=st.integers(-10**9, 10**9), seed=st.integers(0, 10**6))
def test_new_int_keeps_sign_and_digits(value, seed):
    new = new_int(str(value), random.Random(seed))
    if new is None:  # only when no other value of that width qualifies
        assert abs(value) < 10
        return
    v = int(new)
    assert v != value and v not in (0, 1) and len(str(abs(v))) == len(str(abs(value))) and (v < 0) == (value < 0)
    assert is_pow2(abs(value)) or not is_pow2(abs(v)) or value < 0 and is_pow2(-value)


def test_new_int_rules_by_hand():
    rng = random.Random(1)
    assert all(int(new_int("7", rng)) in {3, 5, 6, 9} for _ in range(50))  # no 0, 1, 7 or powers of two
    assert all(int(new_int("8", rng)) in {2, 3, 4, 5, 6, 7, 9} for _ in range(50))  # 8 is a power of two
    assert all(len(new_int("-37", rng)) == 3 and new_int("-37", rng).startswith("-") for _ in range(20))
    assert new_int("0x1F", rng) is None and new_int("1_000", rng) is None and new_int("007", rng) is None
    assert new_int("37", random.Random("s")) == new_int("37", random.Random("s"))


@settings(max_examples=200, deadline=None)
@given(mantissa=st.integers(1, 10**6), shift=st.integers(0, 6), seed=st.integers(0, 10**6))
def test_new_float_keeps_magnitude_and_changes_one_digit(mantissa, shift, seed):
    source = f"{mantissa / 10**shift:.{shift}f}" if shift else f"{mantissa}.0"
    new = new_float(source, random.Random(seed))
    assert new is not None and len(new) == len(source) and float(new) != float(source)
    assert sum(a != b for a, b in zip(new, source)) == 1
    lead = lambda s: next(i for i, c in enumerate(s) if c not in "0.")  # noqa: E731
    assert lead(new) == lead(source)  # same magnitude


def test_new_float_and_str_by_hand():
    rng = random.Random(2)
    assert new_float("0.0", rng) is None and new_float("1e-3", rng) is None and new_float("2.5e-3", rng) is not None
    for source in ("'utf-8'", '"strict"', "'ab'", "''", '"x"'):
        new = new_str(source, rng)
        assert new[0] == source[0] and new[-1] == source[-1] and new != source
        if len(source) - 2 >= 3:
            assert len(new) == len(source) and sum(a != b for a, b in zip(new, source)) == 1
        else:
            assert len(new) == len(source) + 2
    for not_plain in ("b'x'", "r'x'", "'''doc'''", "'a\\nb'", "f'{x}'"):
        assert new_str(not_plain, rng) is None


SOURCE = '''"""Doc."""

LIMIT  =  120   # spacing kept
RATIO = 0.25
LABEL = "spool-default"
FLAG = True
NOTHING = None
lower = 5
TAU = -3


def f(a, b=7, *, c='x', d=None, e=False, g=2.5):
    return a


class C:
    def m(self, size=40):
        return size
'''


def test_literal_edits_touch_only_the_literals():
    tree = ast.parse(SOURCE)
    edits = literal_edits("Lib/m.py", SOURCE, tree, seed=1)
    kinds = {(e[3]["kind"], e[3]["qualname"], e[3]["param"]) for e in edits}
    assert kinds == {
        ("constant", "LIMIT", None), ("constant", "RATIO", None), ("constant", "LABEL", None), ("constant", "TAU", None),
        ("default", "f", "b"), ("default", "f", "c"), ("default", "f", "g"), ("default", "C.m", "size"),
    }  # fmt: skip
    new = apply_edits(SOURCE, edits)
    assert "LIMIT  =  " in new and "   # spacing kept" in new and "FLAG = True" in new and "d=None, e=False" in new
    assert ast.parse(new) is not None
    # Outside the edited spans the text is byte-identical.
    keep = SOURCE
    for start, end, _, _ in sorted(edits, reverse=True):
        keep = keep[:start] + "\0" + keep[end:]
    out = new
    for start, end, value, _ in sorted(edits, reverse=True):
        shift = sum(len(v) - (e - s) for s, e, v, _ in edits if s < start)
        out = out[: start + shift] + "\0" + out[start + shift + len(value) :]
    assert out == keep
    record = next(e[3] for e in edits if e[3]["qualname"] == "TAU")
    assert record["old"] == "-3" and record["new"].startswith("-") and record["line"] == 9
    assert literal_edits("Lib/m.py", SOURCE, tree, seed=1) == edits and literal_edits("Lib/m.py", SOURCE, tree, seed=2) != edits


def test_non_ascii_text_before_a_literal():
    text = 'NAME = "é"; LIMIT = 50\nWIDE = "ünïcode-value"\n'
    edits = literal_edits("Lib/u.py", text, ast.parse(text), seed=4)
    new = apply_edits(text, edits)
    values = {n.targets[0].id: n.value.value for n in ast.parse(new).body}
    assert values["NAME"].startswith("é") and len(values["NAME"]) == 3  # a suffix: too short to substitute
    assert values["LIMIT"] != 50 and 10 <= values["LIMIT"] <= 99
    assert len(values["WIDE"]) == len("ünïcode-value") and values["WIDE"] != "ünïcode-value"
    assert new.count(";") == 1 and new.split("\n")[0].startswith('NAME = "é')  # offsets past non-ASCII text are right


def corpus():
    files = sorted(fixture_files().items())
    order = [p for p, _ in files]
    return files, order


def test_renames_are_consistent_and_fresh():
    files, order = corpus()
    texts = dict(files)
    trees = {p: ast.parse(t) for p, t in files if p.endswith(".py")}
    renames = choose_renames(texts, trees, ["Lib/spool.py"], seed=5)
    assert "drain_spool_step0" in renames  # gold names come first
    words = set(re.findall(r"\w+", " ".join(texts.values())))
    for old, (new, path, qualname) in renames.items():
        assert len(old) >= 6 and new not in words and re.fullmatch(r"_?[a-z]+(_[a-z]+)+", new) and qualname.endswith(old)
    assert "__init__" not in renames and "capacity" not in renames
    perturbed, records, excluded = perturb_corpus(files, order, ["Lib/spool.py"], seed=5)
    assert excluded == []
    new0 = renames["drain_spool_step0"][0]
    assert new0 in perturbed["Lib/spool.py"] and "drain_spool_step0" not in perturbed["Lib/spool.py"]
    assert f"from spool import {new0}" in perturbed["Lib/test/test_spool.py"]  # the importing test module too
    assert f":func:`{new0}`" in perturbed["Doc/library/spool.rst"]  # and the prose that names it
    defs = [r for r in records if r["kind"] == "rename" and r["old"] == "drain_spool_step0"]
    assert len(defs) == 1 and defs[0]["path"] == "Lib/spool.py" and defs[0]["qualname"] == "drain_spool_step0"
    for path, text in perturbed.items():
        if path.endswith(".py"):
            assert pyscan.parse(text) is not None
    assert all(set(r) == {"path", "line", "kind", "qualname", "param", "old", "new"} for r in records)
    assert perturb_corpus(files, order, ["Lib/spool.py"], seed=5) == (perturbed, records, excluded)


def test_names_inside_strings_or_defined_twice_are_not_renamed():
    texts = {
        "Lib/a.py": 'def unique_helper():\n    return getattr(obj, "dynamic_lookup")\n\n\ndef dynamic_lookup():\n    pass\n\n\ndef twice_defined():\n    pass\n',
        "Lib/b.py": "def twice_defined():\n    pass\n",
    }
    trees = {p: ast.parse(t) for p, t in texts.items()}
    renames = choose_renames(texts, trees, [], seed=1)
    assert set(renames) == {"unique_helper"}


def test_a_file_that_stops_parsing_is_reverted(monkeypatch):
    files, order = corpus()
    monkeypatch.setattr(perturb_mod, "new_str", lambda source, rng: source[:-1] + "x")  # drops the closing quote
    perturbed, records, excluded = perturb_corpus(files, order, [], seed=5)
    py = {p for p, _ in files if p.endswith(".py") and pyscan.literal_sites(ast.parse(dict(files)[p]))}
    assert py <= set(excluded) and not any(r["path"] in excluded for r in records)
    assert not set(excluded) & set(perturbed)


def test_overlapping_edits_are_refused():
    with pytest.raises(ValueError):
        apply_edits("abcdef", [(0, 3, "x", {}), (2, 4, "y", {})])
