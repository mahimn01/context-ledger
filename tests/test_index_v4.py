"""BUILD-V4 item 1: the address index. Exact lookups (every anchored name resolves to its own span
and hash, nothing else does), near misses labelled as different identifiers, the Merkle tree,
the session-note table, and lookup cost: probes (visited nodes) grow with log2 of the corpus's
names, not with the corpus."""

from __future__ import annotations

import hashlib
import math
import statistics
import time

import pytest
from hypothesis import given, settings, strategies as st

from context_ledger.anchor.python import module_parts, python_anchors
from context_ledger.index import AddressIndex, NoteTable, build_address_index, edit_distance, merkle, parse_address, span_sha
from context_ledger.index.build import ELIDED, file_symbols
from pilot_fixture import v2_fixture_files


def _build(tmp_path, texts: dict[str, str], rho: dict[str, int] | None = None, name: str = "ix.sqlite") -> tuple:
    rho = rho or dict.fromkeys(texts, 10)
    path = tmp_path / name
    meta = build_address_index(texts, rho, sorted(set(rho.values())), path, corpus_id="test")
    return path, meta


@pytest.fixture(scope="module")
def fixture_index(tmp_path_factory):
    texts = v2_fixture_files()
    rho = {p: (10 if p.startswith("Lib/") else 20) for p in texts}
    path, meta = _build(tmp_path_factory.mktemp("ix"), texts, rho)
    return texts, path, meta


def test_every_anchored_name_resolves_exactly_to_its_own_span_and_hash(fixture_index):
    texts, path, _ = fixture_index
    ix = AddressIndex(path, 20)
    checked = 0
    for p, text in texts.items():
        if not p.endswith(".py"):
            continue
        module = ".".join(module_parts(p))
        for anchor in python_anchors(text, p):
            if "<locals>" in anchor.qualname:
                continue
            key = f"{module}.{anchor.qualname}"
            found = ix.exact(key)
            assert found, key
            start = text.count("\n", 0, anchor.start) + 1
            end = text.count("\n", 0, anchor.end - 1) + 1
            mine = [a for a in found if (a.path, a.start, a.end) == (p, start, end)]
            assert mine and mine[0].sha == span_sha(text, start, end) == hashlib.sha256("\n".join(text.split("\n")[start - 1 : end]).encode()).hexdigest()
            assert all(a.key == key for a in found)  # exact: never another name
            checked += 1
    assert checked >= 90


def test_summaries_never_show_a_value(fixture_index):
    texts, path, _ = fixture_index
    for p, text in texts.items():
        for parent, leaf, kind, start, end, summary in file_symbols(p, text):
            if kind in ("constant", "param"):
                assert ELIDED in summary and "=" + ELIDED in summary.replace(" ", "")
            if kind == "function":
                assert "retries=3" not in summary and "=" + ELIDED in summary or "=" not in summary


def test_a_bare_name_resolves_by_its_last_component_and_a_path_by_its_file(fixture_index):
    texts, path, _ = fixture_index
    ix = AddressIndex(path, 20)
    found = ix.exact("predict_weather_step2")
    assert [a.key for a in found] == ["weather.predict_weather_step2"]
    both = ix.exact("drain_spool_step0")  # defined in spool.py, fleet/dock.py and (rho 20) a tool script
    assert {a.path for a in both} == {"Lib/spool.py", "Lib/fleet/dock.py", "Tools/scripts/spool_tool.py"}
    assert {a.path for a in AddressIndex(path, 10).exact("drain_spool_step0")} == {"Lib/spool.py", "Lib/fleet/dock.py"}  # C10 only
    node = ix.exact("Lib/spool.py")[0]
    assert (node.kind, node.start, node.end) == ("file", 1, texts["Lib/spool.py"].count("\n") + 1)


def test_a_miss_lists_only_different_identifiers_labelled_as_such(fixture_index):
    _, path, _ = fixture_index
    ix = AddressIndex(path, 20)
    r = ix.lookup("spool.SPOOL_LIMTI")
    assert not r.exact and r.near
    assert r.near[0].address.key == "spool.SPOOL_LIMIT" and "1 edit" in r.near[0].why
    text = r.render()
    assert text.startswith("no exact address for `spool.SPOOL_LIMTI`") and "DIFFERENT identifiers" in text
    assert all(line.strip().startswith("different identifier") for line in text.splitlines()[2:])
    other = ix.lookup("garden.drain_spool_step0")  # the name exists, in other modules only
    assert not other.exact and {n.address.path for n in other.near} >= {"Lib/spool.py"}
    assert any("same name" in n.why for n in other.near)


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_one_character_edits_of_a_name_never_resolve_to_it(fixture_index, data):
    texts, path, _ = fixture_index
    ix = AddressIndex(path, 20)
    keys = sorted({f"{'.'.join(module_parts(p))}.{a.qualname}" for p, t in texts.items() if p.endswith(".py") for a in python_anchors(t, p) if "<locals>" not in a.qualname})
    key = data.draw(st.sampled_from(keys))
    leaf = key.rsplit(".", 1)[-1]
    k = data.draw(st.integers(0, len(leaf) - 1))
    c = data.draw(st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"))
    edited = data.draw(st.sampled_from([leaf[:k] + c + leaf[k + 1 :], leaf[:k] + leaf[k + 1 :], leaf[:k] + c + leaf[k:]]))
    if not edited or edited == leaf:
        return
    asked = key[: len(key) - len(leaf)] + edited
    assert all(a.key == asked for a in ix.exact(asked))  # only ever the name asked, never the original
    if asked not in keys:
        assert key not in [a.key for a in ix.lookup(asked).exact]


def test_edit_distance():
    assert edit_distance("INALID_STATE_ERR", "INVALID_STATE_ERR") == 1
    assert edit_distance("abcd", "abdc") == 1  # an adjacent swap
    assert edit_distance("ENDMARKER", "MINUS") > 2


def test_parse_address():
    assert parse_address("Lib/a.py:10-20#abcd1234") == ("Lib/a.py", 10, 20, "abcd1234")
    assert parse_address("Lib/a.py:7") == ("Lib/a.py", 7, 7, None)
    assert parse_address("a.b.c") is None


def test_merkle_root_certifies_every_byte_and_nests_by_rho(tmp_path):
    texts = v2_fixture_files()
    _, meta = _build(tmp_path, texts, name="a.sqlite")
    changed = dict(texts)
    changed["Lib/spool.py"] = changed["Lib/spool.py"].replace("SPOOL_LIMIT = 120", "SPOOL_LIMIT = 121")
    _, meta2 = _build(tmp_path, changed, name="b.sqlite")
    assert meta["levels"]["10"]["root"] != meta2["levels"]["10"]["root"]
    nodes, root = merkle([("Lib/a.py", "h1", "", 3), ("Lib/b/c.py", "h2", "", 4)])
    nodes2, root2 = merkle([("Lib/b/c.py", "h2", "", 4), ("Lib/a.py", "h1", "", 3)])
    assert root == root2 and sorted(nodes) == sorted(nodes2)  # order-free
    assert root != merkle([("Lib/a.py", "h1", "", 3), ("Lib/b/c.py", "h3", "", 4)])[1]
    ix = AddressIndex(tmp_path / "a.sqlite", 10)
    assert "Lib/fleet/" in ix.outline("Lib") and "spool.SPOOL_LIMIT" in ix.outline("Lib/spool.py")
    assert "Spooler.drain_all" in ix.outline("spool.Spooler")


def test_the_builder_sees_corpus_texts_only():
    import inspect

    params = list(inspect.signature(build_address_index).parameters)
    assert params == ["texts", "member_rho", "rhos", "out", "corpus_id", "progress"]  # no question, gold or run input


def test_the_note_table_follows_the_gates_supersession_rule():
    notes = NoteTable()
    notes.add("note:1", "Deployment note: in this deployment, the queue limit of service phecda-relayd is 68281.")
    notes.add("note:3", "Deployment note: in this deployment, the queue limit of service phecda-relay is 11111.")
    notes.add("note:7", "Update: the queue limit of service phecda-relayd is now 39852.")
    assert notes.chain("phecda-relayd", "queue limit") == ["note:1", "note:7"] and notes.latest("phecda-relayd", "queue limit") == "note:7"
    assert notes.latest("phecda-relay", "queue limit") == "note:3"  # sadr-router never matches sadr-routerd: exact tokens
    assert notes.latest("phecda-relayd", "queue size") is None
    assert ("phecda-relay", "queue limit", "note:3") in notes.different("phecda-relayd", "queue limit")


def _synthetic(modules: int, names: int = 12) -> dict[str, str]:
    """`modules` generated modules of `names` constants and `names` functions each."""
    out = {}
    for m in range(modules):
        body = "".join(f"CONST_{m}_{k} = {k}\n" for k in range(names))
        body += "".join(f"\ndef func_{m}_{k}(x, limit={k}):\n    return x + limit\n" for k in range(names))
        out[f"Lib/pkg{m // 100}/mod{m}.py"] = f'"""Module {m}."""\n\n{body}'
    return out


def test_lookup_probes_grow_with_log_n_not_with_n(tmp_path):
    """Nested levels of 10, 100 and 1000 modules: an exact lookup costs ceil(log2(N + 1)) probes of
    the sorted names of its own level, a miss a few such searches, and the rows read after the
    searches stay under fixed caps. The same 40 names are asked at every level."""
    texts = _synthetic(1000)
    rho = {p: (10 if int(p.rsplit("mod", 1)[1][:-3]) < 10 else 100 if int(p.rsplit("mod", 1)[1][:-3]) < 100 else 1000) for p in texts}
    path, meta = _build(tmp_path, texts, rho)
    asked = [f"mod{m}.CONST_{m}_{k}" for m in range(0, 10, 3) for k in range(0, 12, 3)] + [f"mod{m}.func_{m}_{k}" for m in range(1, 10, 3) for k in range(1, 12, 4)]
    asked = [a.replace("mod", f"pkg0.mod", 1) for a in asked]
    misses = [a[:-1] + "Q" for a in asked]
    stats = {}
    for level in (10, 100, 1000):
        ix = AddressIndex(path, level)
        n = ix.size["sym"]
        hits = [ix.lookup(a) for a in asked]
        assert all(len(h.exact) == 1 for h in hits)
        bound = math.ceil(math.log2(n + 1))
        assert all(h.visited <= bound for h in hits) and all(h.scanned <= 8 for h in hits)
        missed = [ix.lookup(a) for a in misses]
        assert all(not m.exact for m in missed)
        # the prefix walk (one search per level), two last-component tables, the siblings
        assert all(m.visited <= 6 * math.ceil(math.log2(n + 1)) for m in missed)
        assert all(m.scanned <= 2048 + 4 * 8 + 8 for m in missed)
        times = []
        for a in asked * 5:
            t = time.perf_counter()
            ix.lookup(a)
            times.append(time.perf_counter() - t)
        stats[level] = (n, statistics.mean(h.visited for h in hits), statistics.median(times))
    (n10, v10, t10), (n1000, v1000, t1000) = stats[10], stats[1000]
    assert n1000 >= 90 * n10
    assert v1000 - v10 <= math.ceil(math.log2(n1000 / n10)) + 1  # about 7 more probes for 100x the names
    assert t1000 < 10 * t10  # time follows the probes, far from the 100x of a scan


def test_dotted_names_ending_like_a_file_and_bare_class_outlines(tmp_path):
    """From review: `m.f.c` (a parameter) is a name, not a C file; outlining a bare class name
    lists its methods; addresses start at line 1 and never run backwards."""
    path, _ = _build(tmp_path, {"Lib/m.py": "def f(c=1):\n    return c\n\n\nclass Keeper:\n    def guard(self):\n        pass\n"})
    ix = AddressIndex(path, 10)
    assert [a.key for a in ix.exact("m.f.c")] == ["m.f.c"]
    assert "`m.Keeper.guard`" in ix.outline("Keeper")
    assert [a.kind for a in ix.exact("Lib/m.py")] == ["file"]
    assert parse_address("Lib/m.py:0-3") is None and parse_address("Lib/m.py:5-2") is None
