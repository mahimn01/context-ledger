"""C.1: the corpus builder on a fixture tarball (the real tarball is the opt-in `-m corpus` test)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from context_ledger.eval import corpus as corpus_mod
from context_ledger.eval.corpus import (
    CorpusError,
    CorpusIndex,
    build_corpus,
    chunk_lines,
    cut_points,
    fts_query,
    load_manifest,
    order_files,
    read_sources,
    require_fts5,
)
from context_ledger.tokens import BYTES4_V0
from pilot_fixture import TOPICS, fixture_files, write_fixture_tarball


@pytest.fixture
def tarball(tmp_path):
    return write_fixture_tarball(tmp_path)


def build(tmp_path, tarball, **kw):
    path, sha = tarball
    kw.setdefault("window", 300)
    kw.setdefault("rhos", (1, 10))
    kw.setdefault("n_gold", 4)
    return build_corpus(seed=3, tarball=path, sha256=sha, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache", progress=lambda m: None, **kw)


def test_reads_only_eligible_files_after_verifying_the_hash(tmp_path, tarball):
    path, sha = tarball
    files = dict(read_sources(path, sha))
    assert set(files) == set(fixture_files())
    with pytest.raises(CorpusError, match="refusing"):
        read_sources(path, "0" * 64)
    assert not (tmp_path / "Python-3.15.0").exists()  # never extracted


def test_order_puts_gold_first_then_python_then_the_rest(tarball):
    files = read_sources(*tarball)
    gold, order = order_files(files, seed=3, n_gold=4)
    assert len(gold) == 4 and order[:4] == gold and all(p.startswith("Lib/") and p.endswith(".py") for p in gold)
    rest = order[4:]
    first_other = next(i for i, p in enumerate(rest) if not p.endswith(".py"))
    assert all(p.endswith(".py") for p in rest[:first_other]) and not any(p.endswith(".py") for p in rest[first_other:])
    assert "Lib/test/test_spool.py" not in gold
    assert order_files(files, seed=3, n_gold=4) == (gold, order) and order_files(files, seed=4, n_gold=4) != (gold, order)


def test_cut_points_and_the_shortfall():
    order = ["a", "b", "c"]
    tokens = {"a": 5, "b": 5, "c": 5}
    assert cut_points(order, tokens, {1: 5, 2: 6, 3: 15}) == {1: 1, 2: 2, 3: 3}
    with pytest.raises(CorpusError, match="short by 5"):
        cut_points(order, tokens, {1: 20})


def test_build_writes_a_nested_manifest_and_an_index(tmp_path, tarball):
    manifest = build(tmp_path, tarball, perturb=None)
    assert manifest["release"] == "v3.15.0" and manifest["sha256"] == tarball[1] and manifest["estimator"] == BYTES4_V0.payload()
    totals, targets = manifest["totals"], manifest["targets"]
    assert targets == {"1": 300, "10": 3000} and all(totals[k] >= targets[k] for k in targets)
    counts = manifest["counts"]
    paths = [f["path"] for f in manifest["files"]]
    assert len(paths) == counts["10"] and paths[: len(manifest["gold"])] == manifest["gold"]
    assert [f["rho"] for f in manifest["files"]] == [1] * counts["1"] + [10] * (counts["10"] - counts["1"])
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in manifest["files"])
    assert (tmp_path / "runs" / manifest["id"] / "manifest.json").exists()
    assert (tmp_path / "runs" / manifest["id"] / "perturbations.jsonl").read_text() == "" and manifest["perturbed"] is False
    assert load_manifest(manifest["id"], tmp_path / "runs") == json.loads((tmp_path / "runs" / manifest["id"] / "manifest.json").read_text())
    assert build(tmp_path, tarball, perturb=None)["id"] == manifest["id"]  # the id is a hash of the inputs
    assert build(tmp_path, tarball, perturb=None, window=310)["id"] != manifest["id"]
    perturbed = build(tmp_path, tarball)
    assert perturbed["perturbed"] is True and perturbed["id"] != manifest["id"]
    assert (tmp_path / "runs" / perturbed["id"] / "perturbations.jsonl").read_text().splitlines()

    index = CorpusIndex(tmp_path / "cache" / manifest["id"], 10)
    gold = manifest["gold"][0]
    name, _, verb = next(t for t in TOPICS if f"Lib/{t[0]}.py" == gold)
    hits = index.search(f"{verb} {name} step 3 retries", limit=5)
    assert hits and hits[0].path == gold and f"{verb}_{name}_step3" in index.text(gold)
    assert all(h.end - h.start < 60 for h in hits) and hits[0].snippet
    small = CorpusIndex(tmp_path / "cache" / manifest["id"], 1)
    assert small.paths < index.paths
    with pytest.raises(CorpusError):
        CorpusIndex(tmp_path / "cache" / manifest["id"], 100)
    assert index.search("") == [] and index.search("!!!") == []


def test_shortfall_fails_loudly(tmp_path, tarball):
    with pytest.raises(CorpusError, match="short by"):
        build(tmp_path, tarball, window=16_000, rhos=(10,))


def test_chunks_cut_python_at_top_level_definitions():
    text = "import os\n\n\ndef a():\n    return 1\n\n\n@decorate\ndef b():\n    return 2\n" + "x = 1\n" * 130
    chunks = chunk_lines("Lib/m.py", text)
    assert chunks[:3] == [(1, 3), (4, 7), (8, 67)] and chunks[-1][1] == text.count("\n")
    assert all(e - s < 60 for s, e in chunks)
    prose = chunk_lines("Doc/x.rst", "line\n" * 125)
    assert prose == [(1, 60), (61, 120), (121, 125)]
    assert chunk_lines("Lib/broken.py", "def (:\n" * 70) == [(1, 60), (61, 70)]


def test_fts_query_and_fts5_requirement(monkeypatch):
    assert fts_query("What is DEFAULT_TIMEOUT of `a.b`?") == '"what" OR "is" OR "default" OR "timeout" OR "of" OR "a" OR "b"'
    assert fts_query("?!") is None
    require_fts5()

    class NoFts:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql):
            raise sqlite3.OperationalError("no such module: fts5")

    monkeypatch.setattr(corpus_mod.sqlite3, "connect", lambda *a, **k: NoFts())
    with pytest.raises(CorpusError, match="no FTS5"):
        require_fts5()
