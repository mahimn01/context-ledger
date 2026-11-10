"""C.3: questions and episodes from the perturbed fixture corpus."""

from __future__ import annotations

import ast
import json
from collections import Counter

import pytest

from context_ledger.eval import pyscan
from context_ledger.eval.corpus import build_corpus, corpus_texts
from context_ledger.eval.questions import QuestionError, build_questions, file_candidates, load_questions, near_miss, smoke_questions
from pilot_fixture import fixture_files, write_fixture_tarball

KEYS = {"qid", "episode", "pos", "template", "text", "answer", "upstream_answer", "gold_path", "gold_lines", "answerable", "fold_probe", "probe_of", "subject"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("questions")
    path, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=path, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    rows = build_questions(manifest, seed=9, episodes=4, runs_dir=tmp / "runs")  # 4 gold files: a second pass shares them
    return tmp, manifest, rows, None


def test_rows_and_episode_shape(built):
    tmp, manifest, rows, warnings = built
    assert rows == load_questions(manifest, tmp / "runs") and all(set(r) == KEYS for r in rows)
    assert len({r["qid"] for r in rows}) == len(rows)
    episodes = sorted({r["episode"] for r in rows})
    assert episodes == [1, 2, 3, 4]
    with pytest.raises(QuestionError, match="built only 4 of 5 episodes"):  # the fixture is far too small for more: fail loudly
        build_questions(manifest, seed=9, episodes=5, runs_dir=tmp / "corpus-too-small")
    for e in episodes:
        ep = sorted((r for r in rows if r["episode"] == e), key=lambda r: r["pos"])
        assert [r["pos"] for r in ep] == list(range(8))
        assert sum(r["answerable"] for r in ep) == 6 and sum(not r["answerable"] for r in ep) == 2
        assert any(not r["answerable"] for r in ep[:3]) and any(not r["answerable"] for r in ep[4:])
        answerable = [r for r in ep if r["answerable"]]
        assert [r["fold_probe"] for r in answerable] == [False] * 3 + [True] * 3  # the last three answerable are probes
        anchors = {r["qid"]: r for r in ep[:4] if r["answerable"]}
        for probe in answerable[3:]:
            anchor = anchors[probe["probe_of"]]
            assert probe["gold_path"] == anchor["gold_path"] and probe["text"] != anchor["text"]
        assert 2 <= len({r["gold_path"] for r in answerable}) <= 3


def test_answers_come_from_the_perturbed_source(built):
    tmp, manifest, rows, _ = built
    texts = corpus_texts(manifest)
    originals = fixture_files()
    edits = [json.loads(line) for line in (tmp / "runs" / manifest["id"] / "perturbations.jsonl").read_text().splitlines()]
    for r in rows:
        if not r["answerable"]:
            assert all(r["subject"] not in t for t in texts.values()) and r["answer"] is None and r["gold_path"] is None
            continue
        path, (start, end) = r["gold_path"], r["gold_lines"]
        span = "\n".join(texts[path].split("\n")[start - 1 : end])
        assert r["subject"] in span
        if r["template"] in ("default", "constant"):
            assert r["answer"] != r["upstream_answer"] and r["answer"] in span
            assert any(e["path"] == path and e["new"] == r["answer"] and e["old"] == r["upstream_answer"] for e in edits)
            assert r["upstream_answer"] in "\n".join(originals[path].split("\n")[start - 1 : end])
        elif r["template"] == "raises":
            func = next(f for q, f in pyscan.functions(ast.parse(texts[path])) if q.rsplit(".", 1)[-1] == r["subject"])
            assert r["answer"] == sorted(pyscan.direct_raises(func))
        else:
            assert r["answer"] == pyscan.module_path(path)
            assert r["subject"] not in originals[path]  # a renamed identifier: memory cannot answer it
    assert {r["template"] for r in rows if r["answerable"]} >= {"default", "constant"}


def test_generation_is_deterministic(built):
    tmp, manifest, rows, _ = built
    assert build_questions(manifest, seed=9, episodes=4, runs_dir=tmp / "runs") == rows
    assert build_questions(manifest, seed=10, episodes=4, runs_dir=tmp / "runs") != rows
    build_questions(manifest, seed=9, episodes=4, runs_dir=tmp / "runs")  # leave the seed-9 file in place


def test_smoke_profile(built):
    _, _, rows, _ = built
    three, probe = smoke_questions(rows)
    assert [r["pos"] for r in three] == [0, 1, 2] and sum(not r["answerable"] for r in three) == 1
    first = next(r for r in three if r["answerable"])
    assert probe["fold_probe"] and probe["probe_of"] == first["qid"] and probe["gold_path"] == first["gold_path"]


def test_unique_definition_filter():
    text = 'LIMIT = 5\n\n\ndef helper_one(x, size=3):\n    raise ValueError("no")\n'
    records = [{"path": "Lib/m.py", "line": 1, "kind": "constant", "qualname": "LIMIT", "param": None, "old": "4", "new": "5"},
               {"path": "Lib/m.py", "line": 4, "kind": "default", "qualname": "helper_one", "param": "size", "old": "2", "new": "3"}]  # fmt: skip
    unique = file_candidates("Lib/m.py", text, records, Counter({"LIMIT": 1, "helper_one": 1}))
    assert {c.template for c in unique} == {"constant", "default", "raises", "locate"}
    twice = file_candidates("Lib/m.py", text, records, Counter({"LIMIT": 2, "helper_one": 2}))
    assert twice == []
    stale = [dict(records[0], new="9")]  # a record that no longer matches the source is ignored
    assert [c.template for c in file_candidates("Lib/m.py", text, stale, Counter({"LIMIT": 1}))] == []


def test_near_miss_is_absent_and_close():
    import random

    haystack = ["def drain_spool_step0(): pass", "drain_spool_step1"]
    for seed in range(20):
        fake = near_miss("drain_spool_step0", random.Random(seed), haystack)
        assert fake and fake != "drain_spool_step0" and all(fake not in h for h in haystack) and "__" not in fake
        assert abs(len(fake) - len("drain_spool_step0")) <= 3


def test_no_candidates_fails_clearly(tmp_path):
    manifest = {"id": "cx", "rhos": [10], "files": [], "gold": [], "cache_dir": str(tmp_path)}
    with pytest.raises(QuestionError):
        build_questions(manifest, seed=1, runs_dir=tmp_path)
