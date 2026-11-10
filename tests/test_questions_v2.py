"""BUILD-V2 item 6: v2 questions (multi-hop, confusable subjects, longer episodes with delayed
fold probes) on the v2 fixture tarball; v1 stays as it was."""

from __future__ import annotations

import ast
import json
import random
import re

import pytest

from context_ledger.eval import cli
from context_ledger.eval.corpus import build_corpus, chunk_lines, corpus_texts
from context_ledger.eval.questions import QuestionError
from context_ledger.eval.questions_v2 import (
    FILENAME,
    Mix,
    ModuleIndex,
    Q,
    build_questions_v2,
    build_v2,
    capacity,
    layout,
    leaks,
    load_questions_v2,
)
from context_ledger.eval.score import score_result
from pilot_fixture import v2_fixture_files, write_fixture_tarball

MIX = Mix(anchors=5, probes=3, multihop=2, unanswerable=2)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("q2")
    tar, sha = write_fixture_tarball(tmp, v2_fixture_files())
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=400, rhos=(10, 20), n_gold=8, runs_dir=tmp / "runs", cache_dir=tmp / "cache",
                            progress=lambda m: None, version="v2")  # fmt: skip
    return manifest, tmp / "runs"


@pytest.fixture(scope="module")
def rows(corpus):
    manifest, runs = corpus
    return build_questions_v2(manifest, seed=1, episodes=6, questions=12, mix=MIX, probe_gap=4, runs_dir=runs)


def by_episode(rows):
    out = {}
    for r in rows:
        out.setdefault(r["episode"], []).append(r)
    return out


# Multi-hop resolution


def resolve(texts: dict[str, str], path: str, expr: str):
    found = ModuleIndex(texts).resolve(path, ast.parse(expr, mode="eval").body)
    return None if found is None else (found[0], found[1])


def test_static_resolution_of_names_attributes_and_relative_imports():
    texts = {
        "Lib/pkg/__init__.py": "",
        "Lib/pkg/consts.py": "LIMIT = 4\nlower = 5\nTWICE = 1\nTWICE = 2\nCOMPUTED = 2 * 3\n",
        "Lib/pkg/sub/deep.py": "from ..consts import LIMIT\nfrom .. import consts\nimport pkg.consts\nimport pkg.consts as pc\nLOCAL = 7\nif x:\n    MAYBE = 1\nMAYBE = 2\n",
        "Lib/pkg/sub/__init__.py": "",
    }
    path = "Lib/pkg/sub/deep.py"
    assert resolve(texts, path, "LIMIT") == ("Lib/pkg/consts.py", "LIMIT")  # from ..consts import LIMIT
    assert resolve(texts, path, "consts.LIMIT") == ("Lib/pkg/consts.py", "LIMIT")  # from .. import consts
    assert resolve(texts, path, "pkg.consts.LIMIT") == ("Lib/pkg/consts.py", "LIMIT")  # import pkg.consts
    assert resolve(texts, path, "pc.LIMIT") == ("Lib/pkg/consts.py", "LIMIT")  # import pkg.consts as pc
    assert resolve(texts, path, "LOCAL") == (path, "LOCAL")
    assert resolve(texts, path, "MAYBE") is None  # bound twice at module level (one in an if)
    assert resolve(texts, path, "consts.lower") is None  # not an upper-case constant
    assert resolve(texts, path, "consts.TWICE") is None  # assigned twice
    assert resolve(texts, path, "consts.COMPUTED") is None  # not a literal
    assert resolve(texts, path, "missing.X") is None and resolve(texts, path, "f(x)") is None


def test_multihop_questions_resolve_within_and_across_files(rows, corpus):
    manifest, _ = corpus
    texts = corpus_texts(manifest)
    multi = [r for r in rows if r["hops"] == 2]
    assert {r["template"] for r in multi} == {"default_ref", "constant_ref"}
    assert any(r["cross_file"] for r in multi) and any(not r["cross_file"] for r in multi)
    small = {f["path"] for f in manifest["files"] if f["rho"] == 10}
    for r in multi:
        (p1, a1, b1), (p2, a2, b2) = r["gold_lines_all"]
        assert (p1, [a1, b1]) == (r["gold_path"], r["gold_lines"]) and r["gold_paths"] == list(dict.fromkeys([p1, p2]))
        assert {p1, p2} <= small  # every hop file is in the smallest corpus
        assert r["cross_file"] == (p1 != p2)
        target = "\n".join(texts[p2].split("\n")[a2 - 1 : b2])
        name, value = target.split("=", 1)
        assert ast.literal_eval(value.strip()) == ast.literal_eval(r["answer"])  # the answer is the hop target's perturbed literal
        assert r["upstream_answer"] != r["answer"] and "Resolve any named constant to its literal value." in r["text"]


def test_every_fixture_multihop_case_is_found(corpus):
    manifest, runs = corpus
    built = build_v2(manifest, seed=1, episodes=0, questions=12, mix=MIX, probe_gap=4, runs_dir=runs)
    texts = {m.candidate.text: m for m in built.pools["Lib/fleet/dock.py"].multi}
    # (function names are renamed by the perturbation, so the parameter and the hop identify each case)
    params = sorted((re.search(r"parameter `(\w+)` of `([\w.]+)`", t).group(1), m.hop.name, m.hop.path, "Dockmaster." in t) for t, m in texts.items() if "parameter" in t)
    assert params == sorted([
        ("limit", "DOCK_LIMIT", "Lib/fleet/consts.py", False),  # from .consts import DOCK_LIMIT
        ("port", "PORT_NAME", "Lib/fleet/consts.py", False),
        ("speed", "LOCAL_SPEED", "Lib/fleet/dock.py", False),  # same module
        ("ratio", "TIDE_RATIO", "Lib/fleet/consts.py", False),  # from . import consts; consts.TIDE_RATIO
        ("cap", "HARBOR_LIMIT", "Lib/harbor.py", False),  # import harbor; harbor.HARBOR_LIMIT
        ("preferred", "LOCAL_SPEED", "Lib/fleet/dock.py", True),  # a method, no class-body shadowing
    ])  # fmt: skip
    constants = {re.search(r"constant `([\w.]+)`", t).group(1): m.hop.name for t, m in texts.items() if "constant `" in t}
    assert constants == {"fleet.dock.BERTH_LIMIT": "DOCK_LIMIT", "fleet.dock.HOME_PORT": "PORT_NAME", "fleet.dock.HARBOR_CAP": "HARBOR_LIMIT",
                         "fleet.dock.CRUISE_SPEED": "LOCAL_SPEED"}  # fmt: skip


# Episodes: size, mix, layout, probe gap, gold files


def test_episodes_have_the_size_mix_layout_and_gold_files(rows):
    episodes = by_episode(rows)
    assert len(episodes) == 6
    for e, qs in episodes.items():
        assert [r["pos"] for r in qs] == list(range(12)) and all(r["qid"] == f"v2e{e:02d}q{r['pos']:02d}" for r in qs)
        probes = [r for r in qs if r["fold_probe"]]
        multi = [r for r in qs if r["hops"] == 2]
        unans = [r for r in qs if not r["answerable"]]
        assert (len(probes), len(unans)) == (3, 2) and len(multi) <= 2 and len(qs) - len(probes) - len(unans) - len(multi) == 5 + 2 - len(multi)
        assert len({r["gold_path"] for r in qs if r["answerable"]}) >= 4
        assert all(r["pos"] >= 6 for r in probes)  # probes in the second half
        assert sum(r["pos"] < 6 for r in unans) == 1 and sum(r["pos"] < 6 for r in multi) == (len(multi) + 1) // 2


def test_probes_ask_about_their_anchors_chunk_at_least_the_gap_later(rows, corpus):
    manifest, _ = corpus
    texts = corpus_texts(manifest)
    qid = {r["qid"]: r for r in rows}
    for probe in (r for r in rows if r["fold_probe"]):
        anchor = qid[probe["probe_of"]]
        assert anchor["episode"] == probe["episode"] and anchor["text"] != probe["text"] and not anchor["fold_probe"]
        assert probe["pos"] - anchor["pos"] == probe["probe_gap"] >= 4
        chunks = chunk_lines(probe["gold_path"], texts[probe["gold_path"]])
        chunk = lambda r: next(k for k, (s, e) in enumerate(chunks) if s <= r["gold_lines"][0] <= e)  # noqa: E731
        assert anchor["gold_path"] == probe["gold_path"] and chunk(anchor) == chunk(probe)


def test_layout_meets_the_gap_or_refuses():
    slots = layout(Mix(9, 6, 3, 2), 20, 8, random.Random(1))
    assert len(slots) == 20 and slots.count("Ap") == slots.count("P") == 6
    anchors, probes = [n for n, k in enumerate(slots) if k == "Ap"], [n for n, k in enumerate(slots) if k == "P"]
    assert all(p - a >= 8 for a, p in zip(anchors, probes)) and max(anchors) < 10 <= min(probes)
    with pytest.raises(QuestionError):
        layout(Mix(9, 6, 3, 2), 20, 15, random.Random(1))
    with pytest.raises(QuestionError):
        layout(Mix(4, 4, 1, 1), 10, 1, random.Random(1))  # 4 probed anchors + 1 multi-hop + 1 unanswerable do not fit a half of 5
    assert Mix.default(20) == Mix(9, 6, 3, 2) and Mix.parse("anchors=9,probes=6,multihop=3,unanswerable=2") == Mix(9, 6, 3, 2)
    for bad in ("anchors=9,probes=6", "anchors=x,probes=6,multihop=3,unanswerable=2"):
        with pytest.raises(QuestionError):
            Mix.parse(bad)
    with pytest.raises(QuestionError):
        Mix(9, 6, 3, 2).check(21)
    with pytest.raises(QuestionError):
        Mix(2, 3, 3, 2).check(10)  # more probes than anchors


def test_too_few_multihop_questions_are_topped_up_with_anchors(corpus):
    manifest, runs = corpus
    built = build_v2(manifest, seed=1, episodes=3, questions=12, mix=Mix(3, 3, 4, 2), probe_gap=4, runs_dir=runs)
    for qs in by_episode(built.rows).values():
        multi = sum(r["hops"] == 2 for r in qs)
        assert len(qs) == 12 and sum(r["fold_probe"] for r in qs) == 3
        assert sum(r["answerable"] and not r["fold_probe"] and r["hops"] == 1 for r in qs) == 3 + 4 - multi


# Leak rules, repeats, determinism


def test_no_repeats_and_no_leaks(rows):
    texts = [r["text"] for r in rows]
    assert len(texts) == len(set(texts))
    for qs in by_episode(rows).values():
        for r in qs:
            if r["template"] == "locate":
                assert not any(o is not r and f"`{r['answer']}." in o["text"] for o in qs)
            if r["hops"] == 2:
                target = (r["gold_lines_all"][1][0], r["gold_lines_all"][1][1])
                for o in qs:
                    if o is not r and o["answerable"] and o["template"] in ("constant", "constant_ref"):
                        assert (o["gold_path"], o["gold_lines"][0]) != target


def test_the_leak_check():
    from context_ledger.eval.questions import Candidate
    from context_ledger.eval.questions_v2 import Hop

    locate = Candidate("locate", "Lib/a.py", "g", "Which module defines the function `g`?", "a", None, (1, 2), ())
    names_a = Candidate("default", "Lib/a.py", "f", "What is the default value of parameter `p` of `a.f`?", "3", "4", (3, 4), ())
    assert leaks([Q(locate, "anchor"), Q(names_a, "anchor")]) and not leaks([Q(locate, "anchor")])
    hop = Hop("Lib/b.py", "LIMIT", (1, 1))
    ref = Candidate("default_ref", "Lib/a.py", "f", "x", "3", "4", (3, 4), ())
    asks = Candidate("constant", "Lib/b.py", "LIMIT", "What is the value of the module constant `b.LIMIT`?", "3", "4", (1, 1), ())
    assert leaks([Q(ref, "multihop", hop=hop), Q(asks, "anchor")]) and not leaks([Q(ref, "multihop", hop=hop)])


def test_determinism_by_seed(corpus, rows):
    manifest, runs = corpus
    again = build_v2(manifest, seed=1, episodes=6, questions=12, mix=MIX, probe_gap=4, runs_dir=runs).rows
    other = build_v2(manifest, seed=2, episodes=6, questions=12, mix=MIX, probe_gap=4, runs_dir=runs).rows
    assert again == rows and other != rows


# Confusers


def test_confusers_count_other_definitions_per_rho(rows, corpus):
    manifest, runs = corpus
    built = build_v2(manifest, seed=1, episodes=None, questions=12, mix=MIX, probe_gap=4, runs_dir=runs)
    rho_of = {f["path"]: f["rho"] for f in manifest["files"]}
    assert rho_of["Tools/scripts/spool_tool.py"] == 20  # the third drain_spool_step0 is only in C20
    drains = [r for r in built.rows if r["answerable"] and r["subject"] == "drain_spool_step0"]
    assert drains and all(r["confusers"] == {"10": 1, "20": 2} for r in drains)
    for r in rows:
        assert set(r["confusers"]) == {"10", "20"} and r["confusers"]["10"] <= r["confusers"]["20"]
        if not r["answerable"]:
            assert r["confusers"] == {"10": 0, "20": 0}
        if r["template"] == "locate":
            assert r["confusers"]["20"] == 0  # locate keeps the unique-definition rule


# Files, CLI, grading, v1 untouched


def test_v2_writes_its_own_file_with_metadata_and_never_touches_v1(corpus, rows):
    manifest, runs = corpus
    path = runs / manifest["id"] / FILENAME
    meta = json.loads(path.read_text().split("\n", 1)[0])["metadata"]
    assert meta["version"] == "v2" and meta["episodes"] == 6 and meta["questions"] == len(rows) and meta["mix"] == MIX.to_dict()
    assert meta["multihop"] == sum(r["hops"] == 2 for r in rows) and meta["probe_gap"] == 4
    assert load_questions_v2(manifest, runs) == rows
    assert not (runs / manifest["id"] / "questions.jsonl").exists()
    with pytest.raises(QuestionError):
        build_questions_v2(manifest, seed=1, episodes=500, questions=12, mix=MIX, probe_gap=4, runs_dir=runs)


def test_dry_run_prints_capacity_and_writes_nothing(tmp_path, capsys):
    tar, sha = write_fixture_tarball(tmp_path, v2_fixture_files())
    base = ["--workdir", str(tmp_path)]
    assert cli.main([*base, "corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
                     "--tarball", str(tar), "--sha256", sha]) == 0  # fmt: skip
    corpus_dir = next((tmp_path / "runs" / "corpus").iterdir())
    before = sorted(p.name for p in corpus_dir.iterdir())
    capsys.readouterr()
    assert cli.main([*base, "questions", "build", "--version", "v2", "--questions-per-episode", "12", "--mix", "anchors=5,probes=3,multihop=2,unanswerable=2",
                     "--probe-gap", "4", "--dry-run"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert re.search(r"capacity: \d+ episodes of 12 questions", out) and "confusers in C20" in out and "default_ref" in out
    assert sorted(p.name for p in corpus_dir.iterdir()) == before
    assert cli.main([*base, "questions", "build", "--dry-run"]) == 2  # v1 has no dry run
    assert cli.main([*base, "questions", "build", "--version", "v2", "--episodes", "2", "--questions-per-episode", "12",
                     "--mix", "anchors=5,probes=3,multihop=2,unanswerable=2", "--probe-gap", "4"]) == 0  # fmt: skip
    assert (corpus_dir / FILENAME).exists() and not (corpus_dir / "questions.jsonl").exists()


def test_capacity_counts_candidates_and_episodes(corpus):
    manifest, runs = corpus
    c = capacity(manifest, seed=1, questions=12, mix=MIX, probe_gap=4, runs_dir=runs)
    assert c["episodes"] >= 6 and c["candidates"]["default_ref"] >= 6 and c["candidates"]["constant_ref"] >= 4
    assert set(c["confusers"]) == {"10", "20"}


def test_the_grader_scores_multihop_answers_by_literal_value(rows):
    row = next(r for r in rows if r["hops"] == 2)
    result = {"arm": "L", "rho": 10, "model": "m", "episode": row["episode"], "qid": row["qid"], "abstained": False, "citations": [], "bound_hit": False,
              "calls": 1, "cost_pusd": 0}  # fmt: skip
    value = ast.literal_eval(row["answer"])
    shown = repr(value) if isinstance(value, str) else str(value)
    assert score_result({**result, "answer": shown}, row, {})["outcome"] == "correct"
    assert score_result({**result, "answer": f'"{value}"' if isinstance(value, str) else f"{value}.0"}, row, {})["outcome"] == "correct"
    wrong = score_result({**result, "answer": "'nope'" if isinstance(value, str) else str(value + 1)}, row, {})
    assert wrong["outcome"] == "wrong" and wrong["template"] == row["template"]
    leaked = score_result({**result, "answer": row["upstream_answer"]}, row, {})
    assert leaked["leaked"] and leaked["has_upstream"]


def test_the_v1_corpus_builder_is_unchanged(tmp_path):
    """v1 (the default) keeps M4's builder id and order; v2 is a different corpus."""
    tar, sha = write_fixture_tarball(tmp_path, v2_fixture_files())
    kw = dict(seed=3, tarball=tar, sha256=sha, window=400, rhos=(10, 20), n_gold=8, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache", progress=lambda m: None)
    v1, v1_again, v2 = build_corpus(**kw), build_corpus(version="v1", **kw), build_corpus(version="v2", **kw)
    assert v1["builder"] == "corpus-v2" and v1["id"] == v1_again["id"] and v2["builder"] == "corpus-v3" and v2["id"] != v1["id"]
    order = [f["path"] for f in v2["files"]]
    assert order.index("Lib/fleet/consts.py") == order.index("Lib/fleet/dock.py") + 1  # the hop file right after its gold file
