"""M4 item 4: all 12 episodes, deterministically, from a corpus whose C10 is small."""

from __future__ import annotations

import json
import re

import pytest

from context_ledger.eval.corpus import build_corpus, order_files, read_sources
from context_ledger.eval.questions import QuestionError, build_questions, load_question_metadata, load_questions
from context_ledger.tokens import load_calibration
from pilot_fixture import write_fixture_tarball

WORDS = "amber basil cedar dahlia ember fable garnet hazel indigo jasper kettle lumen maple nectar onyx pepper quartz raven sorrel tundra umber velvet willow yarrow".split()
EXC = ["ValueError", "KeyError", "TypeError", "RuntimeError", "LookupError", "OverflowError"]


def live_module(n: int) -> str:
    """A library module questions can be asked about: unique names, perturbable literals, raises."""
    w = WORDS[n]
    funcs = "".join(
        f'\n\ndef {w}_stage{k}_apply(value, retries={3 + k}, label="{w}-{k}", scale={1.5 + k}):\n'
        f'    """Apply stage {k}."""\n    if value is None:\n        raise {EXC[(n + k) % 6]}("missing {k}")\n'
        f"    return value * scale + retries + len(label)\n"
        for k in range(11)
    )
    return f'"""The {w} module."""\n\n{w.upper()}_LIMIT = {100 + n}\n{w.upper()}_MODE = "{w}-mode"\n{funcs}'


def dead_module(n: int) -> str:
    """A codec-table-like module: its methods are defined in every such file, so no question
    passes the unique-definition filter."""
    table = ",\n".join(f"    '\\x{k:02x}'" for k in range(256))
    return (f'"""Codec {n}."""\n\n\nclass Codec:\n    def encode(self, input, errors="strict"):\n        return input\n\n'
            f'    def decode(self, input, errors="strict"):\n        return input\n\n\ndecoding_table = (\n{table}\n)\n')  # fmt: skip


def locate_module(n: int) -> str:
    """Functions with rare names and nothing else to ask about: only locate questions."""
    funcs = "".join(f"\n\ndef {WORDS[n]}_{WORDS[k]}_routine(value):\n    return [value] * {k + 2} + [value, value, value, value]\n" for k in range(24))
    shared = "DEFAULT_SIZE = 4\nDEFAULT_NAME = 'routine'\n"  # literal sites, but defined in every such module: no question
    return f'"""Routines of {WORDS[n]}."""\n\nimport os\n\n{shared}{funcs}'


LIVE = len(WORDS) - 4


def files() -> dict[str, str]:
    out = {f"Lib/encodings/cp{900 + n}.py": dead_module(n) for n in range(8)}
    out.update({f"Lib/{WORDS[n]}.py": live_module(n) for n in range(LIVE)})
    out.update({f"Lib/routines/{WORDS[n]}.py": locate_module(n) for n in range(LIVE, len(WORDS))})
    out["Doc/library/intro.rst"] = "Introduction\n============\n\n" + "Plain words about the library.\n" * 400
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("episodes")
    tar, sha = write_fixture_tarball(tmp, files())
    manifest = build_corpus(seed=1, tarball=tar, sha256=sha, window=1700, rhos=(10,), n_gold=40, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    return tmp, tar, sha, manifest, build_questions(manifest, seed=1, runs_dir=tmp / "runs")


def test_gold_files_are_ordered_by_question_density(built):
    tmp, tar, sha, manifest, _ = built
    gold = manifest["gold"]
    dead = [k for k, p in enumerate(gold) if "encodings" in p]
    assert len(dead) == 8 and min(dead) >= len(gold) - 8  # codec tables come last: nothing to ask about them
    assert order_files(read_sources(tar, sha), 1)[0] == order_files(read_sources(tar, sha), 1)[0]  # deterministic


def test_twelve_full_episodes_with_metadata(built):
    tmp, _, _, manifest, rows = built
    assert rows == load_questions(manifest, tmp / "runs")
    assert len({r["episode"] for r in rows}) == 12 and len(rows) == 96
    assert sum(r["answerable"] for r in rows) == 72 and sum(r["fold_probe"] for r in rows) == 36 and sum(not r["answerable"] for r in rows) == 24
    meta = load_question_metadata(manifest, tmp / "runs")
    assert meta["episodes"] == 12 and meta["questions"] == 96 and meta["fold_probes"] == 36 and meta["unanswerable"] == 24
    assert sum(meta["template_mix"].values()) == 72 and meta["distinct_gold_files"] == len({r["gold_path"] for r in rows if r["answerable"]})
    first = json.loads((tmp / "runs" / manifest["id"] / "questions.jsonl").read_text().split("\n", 1)[0])
    assert set(first) == {"metadata"}


def test_no_question_repeats_probes_stay_in_episode_and_files_serve_at_most_two(built):
    _, _, _, _, rows = built
    assert len({r["text"] for r in rows}) == len(rows)
    by_qid = {r["qid"]: r for r in rows}
    for r in rows:
        if r["fold_probe"]:
            anchor = by_qid[r["probe_of"]]
            assert anchor["episode"] == r["episode"] and anchor["gold_path"] == r["gold_path"] and anchor["pos"] < r["pos"]
    episodes_of = {}
    for r in rows:
        if r["answerable"]:
            episodes_of.setdefault(r["gold_path"], set()).add(r["episode"])
    assert max(len(e) for e in episodes_of.values()) <= 2
    for e in range(1, 13):
        assert 2 <= len({r["gold_path"] for r in rows if r["episode"] == e and r["answerable"]}) <= 3


def test_no_earlier_question_gives_a_locate_answer_away(built):
    _, _, _, _, rows = built
    locate = [r for r in rows if r["template"] == "locate" and r["answerable"]]
    assert locate
    for r in locate:
        earlier = [q["text"] for q in rows if q["episode"] == r["episode"] and q["pos"] < r["pos"]]
        assert not any(re.search(rf"(?<![\w.]){re.escape(r['answer'])}\.", t) for t in earlier), r["qid"]


def test_generation_is_deterministic_and_fails_loudly_when_short(built):
    tmp, _, _, manifest, rows = built
    assert build_questions(manifest, seed=1, runs_dir=tmp / "runs") == rows
    with pytest.raises(QuestionError, match="built only"):
        build_questions(manifest, seed=1, episodes=40, runs_dir=tmp / "runs")
    build_questions(manifest, seed=1, runs_dir=tmp / "runs")  # leave the 12-episode file in place


@pytest.mark.corpus
def test_real_tarball_at_w6000_calib_v1_seed_1(tmp_path):
    """The pilot's corpus: W = 6,000, the live calib-v1 calibration, seed 1."""
    manifest = build_corpus(seed=1, estimator=load_calibration("claude-haiku-5-5"), window=6000, runs_dir=tmp_path / "runs",
                            cache_dir=tmp_path / "cache", progress=lambda m: None)  # fmt: skip
    assert 55_000 <= manifest["totals"]["10"] <= 70_000 and manifest["estimator"]["estimator"] == "calib-v1"
    rows = build_questions(manifest, seed=1, runs_dir=tmp_path / "runs")
    assert len({r["episode"] for r in rows}) == 12 and len(rows) == 96
    assert sum(r["answerable"] for r in rows) == 72 and sum(r["fold_probe"] for r in rows) == 36
    in_c10 = {f["path"] for f in manifest["files"] if f["rho"] == 10}
    assert all(r["gold_path"] in in_c10 for r in rows if r["answerable"])
    meta = load_question_metadata(manifest, tmp_path / "runs")
    assert meta["episodes"] == 12 and meta["distinct_gold_files"] >= 12
    for r in (r for r in rows if r["template"] == "locate" and r["answerable"]):
        earlier = [q["text"] for q in rows if q["episode"] == r["episode"] and q["pos"] < r["pos"]]
        assert not any(re.search(rf"(?<![\w.]){re.escape(r['answer'])}\.", t) for t in earlier)
