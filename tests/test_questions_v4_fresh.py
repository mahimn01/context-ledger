"""BUILD-V4B decision 2: questions v4 never reuse an item questions v3 asked in a run. The overlap rule
(target identifiers and gold answer, or the normalized question text), the filter applied before
episodes are drawn (corpus candidates, near-miss unanswerables, session probes), the extras that
fill the pools when the filter alone leaves fewer than ten episodes, the metadata the file records,
and `questions overlap`. v3's generator is untouched (tests/test_questions_v3.py,
tests/test_v3_reproducible.py)."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter

import pytest

from context_ledger.eval import cli, pyscan
from context_ledger.eval import questions_v2 as q2
from context_ledger.eval.corpus import build_corpus, corpus_texts
from context_ledger.eval.questions import QuestionError, defined_counts
from pilot_fixture import v2_fixture_files, write_fixture_tarball

MIX = "anchors=5,probes=3,multihop=2,unanswerable=2,notes=4,supersessions=1,session=3,session_unanswerable=1"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v4fresh")
    tar, sha = write_fixture_tarball(tmp, v2_fixture_files())
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=400, rhos=(10, 20), n_gold=8, runs_dir=tmp / "runs", cache_dir=tmp / "cache",
                            progress=lambda m: None, version="v2")  # fmt: skip
    q2.build_questions_v2(manifest, seed=2, episodes=2, questions=12, mix=q2.Mix.parse(MIX), probe_gap=4, runs_dir=tmp / "runs", version="v3")
    return manifest, tmp / "runs"


@pytest.fixture(scope="module")
def v4(corpus):
    manifest, runs = corpus
    rows = q2.build_questions_v2(manifest, seed=3, episodes=2, questions=12, mix=q2.Mix.parse(MIX), probe_gap=4, runs_dir=runs, version="v4")
    path = runs / manifest["id"] / q2.V4_FILENAME
    return rows, json.loads(path.read_text().split("\n", 1)[0])["metadata"], path


def test_the_overlap_rule():
    ref = [{"episode": 1, "template": "default", "text": "What is the default value of parameter `p` of `m.f`?", "answer": "'abq'", "answerable": True, "subject": "f"},
           {"episode": 2, "template": "session", "text": "What is the current retry limit of service vega-relay?", "answer": "4821", "answerable": True,
            "subject": "vega-relay"}]  # fmt: skip
    ov = q2.Overlap(frozenset((q2.target_key(r["text"]), q2._answer_key(r["answer"])) for r in ref), frozenset(q2.normalized(r["text"]) for r in ref),
                    frozenset(), {})  # fmt: skip
    assert q2.target_key(ref[0]["text"]) == ("m.f", "p") and q2.target_key(ref[1]["text"]) == ("session", "vega-relay", "retry limit")
    # the same target and answer in other words overlaps; the same words (case and spacing aside) overlap whatever the answer
    assert ov.hit("What value does parameter `p` of `m.f` default to? Resolve any named constant to its literal value.", "'abq'")
    assert ov.hit("what is the  DEFAULT value of parameter `p` of `m.f`?", "'other'")
    assert not ov.hit("What value does parameter `p` of `m.f` default to? Resolve any named constant to its literal value.", "'xyz'")
    assert not ov.hit("What is the default value of parameter `q` of `m.f`?", "'abq'")
    assert not ov.fresh_probe(q2.Fact("vega-relay", "retry limit", "int")) and ov.fresh_probe(q2.Fact("vega-relayd", "retry limit", "int"))


def test_nothing_in_v4_overlaps_the_reference_episodes(corpus, v4):
    manifest, runs = corpus
    rows, meta, _ = v4
    reference = q2.question_rows((runs / manifest["id"] / q2.V3_FILENAME).read_bytes())
    asked = [r for r in reference if r["episode"] in {e for eps in q2.V3_REFERENCE.values() for e in eps}]
    assert asked  # the fixture's v3 episodes 1-2 are grid episodes
    keys = {(q2.target_key(r["text"]), json.dumps(r["answer"])) for r in asked}
    texts = {q2.normalized(r["text"]) for r in asked}
    assert not [r for r in rows if (q2.target_key(r["text"]), json.dumps(r["answer"])) in keys or q2.normalized(r["text"]) in texts]
    report = q2.overlap_report(rows, reference)
    assert report["overlapping_reference"] == 0 and report["items"] == len(rows)
    assert q2.overlap_report(reference, reference)["overlapping_reference"] == len(asked)  # the report sees overlaps when there are some


def test_v4_episodes_are_complete_with_v3s_mix(v4):
    rows, meta, _ = v4
    for e in (1, 2):
        er = [r for r in rows if r["episode"] == e]
        session = [r for r in er if r["template"] == "session"]
        corpus = [r for r in er if r["template"] != "session"]
        assert len(er) == 12 + 3 + 1 and len(corpus) == 12 and sum(not r["answerable"] for r in corpus) == 2
        assert sum(r["answerable"] for r in session) == 3 and sum(r["superseded"] for r in session) == 1
    assert all(r["origin"] in q2.ORIGINS for r in rows if r["answerable"] and r["template"] != "session")


def test_the_metadata_records_the_rule_the_reference_and_the_exclusions(corpus, v4):
    manifest, runs = corpus
    rows, meta, _ = v4
    fresh = meta["fresh"]
    assert fresh["rule"] == q2.OVERLAP_RULE and fresh["min_fresh_episodes"] == q2.MIN_FRESH_EPISODES
    assert fresh["reference"]["sha256"] == hashlib.sha256((runs / manifest["id"] / q2.V3_FILENAME).read_bytes()).hexdigest()
    assert fresh["reference"]["episodes"] == {"grid": list(range(1, 11)), "pilot": [11, 12], "smoke": [19]}
    assert sum(v for k, v in fresh["excluded"].items() if k.startswith("overlap_")) > 0  # the fixture's v3 items were kept out
    # the fixture's small episodes are plentiful with the filter alone: no extras
    assert fresh["episodes_filter_only"] >= q2.MIN_FRESH_EPISODES and not fresh["extras"] and fresh["extra_rule"] is None
    assert fresh["origins"] == dict(sorted(Counter(r["origin"] for r in rows if r["answerable"] and r["template"] != "session").items())) == {"generator": 20}


def test_extras_come_in_only_below_ten_filtered_episodes(corpus, monkeypatch):
    manifest, runs = corpus
    monkeypatch.setattr(q2, "MIN_FRESH_EPISODES", 100)
    built, fresh = q2.build_fresh(manifest, seed=3, episodes=2, questions=12, mix=q2.Mix.parse(MIX), probe_gap=4, runs_dir=runs)
    assert fresh["extras"] and fresh["extra_rule"] == q2.EXTRA_RULE and fresh["episodes_filter_only"] < 100
    assert fresh["episodes_available"] >= fresh["episodes_filter_only"] and len({r["episode"] for r in built.rows}) == 2


def test_the_build_is_deterministic(corpus, v4):
    manifest, runs = corpus
    _, _, path = v4
    before = path.read_bytes()
    q2.build_questions_v2(manifest, seed=3, episodes=2, questions=12, mix=q2.Mix.parse(MIX), probe_gap=4, runs_dir=runs, version="v4")
    assert path.read_bytes() == before


def test_v4_needs_its_reference(corpus, tmp_path):
    manifest, runs = corpus
    (tmp_path / manifest["id"]).mkdir()
    (tmp_path / manifest["id"] / "perturbations.jsonl").write_bytes((runs / manifest["id"] / "perturbations.jsonl").read_bytes())
    with pytest.raises(QuestionError, match="overlap reference"):
        q2.build_questions_v2(manifest, seed=3, episodes=1, questions=12, mix=q2.Mix.parse(MIX), probe_gap=4, runs_dir=tmp_path, version="v4")


def test_extras_ask_in_v3s_wording_about_unperturbed_constants_and_unrenamed_functions():
    text = ('"""m."""\n_MODE_READ = 1\nS_IFDIR = 0o040000\nLIMIT = 4821\nlowercase = 3\nTWICE = 1\nTWICE = 2\n\n'
            "def amber_ridge(x):\n    return x\n\n\ndef helper_function(x):\n    return x\n\n\ndef short(x):\n    return x\n")
    path = "Lib/m.py"
    records = [{"path": path, "line": 4, "kind": "constant", "qualname": "LIMIT", "param": None, "old": "4000", "new": "4821"},
               {"path": path, "line": 9, "kind": "rename", "qualname": "amber_ridge", "param": None, "old": "old_name", "new": "amber_ridge"}]  # fmt: skip
    texts = {path: text}
    extra = q2.extra_candidates(path, text, records, defined_counts(texts), q2.ModuleIndex(texts))
    got = {(c.template, c.subject, c.answer, c.origin) for c in extra}
    assert got == {("constant", "_MODE_READ", "1", "constant_unperturbed"), ("constant", "S_IFDIR", "0o040000", "constant_unperturbed"),
                   ("locate", "helper_function", "m", "locate_unrenamed")}  # fmt: skip
    # LIMIT was perturbed (the v3 generator asks it), TWICE is bound twice, amber_ridge is renamed, short is under 6 characters
    assert all(c.upstream is None for c in extra)
    by = {c.subject: c for c in extra}
    assert by["S_IFDIR"].text == "What is the value of the module constant `m.S_IFDIR`?" and by["helper_function"].text == "Which module defines the function `helper_function`?"
    assert ast.literal_eval(by["S_IFDIR"].answer) == 0o040000


def test_extras_avoid_every_name_v3_asks_about(corpus):
    manifest, runs = corpus
    ov = q2.load_overlap(manifest, runs)
    texts = corpus_texts(manifest)
    reference = q2.question_rows((runs / manifest["id"] / q2.V3_FILENAME).read_bytes())
    named = {n for r in reference if r["template"] != "session" for n in q2._BACKTICKED.findall(r["text"])}
    assert named <= ov.subjects
    built = q2.build_v2(manifest, seed=3, episodes=None, questions=12, mix=q2.Mix.parse(MIX), probe_gap=4, runs_dir=runs, version="v4", overlap=ov, extras=True)
    for r in built.rows:
        if r.get("origin") not in (None, "generator"):
            assert not set(q2._BACKTICKED.findall(r["text"])) & named and f"{r['answer']}.{r['subject']}" not in ov.subjects
    assert texts  # the corpus the extras came from


def test_v4_session_probes_never_repeat_a_reference_probe(corpus, v4):
    manifest, runs = corpus
    rows, _, _ = v4
    reference = q2.question_rows((runs / manifest["id"] / q2.V3_FILENAME).read_bytes())
    asked = {q2.normalized(r["text"]) for r in reference if r["template"] == "session"}
    assert asked and not [r for r in rows if r["template"] == "session" and q2.normalized(r["text"]) in asked]


def test_a_mix_without_session_facts_stays_refused_and_v3_rows_have_no_origin(corpus):
    manifest, runs = corpus
    reference = q2.question_rows((runs / manifest["id"] / q2.V3_FILENAME).read_bytes())
    assert not any("origin" in r for r in reference)
    with pytest.raises(QuestionError):
        q2.build_questions_v2(manifest, seed=3, episodes=1, questions=12, mix=q2.Mix.parse(MIX.split(",notes")[0]), probe_gap=4, runs_dir=runs, version="v4")


def test_the_cli_reports_overlap(tmp_path, capsys):
    tar, sha = write_fixture_tarball(tmp_path, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(tmp_path), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0", "--tarball", str(tar), "--sha256", sha) == 0
    assert e("questions", "build", "--version", "v3", "--seed", "2", "--episodes", "2", "--questions-per-episode", "12", "--mix", MIX, "--probe-gap", "4") == 0
    assert e("questions", "build", "--version", "v4", "--episodes", "2", "--questions-per-episode", "12", "--mix", MIX, "--probe-gap", "4") == 0
    out = capsys.readouterr().out
    assert "fresh: " in out and "episodes with the overlap filter alone" in out
    assert e("questions", "overlap", "--version", "v4") == 0
    assert "0 overlap questions v3 episodes 1-10, 11-12, 19" in capsys.readouterr().out
    assert e("questions", "overlap", "--version", "v3") == 1  # v3 against itself: its grid episodes overlap
    assert e("questions", "build", "--version", "v4", "--episodes", "2", "--questions-per-episode", "12", "--mix", MIX, "--probe-gap", "4", "--dry-run") == 0
    assert "fresh (v4): " in capsys.readouterr().out


def test_unperturbed_answers_have_no_upstream(v4):
    rows, _, _ = v4
    for r in rows:
        if r.get("origin") in ("constant_unperturbed", "locate_unrenamed"):
            assert r["upstream_answer"] is None
        if r.get("origin") == "constant_unperturbed":
            assert pyscan.UPPER.match(r["subject"].lstrip("_"))
