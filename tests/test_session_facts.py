"""BUILD-V2c commit 2: session-fact probes (questions v2s), prompt clp-v2-pilot-3, and each arm's
fair path to a note: B4 keeps it in its conversation, B3 retrieves it with the corpus BM25, L
carries it as a verbatim span that folds, refetches and verifies like any span. The grader
scores the answer value as for every other question."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import pytest

from context_ledger.eval import cli
from context_ledger.eval.arms import note_piece, with_note
from context_ledger.eval.corpus import CorpusIndex, build_corpus, corpus_texts
from context_ledger.eval.prompts import NOTES_SENTENCE, PILOT_PROMPT_V2_2, PILOT_PROMPT_V2_3, PILOT_PROMPT_VERSION, prompt_sha256, system_prompt
from context_ledger.eval.questions import QuestionError
from context_ledger.eval.questions_v2 import (
    FILENAME,
    SESSION_FILENAME,
    SESSION_GAP,
    Mix,
    build_questions_v2,
    capacity,
    corpus_words,
    load_questions_v2,
)
from context_ledger.eval.score import cite_passes, score
from context_ledger.eval.tools import cited_lines
from context_ledger.fold import extractive_summary_v1
from context_ledger.invariants import check
from context_ledger.model import Citation
from context_ledger.store import Ledger, SqliteBlobStore
from context_ledger.tools import dispatch
from context_ledger.window import kappa
from pilot_fixture import v2_fixture_files, write_fixture_tarball
from test_bounded_floor import session as l_session

ARMS = ("B3", "B4", "L")
CORPUS_MIX = "anchors=5,probes=3,multihop=2,unanswerable=2"
SESSION_MIX = CORPUS_MIX + ",notes=4,supersessions=1,session=3,session_unanswerable=1"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("session")
    tar, sha = write_fixture_tarball(tmp, v2_fixture_files())
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=400, rhos=(10, 20), n_gold=8, runs_dir=tmp / "runs", cache_dir=tmp / "cache",
                            progress=lambda m: None, version="v2")  # fmt: skip
    return manifest, tmp / "runs"


@pytest.fixture(scope="module")
def rows(corpus):
    manifest, runs = corpus
    return build_questions_v2(manifest, seed=1, episodes=4, questions=12, mix=Mix.parse(SESSION_MIX), probe_gap=4, runs_dir=runs)


def episodes(rows):
    out = {}
    for r in rows:
        out.setdefault(r["episode"], []).append(r)
    return out


# The mix


def test_the_mix_takes_optional_session_keys_and_v2_mixes_are_unchanged():
    plain = Mix.parse(CORPUS_MIX)
    assert not plain.has_session and plain.to_dict() == {"anchors": 5, "probes": 3, "multihop": 2, "unanswerable": 2}
    mix = Mix.parse(SESSION_MIX)
    assert mix.has_session and mix.total == 12 and mix.to_dict()["session_unanswerable"] == 1
    default = Mix.default_session()
    assert (default.notes, default.supersessions, default.session, default.session_unanswerable) == (6, 2, 4, 2) and default.total == 20
    assert Mix.default_session().anchors == Mix.default().anchors
    with pytest.raises(QuestionError):
        Mix.parse(CORPUS_MIX + ",sessions=3")
    for bad in ("notes=2,session=3", "notes=4,session=2,supersessions=3"):
        with pytest.raises(QuestionError):
            Mix.parse(f"{CORPUS_MIX},{bad}").check(12)


# The generator


def test_episodes_hold_the_session_mix_with_gaps_updates_and_latest_gold(rows, corpus):
    lo, hi = SESSION_GAP
    words = corpus_words(list(corpus_texts(corpus[0]).values()))
    for episode, rs in episodes(rows).items():
        assert [r["pos"] for r in rs] == list(range(16)) and all(r["qid"] == f"v2s{episode:02d}q{r['pos']:02d}" for r in rs)
        probes = [r for r in rs if r["template"] == "session"]
        notes = {r["pos"]: r["note"] for r in rs if r["note"]}
        assert all(r["template"] != "session" for r in rs if r["note"])  # notes sit on corpus questions, one each
        assert Counter(n["kind"] for n in notes.values()) == {"note": 4, "update": 1}
        assert sum(p["answerable"] for p in probes) == 3 and sum(not p["answerable"] for p in probes) == 1
        services = {n["service"] for n in notes.values()} | {p["subject"] for p in probes}
        assert len({re.match(r"[a-z]+", s).group() for s in services}) == 1  # one confusable family per episode
        for p in probes:
            pair = (p["subject"], p["attribute"])
            on_pair = sorted((pos, n) for pos, n in notes.items() if (n["service"], n["attribute"]) == pair)
            if not p["answerable"]:
                assert on_pair == [] and p["answer"] is None  # never noted
                continue
            assert [pos for pos, _ in on_pair] == p["notes_of"] and p["session_gap"] == p["pos"] - p["notes_of"][0] and lo <= p["session_gap"] <= hi
            assert all(pos < p["pos"] for pos in p["notes_of"]) and on_pair[-1][1]["value"] == p["answer"]  # gold: the latest value
            assert p["superseded"] == (len(on_pair) == 2) and (not p["superseded"] or on_pair[1][1]["kind"] == "update")
            assert p["text"] == f"What is the current {p['attribute']} of service {p['subject']}?"
        for n in notes.values():
            value = str(json.loads(n["value"]) if n["value"].isdigit() else n["value"].strip("'"))
            assert value.lower() not in words and value in n["text"]
        for r in rs:
            if r["fold_probe"]:
                anchor = next(a for a in rs if a["qid"] == r["probe_of"])
                assert r["probe_gap"] == r["pos"] - anchor["pos"] >= 4


def test_the_corpus_part_draws_exactly_as_v2_does(corpus, rows):
    manifest, runs = corpus
    plain = build_questions_v2(manifest, seed=1, episodes=4, questions=12, mix=Mix.parse(CORPUS_MIX), probe_gap=4, runs_dir=runs)
    corpus_rows = [r for r in rows if r["template"] != "session"]
    keep = lambda r: (r["episode"], r["text"], r["answer"], r["template"], r["fold_probe"])  # noqa: E731
    assert [keep(r) for r in corpus_rows] == [keep(r) for r in plain]


def test_v2s_writes_its_own_file_deterministically_and_leaves_v2_alone(corpus, rows):
    manifest, runs = corpus
    v2 = runs / manifest["id"] / FILENAME
    if not v2.exists():
        build_questions_v2(manifest, seed=1, episodes=4, questions=12, mix=Mix.parse(CORPUS_MIX), probe_gap=4, runs_dir=runs)
    before = hashlib.sha256(v2.read_bytes()).hexdigest()
    path = runs / manifest["id"] / SESSION_FILENAME
    first = path.read_bytes()
    again = build_questions_v2(manifest, seed=1, episodes=4, questions=12, mix=Mix.parse(SESSION_MIX), probe_gap=4, runs_dir=runs)
    assert again == rows and path.read_bytes() == first and hashlib.sha256(v2.read_bytes()).hexdigest() == before
    meta = json.loads(first.split(b"\n", 1)[0])["metadata"]
    assert meta["version"] == "v2s" and meta["session"]["probes"] == 12 and meta["session"]["supersessions"] == 4 and meta["template_mix"]["session"] == 12
    assert load_questions_v2(manifest, runs, "v2s") == rows
    other = build_questions_v2(manifest, seed=2, episodes=4, questions=12, mix=Mix.parse(SESSION_MIX), probe_gap=4, runs_dir=runs)
    assert [r.get("note") for r in other] != [r.get("note") for r in rows]
    build_questions_v2(manifest, seed=1, episodes=4, questions=12, mix=Mix.parse(SESSION_MIX), probe_gap=4, runs_dir=runs)  # seed 1 again


def test_capacity_reports_the_session_facts(corpus):
    manifest, runs = corpus
    c = capacity(manifest, seed=1, questions=12, mix=Mix.parse(SESSION_MIX), probe_gap=4, runs_dir=runs)
    assert c["session_drawn"] == {"notes": 5 * c["episodes"], "probes": 3 * c["episodes"], "unanswerable": c["episodes"]}
    assert c["unanswerable_drawn"] == 2 * c["episodes"]


def test_the_cli_builds_v2s_and_refuses_session_keys_without_it(tmp_path, capsys):
    tar, sha = write_fixture_tarball(tmp_path, v2_fixture_files())
    base = ["--workdir", str(tmp_path)]
    assert cli.main([*base, "corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
                     "--tarball", str(tar), "--sha256", sha]) == 0  # fmt: skip
    corpus_dir = next((tmp_path / "runs" / "corpus").iterdir())
    common = ["questions", "build", "--questions-per-episode", "12", "--probe-gap", "4"]
    assert cli.main([*base, *common, "--version", "v2", "--mix", SESSION_MIX]) == 2
    assert cli.main([*base, *common, "--version", "v2s", "--mix", CORPUS_MIX]) == 2
    capsys.readouterr()
    assert cli.main([*base, *common, "--version", "v2s", "--mix", SESSION_MIX, "--dry-run"]) == 0
    assert "session facts drawn:" in capsys.readouterr().out and not (corpus_dir / SESSION_FILENAME).exists()
    assert cli.main([*base, *common, "--version", "v2s", "--mix", SESSION_MIX, "--episodes", "2"]) == 0
    assert "session facts: 8 notes, 2 updates" in capsys.readouterr().out and (corpus_dir / SESSION_FILENAME).exists()


# The prompt


def test_prompt_v2_3_is_v2_2_plus_one_sentence_in_every_arm():
    assert PILOT_PROMPT_VERSION == "clp-v2-pilot-1" and PILOT_PROMPT_V2_3 == "clp-v2-pilot-3"
    for arm in ARMS:
        old, new = system_prompt(arm, PILOT_PROMPT_V2_2), system_prompt(arm, PILOT_PROMPT_V2_3)
        assert new == old.replace("counts as evidence.\n", f"counts as evidence. {NOTES_SENTENCE}\n", 1) and new.count(NOTES_SENTENCE) == 1
        assert prompt_sha256(arm, PILOT_PROMPT_V2_3) != prompt_sha256(arm, PILOT_PROMPT_V2_2)
    assert NOTES_SENTENCE == "Notes from the user can be updated later, and the latest value is the one that counts."
    frames = {system_prompt(arm, PILOT_PROMPT_V2_3).split("\n\n", 2)[0] for arm in ARMS}
    assert len(frames) == 1  # the sentence is in the shared frame


# Each arm's path to a note


def test_every_arm_gets_the_note_text_in_the_noted_questions_message(rows):
    q = next(r for r in rows if r["note"])
    assert note_piece(q) == f"[note:{q['pos']}]\n{q['note']['text']}"
    assert with_note(q, "Question: x") == f"[note:{q['pos']}]\n{q['note']['text']}\n\nQuestion: x"
    plain = next(r for r in rows if not r["note"])
    assert with_note(plain, plain["text"]) == plain["text"]


def test_b3_scores_a_note_with_the_corpus_bm25_and_writes_nothing(corpus):
    manifest, _ = corpus
    db = Path(manifest["cache_dir"]) / "index.sqlite"
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    index = CorpusIndex(manifest["cache_dir"], 20)
    try:
        query = "What is the default value of parameter `retries` of `spool.drain_spool_step2`?"
        hits = index.scored_search(query, 10)
        assert hits and [h for _, h in hits] == index.search(query, 10)
        for score_, hit in hits:  # an indexed chunk scored as a document of the index gets FTS5's own score
            body = index.db.execute("SELECT body FROM chunks_20 WHERE path = ? AND start = ?", (hit.path, hit.start)).fetchone()[0]
            assert math.isclose(index.bm25(query, body), score_, rel_tol=1e-12)
        note = "[note:3]\nDeployment note: in this deployment, the retry limit of service orion-relay is 48213."
        assert index.bm25("What is the current retry limit of service orion-relay?", note) < 0
        assert index.bm25("zzzz qqqq", note) is None
    finally:
        index.close()
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_l_carries_a_note_as_a_span_it_verifies_folds_and_refetches():
    s = l_session(100_000, "v2c")
    text = "Deployment note: in this deployment, the retry limit of service orion-relay is 48213."
    source = s.register_source("note:3", text)
    span = s.register_text_span(source, 0, len(text))
    s.append_span(span.span_id, role="user")
    s.append_raw("user", "What is the value of the module constant `spool.SPOOL_LIMIT`?")
    s.call()
    alias = s.state.aliases.alias_of(span.span_id)
    claim_text = "the retry limit of service `orion-relay` is `48213`"
    claim, _ = s.claim(claim_text, [Citation(alias, text)], turn_id="q3")
    assert claim.status == "grounded" and cited_lines(s.state, s.blobs, span.span_id) == "note:3"
    wrong, _ = s.claim("the retry limit of service `orion-relay` is `7`", [Citation(alias, text)], turn_id="q3-other")
    assert wrong.status == "ungrounded"  # Tier 1: the note's quote does not witness another value
    s.emit_answer("q3", [claim.claim_id])
    s.append_raw("assistant", "ANSWER: 48213")
    s.append_raw("user", "What is the current retry limit of service orion-relay?")
    victims = [seg.segment_id for seg in s.state.segments[1:-1]]
    folded = s.fold(victims, extractive_summary_v1(s.state, victims, s.blobs, rules="v2c", digest=True))
    assert kappa(s.state, span.span_id).value == "summary_only"
    s.call()
    line = s.state.aliases.alias_of(f"fold:{folded.fold_id}#0")
    s.append_blocks([{"type": "tool_use", "id": "toolu_n", "name": "ctx_refetch", "input": {"refs": [line]}}])
    out = dispatch(s, "ctx_refetch", {"refs": [line]}, turn_id="q9", tool_use_id="toolu_n")
    assert not out.is_error and kappa(s.state, span.span_id).value == "verbatim"
    s.append_raw("tool", out.text, tool_use_id="toolu_n")
    s.call()
    again, _ = s.claim(claim_text, [Citation(alias, text)], turn_id="q9")
    assert again.status == "grounded"
    assert check(s.events, s.blobs) == []


# The grader


def test_session_answers_are_graded_by_value_and_note_cites_by_quote(rows):
    probe = next(r for r in rows if r["template"] == "session" and r["answerable"])
    unanswerable = next(r for r in rows if r["template"] == "session" and not r["answerable"])
    noted = next(r for r in rows if r["note"] and r["episode"] == probe["episode"] and (r["note"]["service"], r["note"]["attribute"]) == (probe["subject"], probe["attribute"]))
    shown = str(probe["answer"]).strip("'")
    good = {"ref": f"note:{noted['pos']}", "quote": noted["note"]["text"]}
    base = {"rho": 10, "model": "m", "episode": probe["episode"], "bound_hit": False, "calls": 1, "cost_pusd": 0, "blocked_answers": []}
    results = [
        {**base, "arm": "B3", "qid": probe["qid"], "answer": shown, "abstained": False, "citations": [good]},
        {**base, "arm": "B4", "qid": probe["qid"], "answer": "99", "abstained": False, "citations": [good]},
        {**base, "arm": "L", "qid": probe["qid"], "answer": "99", "abstained": False, "citations": [{**good, "quote": "not in the note at all, made up"}]},
        {**base, "arm": "B3", "qid": unanswerable["qid"], "episode": unanswerable["episode"], "answer": None, "abstained": True, "citations": []},
        {**base, "arm": "B4", "qid": unanswerable["qid"], "episode": unanswerable["episode"], "answer": "5", "abstained": False, "citations": []},
    ]
    scored = score(results, rows, {})
    assert [s["outcome"] for s in scored] == ["correct", "wrong", "wrong", "correct_abstain", "false_answer"]
    assert [s["hallucination"] for s in scored] == [None, "misread", "unsupported", None, "unsupported"]
    assert all(s["template"] == "session" for s in scored)
    notes = {f"note:{noted['pos']}": noted["note"]["text"]}
    assert cite_passes(good, {}, notes) and not cite_passes(good, {}, {}) and not cite_passes({**good, "quote": ""}, {}, notes)


# End to end, offline


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """An offline run of every arm over two v2s episodes at W = 3000 with prompt clp-v2-pilot-3."""
    work = tmp_path_factory.mktemp("session-run")
    tar, sha = write_fixture_tarball(work, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    assert e("questions", "build", "--version", "v2s", "--episodes", "3", "--questions-per-episode", "12", "--mix", SESSION_MIX, "--probe-gap", "4") == 0
    assert e("run", "--questions", "v2s", "--rho", "10", "--episodes", "1-2", "--window", "3000", "--budget", "2", "--run-id", "s",
             "--prompt", PILOT_PROMPT_V2_3) == 0  # fmt: skip
    manifest = json.loads(next((work / "runs" / "corpus").glob("*/manifest.json")).read_text())
    return work, work / "runs" / "s", load_questions_v2(manifest, work / "runs" / "corpus", "v2s")


def test_the_run_records_v2s_and_its_questions_hash(run):
    from context_ledger.eval.runner import questions_sha256

    _, run_dir, questions = run
    meta = json.loads((run_dir / "run.json").read_text())
    chosen = [q for q in questions if q["episode"] in (1, 2)]
    assert meta["questions_version"] == "v2s" and meta["prompts"]["version"] == PILOT_PROMPT_V2_3
    assert meta["questions_sha256"] == questions_sha256({arm: chosen for arm in ARMS})


def test_each_arm_reaches_the_notes_its_own_way(run):
    work, run_dir, questions = run
    results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    scored = score(results, questions, corpus_texts(json.loads(next((work / "runs" / "corpus").glob("*/manifest.json")).read_text())))
    session = Counter((s["arm"], s["outcome"]) for s in scored if s["template"] == "session")
    assert session[("B3", "correct")] >= 3 and session[("L", "correct")] >= 3  # B3 by retrieval, L by its spans
    assert all(session[(arm, "false_answer")] == 0 for arm in ARMS)
    l_cites = [c for r in results if r["arm"] == "L" and r["template"] == "session" and r["answer"] is not None for c in r["citations"]]
    assert l_cites and all(re.fullmatch(r"note:\d+", c["ref"]) for c in l_cites)
    noted = [q for q in questions if q["note"] and q["episode"] in (1, 2)]
    transcript = "".join(p.read_text() for p in sorted((run_dir / "B4").glob("*.jsonl")))
    assert all(json.dumps(q["note"]["text"])[1:-1] in transcript for q in noted)  # B4: in its conversation
    for db in sorted((run_dir / "L").glob("episode-*.sqlite")):
        events = Ledger(db).events()
        uris = {e.payload["source"]["uri"] for e in events if e.type == "source_registered"}
        episode = int(db.stem.split("-")[1])
        assert {f"note:{q['pos']}" for q in noted if q["episode"] == episode} <= uris  # L: every note is a registered span
        assert check(events, SqliteBlobStore(db)) == []


def test_eval_quick_and_by_template_show_the_session_template(run, capsys):
    from context_ledger.eval.analyze import by_template_rows

    work, run_dir, questions = run
    capsys.readouterr()
    assert cli.main(["--workdir", str(work), "quick", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert out.count("       session: ") == 3
    results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    rows = by_template_rows(score(results, questions, {}))
    assert {(r[1], r[3]) for r in rows if r[3] == "session"} == {(arm, "session") for arm in ARMS}
