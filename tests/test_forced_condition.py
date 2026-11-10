"""BUILD-V3 item 2: the forced-answer condition, orthogonal to the arms. Prompt clp-v3-forced is
clp-v2-pilot-3 with its abstention sentence replaced in every arm alike; clp-v3-allowed is
clp-v2-pilot-3 byte for byte. `eval run --condition` picks the prompt, refuses a conflicting
--prompt, and records the condition in run.json and every results row. Whole offline runs of every
arm under both conditions, with the scripted fake model."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from context_ledger.anthropic.client import FORCED_MESSAGE
from context_ledger.eval import cli
from context_ledger.eval.corpus import corpus_texts
from context_ledger.eval.prompts import (
    ABSTENTION_SENTENCE,
    CONDITION_PROMPTS,
    FORCED_SENTENCE,
    PILOT_PROMPT_V2_3,
    PILOT_PROMPT_V3_ALLOWED,
    PILOT_PROMPT_V3_FORCED,
    PROMPTS,
    parse_pilot_answer_v2,
    prompt_sha256,
    system_prompt,
)
from context_ledger.eval.questions_v2 import load_questions_v2
from context_ledger.eval.score import score
from context_ledger.invariants import check
from context_ledger.store import Ledger, SqliteBlobStore
from pilot_fixture import v2_fixture_files, write_fixture_tarball

ARMS = ("B3", "B4", "L")
SESSION_MIX = "anchors=5,probes=3,multihop=2,unanswerable=2,notes=4,supersessions=1,session=3,session_unanswerable=1"


def test_allowed_is_clp_v2_pilot_3_byte_for_byte():
    assert CONDITION_PROMPTS == {"allowed": "clp-v3-allowed", "forced": "clp-v3-forced"}
    for arm in ARMS:
        assert system_prompt(arm, PILOT_PROMPT_V3_ALLOWED).encode() == system_prompt(arm, PILOT_PROMPT_V2_3).encode()
        assert prompt_sha256(arm, PILOT_PROMPT_V3_ALLOWED) == prompt_sha256(arm, PILOT_PROMPT_V2_3)


def test_forced_replaces_the_abstention_sentence_in_every_arm_alike():
    assert FORCED_SENTENCE == ("Always give your best answer. Never write UNANSWERABLE: if you are unsure, give your best guess, still in the "
                               "required format with claims and verbatim quotes.")  # fmt: skip
    for arm in ARMS:
        allowed, forced = system_prompt(arm, PILOT_PROMPT_V3_ALLOWED), system_prompt(arm, PILOT_PROMPT_V3_FORCED)
        assert allowed.count(ABSTENTION_SENTENCE) == 1 and forced == allowed.replace(ABSTENTION_SENTENCE, FORCED_SENTENCE)
        assert prompt_sha256(arm, PILOT_PROMPT_V3_FORCED) != prompt_sha256(arm, PILOT_PROMPT_V3_ALLOWED)
    assert len({system_prompt(arm, PILOT_PROMPT_V3_FORCED).split("\n\n", 2)[0] for arm in ARMS}) == 1
    assert PROMPTS[PILOT_PROMPT_V3_FORCED].parse is parse_pilot_answer_v2  # the parser is unchanged
    assert parse_pilot_answer_v2("ANSWER: UNANSWERABLE\n<claims>[]</claims>").abstain


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    work = tmp_path_factory.mktemp("conditions")
    tar, sha = write_fixture_tarball(work, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    assert e("questions", "build", "--version", "v2s", "--episodes", "2", "--questions-per-episode", "12", "--mix", SESSION_MIX, "--probe-gap", "4") == 0
    return work, e


def test_a_conflicting_prompt_is_refused_and_the_run_id_names_the_condition(work, capsys):
    path, e = work
    common = ["run", "--questions", "v2s", "--arms", "B3", "--episodes", "1", "--questions-per-episode", "2", "--window", "3000", "--budget", "1"]
    assert e(*common, "--condition", "forced", "--prompt", PILOT_PROMPT_V2_3) == 2
    assert e(*common, "--prompt", PILOT_PROMPT_V3_FORCED) == 2  # the default condition is allowed
    assert "conflicts with --condition" in capsys.readouterr().err
    assert e(*common, "--condition", "forced") == 0 and e(*common) == 0
    made = sorted(p.name for p in (path / "runs").iterdir() if p.name.startswith("run-"))
    assert len(made) == 2 and any("-forced-" in m for m in made) and any("-allowed-" in m for m in made)
    for name in made:
        run = json.loads((path / "runs" / name / "run.json").read_text())
        assert run["prompts"]["version"] == CONDITION_PROMPTS[run["condition"]]


@pytest.fixture(scope="module")
def runs(work):
    path, e = work
    out = {}
    for condition in ("allowed", "forced"):
        assert e("run", "--questions", "v2s", "--rho", "10", "--episodes", "1-2", "--window", "3000", "--budget", "2", "--run-id", f"c-{condition}",
                 "--condition", condition) == 0  # fmt: skip
        out[condition] = path / "runs" / f"c-{condition}"
    manifest = json.loads(next((path / "runs" / "corpus").glob("*/manifest.json")).read_text())
    return out, load_questions_v2(manifest, path / "runs" / "corpus", "v2s"), corpus_texts(manifest)


def _results(run_dir):
    return [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]


def test_each_run_records_its_condition(runs):
    dirs, questions, _ = runs
    for condition, run_dir in dirs.items():
        meta = json.loads((run_dir / "run.json").read_text())
        assert meta["condition"] == condition and meta["prompts"]["version"] == CONDITION_PROMPTS[condition] and meta["l_mechanism"] == "l-v3"
        assert meta["leakscan"] == "pass" and not meta["incomplete"]
        rows = _results(run_dir)
        assert Counter(r["arm"] for r in rows) == {arm: 32 for arm in ARMS} and {r["condition"] for r in rows} == {condition}
        # BUILD-V3FIX: the patch is in run.json and every L row; B3 and B4 rows are unchanged
        assert meta["harness_patch"] == "v3.1-empty-text-cache-control"
        assert {(r["arm"], r.get("harness_patch")) for r in rows} == {("B3", None), ("B4", None), ("L", "v3.1-empty-text-cache-control")}


def test_allowed_has_no_forced_flags_and_l_records_its_gate(runs):
    dirs, questions, texts = runs
    rows = _results(dirs["allowed"])
    assert not any(r["forced_violation"] or r["forced_fallback"] for r in rows)
    l_rows = [r for r in rows if r["arm"] == "L"]
    assert all(r["gate_target"] is not None for r in l_rows)  # every v2s question has a target
    answered = [r for r in l_rows if r["reason"] == "answered"]
    assert answered and all(r["gate_reason"] == "anchored" for r in answered)
    assert Counter(s["outcome"] for s in score(l_rows, questions, texts))["correct"] >= 10


def test_forced_b3_and_b4_keep_one_call_flag_violations_and_guess(runs):
    dirs, questions, texts = runs
    rows = _results(dirs["forced"])
    for arm in ("B3", "B4"):
        mine = [r for r in rows if r["arm"] == arm]
        violations = [r for r in mine if r["forced_violation"]]
        assert violations and all(r["abstained"] and r["reason"] == "model_abstained" for r in violations)
        assert not any(r["forced_fallback"] for r in mine)
        scored = {s["qid"]: s for s in score(mine, questions, texts)}
        assert all(scored[r["qid"]]["outcome"] in ("abstain", "correct_abstain") for r in violations)  # the unchanged grader: abstentions
        guesses = [r for r in mine if r["answer"] in ("0", "Exception", "unknown.module")]
        assert guesses  # the fake guesses under the forced prompt
    b3 = [r for r in rows if r["arm"] == "B3"]
    assert all(r["calls"] == 1 for r in b3)


def test_forced_l_asks_for_a_guess_and_falls_back_to_its_blocked_candidate(runs):
    dirs, questions, texts = runs
    rows = [r for r in _results(dirs["forced"]) if r["arm"] == "L"]
    fallback = [r for r in rows if r["forced_fallback"]]
    assert fallback and all(r["reason"] == "forced_blocked" and not r["abstained"] and r["answer"] is not None for r in fallback)
    assert all(r["answer"] in r["blocked_answers"] for r in fallback)
    scored = {s["qid"]: s for s in score(rows, questions, texts)}
    assert all(scored[r["qid"]]["outcome"] in ("correct", "wrong", "false_answer") for r in fallback)  # graded like any answer
    asked = 0
    for db in sorted((dirs["forced"] / "L").glob("episode-*.sqlite")):
        events, blobs = Ledger(db).events(), SqliteBlobStore(db)
        asked += sum(e.type == "segment_appended" and e.payload["segment"]["ref_kind"] == "raw" and blobs.get(e.payload["segment"]["text_hash"]).decode() == FORCED_MESSAGE
                     for e in events)  # fmt: skip
        emitted = [e for e in events if e.type == "answer_emitted"]
        assert all(e.payload["abstained"] or e.payload["gate"]["passed"] for e in emitted)  # nothing un-gated left the ledger
        assert check(events, blobs) == []
    assert asked >= 1
