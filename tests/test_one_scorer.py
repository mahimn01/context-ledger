"""M4 item 3: one scorer for every arm. The ledger's verifier is never the grader."""

from __future__ import annotations

import re

import pytest

from context_ledger.eval.corpus import build_corpus, corpus_texts
from context_ledger.eval.fakemodel import ScriptedPilotModel
from context_ledger.eval.questions import build_questions
from context_ledger.eval.runner import load_run, offline_client, run_pilot
from context_ledger.eval.score import cite_passes, score, score_result
from context_ledger.pricing import PricingTable
from context_ledger.tokens import BYTES4_V0
from pilot_fixture import write_fixture_tarball

TEXTS = {"Lib/m.py": "LIMIT = 41\n\ndef f(x, size=3):\n    return x\n"}
GOOD = {"ref": "Lib/m.py:1-1", "quote": "LIMIT = 41"}
BAD = {"ref": "Lib/m.py:3-4", "quote": "LIMIT = 41"}


def q(**kw):
    return {"qid": "e01q0", "template": "constant", "answerable": True, "answer": "41", "upstream_answer": "40", "fold_probe": False, "pos": 0, **kw}


def res(arm, **kw):
    base = {"arm": arm, "rho": 10, "model": "m", "episode": 1, "qid": "e01q0", "answer": "41", "abstained": False, "citations": [],
            "blocked_answers": [], "bound_hit": False, "calls": 1, "cost_pusd": 5, "error": None}  # fmt: skip
    return {**base, **kw}


ROWS = [
    ({"citations": [GOOD]}, {}),
    ({"answer": "7", "citations": [GOOD]}, {}),
    ({"answer": "7", "citations": [BAD]}, {}),
    ({"answer": "7", "citations": [GOOD, {"ref": "fold:f#0", "quote": None}]}, {}),
    ({"answer": "7"}, {}),
    ({"answer": None, "abstained": True}, {}),
    ({"answer": "7", "citations": [GOOD]}, {"answerable": False, "answer": None, "upstream_answer": None}),
    ({"answer": "40", "citations": [GOOD]}, {}),
]


@pytest.mark.parametrize(("row", "question"), ROWS)
def test_rows_differing_only_in_arm_score_identically(row, question):
    scored = {arm: score_result(res(arm, **row), q(**question), TEXTS) for arm in ("B3", "B4", "L")}
    keys = ("outcome", "hallucination", "leaked")
    assert len({tuple(s[k] for k in keys) for s in scored.values()}) == 1


def test_blocked_answers_carry_their_would_be_grade():
    caught = score_result(res("L", blocked_answers=["0", "41"], citations=[GOOD]), q(), TEXTS)
    assert caught["blocked"] == [{"answer": "0", "grade": "wrong"}, {"answer": "41", "grade": "correct"}] and caught["caught"]
    after_abstain = score_result(res("L", answer=None, abstained=True, blocked_answers=["7"]), q(answerable=False, answer=None), TEXTS)
    assert after_abstain["blocked"] == [{"answer": "7", "grade": "false_answer"}] and after_abstain["caught"]
    not_followed = score_result(res("L", answer="9", blocked_answers=["7"], citations=[GOOD]), q(), TEXTS)
    assert not_followed["outcome"] == "wrong" and not not_followed["caught"]  # a wrong final answer is not a catch


class WrongAnswerModel(ScriptedPilotModel):
    """Answers with verified claims (the true value, quoted) but a wrong ANSWER line."""

    def _format(self, answer, claims, abstain=False):
        text = super()._format(answer, claims, abstain)
        return text if abstain or not self.v2 else re.sub(r"^ANSWER: .*$", "ANSWER: 987654321", text, count=1, flags=re.M)


class OtherDefaultModel(ScriptedPilotModel):
    """BUILD-V3: answers a default question with ANOTHER number of the quoted signature line (the
    default of a different parameter). The quote is about the asked function, names the asked
    parameter and holds the number, so Tier 1 and the answer gate both accept it; it is wrong."""

    def _format(self, answer, claims, abstain=False):
        text = super()._format(answer, claims, abstain)
        quotes = " ".join(str(c.get("quote")) for claim in claims for c in claim["citations"])
        if abstain or not self.v2 or "defaults to" not in "".join(c["text"] for c in claims):
            return text
        other = next((n for n in re.findall(r"=(\d+\.\d+|\d+)\b", quotes) if n != str(answer)), None)
        return text if other is None else re.sub(r"^ANSWER: .*$", f"ANSWER: {other}", text, count=1, flags=re.M)


@pytest.fixture(scope="module")
def fixture_corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scorer")
    tar, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    return manifest, build_questions(manifest, seed=9, episodes=2, runs_dir=tmp / "runs")


def run_l(tmp_path, corpus, responder=None):
    manifest, rows = corpus
    table = PricingTable.verified_default()
    client, guard, _ = offline_client(tmp_path, run_id="s", pricing=table, budget_usd="0.75", responder=responder)
    try:
        run_pilot(manifest=manifest, rows=rows, arms=["L"], rho=10, model="claude-haiku-5-5", window=3000, episodes=1, per_episode=8,
                  smoke=False, run_dir=tmp_path, run_id="s", pricing=table, estimator=BYTES4_V0, client=client, live=False, budget_usd="0.75")  # fmt: skip
    finally:
        guard.close()
    return {r["qid"]: r for r in rows}, load_run(tmp_path)[1], corpus_texts(manifest)


def test_l_citations_are_exported_as_line_ranges_and_checked_like_any_arm(tmp_path, fixture_corpus):
    questions, results, texts = run_l(tmp_path, fixture_corpus)
    answered = [r for r in results if r["reason"] == "answered"]
    assert answered
    for r in answered:
        for c in r["citations"]:
            assert re.fullmatch(r"[^:]+\.py:\d+-\d+", c["ref"]) and c["span"] != c["ref"]
            assert cite_passes(c, texts)  # the grounded span's quote is found at its exported lines


def test_a_wrong_answer_the_verifier_accepted_is_still_scored_wrong(tmp_path, fixture_corpus):
    questions, results, texts = run_l(tmp_path, fixture_corpus, OtherDefaultModel())
    emitted = [r for r in results if r["reason"] == "answered" and r["template"] == "default"]
    scored = score(emitted, questions.values(), texts)
    wrong = [(r, s) for r, s in zip(emitted, scored) if s["outcome"] == "wrong"]
    assert wrong and all("grounded" in r["claim_statuses"] and r["gate_reason"] == "anchored" for r, _ in wrong)  # L's verifier and gate accepted it
    assert all(s["hallucination"] == "misread" for _, s in wrong)  # the citations check out; the answer is wrong


def test_an_answer_no_cited_quote_holds_never_leaves_l(tmp_path, fixture_corpus):
    """BUILD-V3: true claims under a wrong ANSWER line used to be emitted; the gate blocks them."""
    questions, results, texts = run_l(tmp_path, fixture_corpus, WrongAnswerModel())
    assert not any(r["answer"] == "987654321" for r in results)
    blocked = [r for r in results if "987654321" in r["blocked_answers"]]
    assert blocked and all(r["abstained"] and "value_not_with_target" in r["gate_reasons"] for r in blocked)
