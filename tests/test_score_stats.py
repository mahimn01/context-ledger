"""D: scoring normalization and outcomes, bootstrap and McNemar statistics, and the report."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from context_ledger.eval.corpus import build_corpus, corpus_texts
from context_ledger.eval.questions import build_questions
from context_ledger.eval.report import build_report
from context_ledger.eval.runner import offline_client, run_pilot
from context_ledger.eval.score import cite_passes, exception_names, literal, module_name, same_answer, score_result, score_run, set_f1
from context_ledger.eval.stats import Estimate, bootstrap, difference, mcnemar_exact, nearest_rank, paired_counts, ratio, resample_indices
from context_ledger.pricing import PricingTable
from context_ledger.tokens import BYTES4_V0
from pilot_fixture import write_fixture_tarball


def test_mcnemar_exact_by_hand():
    assert mcnemar_exact(0, 0) == 1
    assert mcnemar_exact(1, 5) == Fraction(2 * (1 + 6), 64) == Fraction(7, 32)
    assert mcnemar_exact(5, 1) == Fraction(7, 32)
    assert mcnemar_exact(0, 4) == Fraction(2, 16)
    assert mcnemar_exact(3, 3) == 1  # capped
    assert mcnemar_exact(0, 10) == Fraction(2, 1024)
    assert paired_counts({"a": True, "b": False, "c": True}, {"a": False, "b": True, "c": True, "d": True}) == (1, 1, 3)


def test_ratio_difference_and_nearest_rank():
    assert ratio([(1, 2), (3, 4)]) == Fraction(4, 6) and ratio([(0, 0)]) is None
    assert difference([(1, 1, 0, 1), (1, 1, 1, 1)]) == Fraction(1, 2)
    values = [Fraction(v) for v in range(1, 41)]
    assert nearest_rank(values, Fraction(5, 2)) == 1 and nearest_rank(values, Fraction(195, 2)) == 39
    assert nearest_rank([Fraction(7)], Fraction(5, 2)) == 7


def test_bootstrap_by_hand():
    one = bootstrap([(3, 4)], ratio, resample_indices(1, 50))
    assert one == Estimate(Fraction(3, 4), Fraction(3, 4), Fraction(3, 4), 50)  # one cluster: no spread
    two = bootstrap([(1, 1), (0, 1)], ratio, resample_indices(2, 2000))
    assert two.point == Fraction(1, 2) and (two.low, two.high) == (0, 1)  # resamples take {0, 1/2, 1}
    same = [(2, 4)] * 5
    assert bootstrap(same, ratio).low == bootstrap(same, ratio).high == Fraction(1, 2)
    gaps = bootstrap([(1, 1), (0, 0)], ratio, resample_indices(2, 200))
    assert gaps.valid < 200 and gaps.point == 1  # resamples of only the empty cluster are left out
    assert resample_indices(5, 3) == resample_indices(5, 3) and len(resample_indices(5, 3)) == 3
    assert bootstrap([], ratio) == Estimate(None, None, None, 0)


def test_normalization():
    assert literal("1e3") == 1000.0 and literal("'utf-8'") == "utf-8" and literal(" spam ") == "spam" and literal(41) == 41
    assert same_answer("default", "1e3", "1000.0") and same_answer("constant", 1000, "1000")
    assert same_answer("default", "'utf-8'", '"utf-8"') and same_answer("default", "utf-8", "'utf-8'")
    assert not same_answer("default", "37", "3") and not same_answer("default", "'5'", "5") and not same_answer("default", "True", "1")
    assert module_name("Lib/email/utils.py") == module_name("email/utils.py") == module_name("email.utils") == "email.utils"
    assert module_name("Lib/json/__init__.py") == "json" and same_answer("locate", "Lib/a/b.py", "a.b")
    assert exception_names(["errors.OverflowError", "ValueError"]) == {"OverflowError", "ValueError"}
    assert exception_names("ValueError and KeyError") == {"ValueError", "KeyError"}
    assert set_f1({"A", "B"}, {"A"}) == (2, 3) and set_f1(set(), set()) == (1, 1)
    assert same_answer("raises", ["ValueError", "KeyError"], ["KeyError", "ValueError"]) and not same_answer("raises", ["ValueError"], ["ValueError", "KeyError"])


TEXTS = {"Lib/m.py": "LIMIT = 41\n\ndef f(x, size=3):\n    return x\n"}


def q(**kw):
    base = {"qid": "e01q0", "template": "constant", "answerable": True, "answer": "41", "upstream_answer": "40", "fold_probe": False, "pos": 0}
    return {**base, **kw}


def res(**kw):
    base = {"arm": "B4", "rho": 10, "model": "m", "episode": 1, "qid": "e01q0", "answer": "41", "abstained": False, "citations": [],
            "blocked_answers": [], "bound_hit": False, "calls": 1, "cost_pusd": 5, "error": None}  # fmt: skip
    return {**base, **kw}


def test_outcomes_hallucinations_caught_and_leakage():
    good_cite = {"ref": "Lib/m.py:1-1", "quote": "LIMIT = 41"}
    assert cite_passes(good_cite, TEXTS) and not cite_passes({"ref": "Lib/m.py:3-4", "quote": "LIMIT = 41"}, TEXTS)
    assert not cite_passes({"ref": "@3", "quote": "LIMIT = 41"}, TEXTS) and not cite_passes({"ref": "Lib/m.py:1-1", "quote": None}, TEXTS)
    assert score_result(res(citations=[good_cite]), q(), TEXTS)["outcome"] == "correct"
    wrong = score_result(res(answer="7", citations=[good_cite]), q(), TEXTS)
    assert wrong["outcome"] == "wrong" and wrong["hallucination"] == "misread"  # its cite checks out, it misread
    unsupported = score_result(res(answer="7", citations=[{"ref": "Lib/m.py:3-4", "quote": "LIMIT = 7"}]), q(), TEXTS)
    assert unsupported["hallucination"] == "unsupported"
    assert score_result(res(answer="7"), q(), TEXTS)["hallucination"] == "unsupported"  # no cite at all
    assert score_result(res(answer=None, abstained=True), q(), TEXTS)["outcome"] == "abstain"
    assert score_result(res(answer=None, abstained=True), q(answerable=False, answer=None), TEXTS)["outcome"] == "correct_abstain"
    false = score_result(res(answer="7"), q(answerable=False, answer=None, upstream_answer=None), TEXTS)
    assert false["outcome"] == "false_answer" and false["hallucination"] == "unsupported"
    assert score_result(res(answer="40"), q(), TEXTS)["leaked"] is True
    caught = score_result(res(arm="L", blocked_answers=["0"], citations=[{"ref": "@3", "quote": "LIMIT = 41"}]), q(), TEXTS)
    assert caught["caught"] and caught["outcome"] == "correct"
    assert not score_result(res(arm="L", blocked_answers=["41"]), q(), TEXTS)["caught"]  # the blocked answer was right
    assert score_result(res(arm="L", answer="7", citations=[{"ref": "@3", "quote": "x"}]), q(), TEXTS)["hallucination"] == "unsupported"  # M4: no L shortcut


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("report")
    tar, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    rows = build_questions(manifest, seed=9, episodes=2, runs_dir=tmp / "runs")
    table = PricingTable.verified_default()
    run_dir = tmp / "run"
    client, guard, _ = offline_client(run_dir, run_id="r", pricing=table, budget_usd="0.75")
    run_pilot(manifest=manifest, rows=rows, arms=["B3", "B4", "L"], rho=10, model="claude-haiku-5-5", window=1500, episodes=2, per_episode=8,
              smoke=False, run_dir=run_dir, run_id="r", pricing=table, estimator=BYTES4_V0, client=client, live=False, budget_usd="0.75")  # fmt: skip
    guard.close()
    scores = score_run(run_dir, rows, corpus_texts(manifest))
    return run_dir, scores, build_report(run_dir, resamples=500)


def test_scores_and_report(scored):
    run_dir, scores, report = scored
    assert len(scores) == 48 and (run_dir / "scores.jsonl").exists()
    assert set(report["cells"]) == {"B3|10|claude-haiku-5-5", "B4|10|claude-haiku-5-5", "L|10|claude-haiku-5-5"}
    for cell in report["cells"].values():
        assert cell["counts"]["questions"] == 16 and cell["counts"]["episodes"] == 2
        assert {"HR", "false_answer_rate", "abstention", "selective_accuracy", "leakage", "fold_probe_accuracy", "fold_loss"} <= set(cell)
    b4 = report["cells"]["B4|10|claude-haiku-5-5"]["counts"]
    assert b4["wrong"] > 0 and b4["hallucination_unsupported"] > 0  # the compaction summary made the scripted model guess
    assert report["cells"]["L|10|claude-haiku-5-5"]["counts"]["hallucination_caught"] > 0
    assert len(report["paired"]) == 3 and all(Fraction(p["p_value"]) <= 1 for p in report["paired"])
    assert report["spend"][0]["actual_above_worst"] == 0 and report["spend"][0]["settled_pusd"] > 0
    assert report["folds"] and all(f["predicted_delta"] > 0 and isinstance(f["actual_delta"], int) for f in report["folds"])
    md = (run_dir / "report.md").read_text()
    assert "## Metrics per cell" in md and "## SpendGuard" in md and "## Folds: predicted vs actual (L)" in md and "## HR by rho" in md
    data = json.loads((run_dir / "report.json").read_text())

    def leaves(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from leaves(v)
        elif isinstance(x, list):
            for v in x:
                yield from leaves(v)
        else:
            yield x

    assert not any(isinstance(v, float) for v in leaves(data))
