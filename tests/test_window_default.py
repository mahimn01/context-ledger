"""M4 item 7: W = 6,000 is the pilot default (corpus build, run, plan), the planner works at
that window from the scaled prior or a measured smoke run at the same window, and with W = 6,000
the fake-model L and B4 episodes of 8 questions fold and compact."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from context_ledger.eval import cli
from context_ledger.eval.corpus import WINDOW, build_corpus
from context_ledger.eval.plan import HAIKU, PRIOR, SONNET, PlanError, latest_live_smoke, measured_params, plan, prior_params
from context_ledger.eval.questions import build_questions
from context_ledger.eval.runner import load_run, offline_client, run_pilot
from context_ledger.pricing import PricingTable
from context_ledger.tokens import load_calibration, window_estimator
from pilot_fixture import write_fixture_tarball

TABLE = PricingTable.verified_default()
WORDS = "amber basil cedar dahlia ember fable garnet hazel indigo jasper kettle lumen maple nectar onyx pepper".split()


def test_the_default_window_is_6000_everywhere():
    assert WINDOW == 6000
    parser = cli.build_parser()
    assert parser.parse_args(["corpus", "build", "--seed", "1"]).window == 6000
    assert parser.parse_args(["run", "--budget", "1"]).window == 6000
    assert parser.parse_args(["plan"]).window == 6000
    assert plan(TABLE)["window"] == 6000


def test_prior_scales_with_the_window():
    assert prior_params(16_000) == PRIOR
    p = prior_params(6000)[HAIKU]
    assert p.rag.uncached == 6000 and p.summary.uncached == 5100 and p.window == 6000
    assert p.agent.cache_read + p.agent.cache_write + p.agent.uncached == pytest.approx(10_100 * 6000 / 16_000, abs=2)
    assert p.agent.output == PRIOR[HAIKU].agent.output and p.calls_per_question == PRIOR[HAIKU].calls_per_question
    assert plan(TABLE)["haiku_grid"] < plan(TABLE, window=16_000)["haiku_grid"]


def smoke_dir(root, name: str, window: int, cost: int):
    run = root / name
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": name, "live": True, "smoke": True, "model": HAIKU, "window": window}))
    rows = [{"arm": arm, "qid": f"e01q{q}", "kind": "answer", "cost_pusd": cost, "usage": {"input_tokens": 4, "cache_read_input_tokens": 3000,
             "cache_creation_input_tokens": 400, "output_tokens": 150}} for arm in ("B4", "L") for q in range(4) for _ in range(2)]  # fmt: skip
    rows.append({"arm": "B4", "qid": "e01q3", "kind": "summary", "cost_pusd": 10 * cost, "usage": {"input_tokens": 3000, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0, "output_tokens": 500}})  # fmt: skip
    (run / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run


def test_measured_per_episode_figures_at_the_same_window(tmp_path):
    runs = tmp_path / "runs"
    at6000 = smoke_dir(runs, "smoke-a", 6000, 1_000_000)
    smoke_dir(runs, "smoke-b", 16_000, 9_000_000)
    assert latest_live_smoke(runs, 6000) == at6000 and latest_live_smoke(runs, 16_000).name == "smoke-b"
    params = measured_params(at6000)
    m = params[HAIKU]
    assert m.window == 6000 and dict(m.per_episode) == {"B4": Fraction(2 * 1_000_000 * 4 + 10_000_000, 4) * 8, "L": Fraction(2 * 1_000_000 * 4, 4) * 8}
    p = plan(TABLE, params, window=6000)
    assert p["haiku_episode"]["L"] == 16_000_000 and p["haiku_episode"]["B4"] == 36_000_000
    assert p["haiku_episode"]["B3"] == plan(TABLE)["haiku_episode"]["B3"]  # B3 did not run in this smoke: the scaled prior
    assert params[SONNET] == prior_params(6000)[SONNET] and "measured" in p["source"]
    with pytest.raises(PlanError, match="do not transfer"):
        cli._plan_params({"runs": runs}, 6000, str(runs / "smoke-b"))
    assert cli._plan_params({"runs": runs}, 6000, None)[HAIKU].source.endswith("smoke-a")


def big_module(n: int) -> str:
    """One class of many methods, so a read of it is a full 60-line chunk."""
    w = WORDS[n]
    methods = "".join(
        f'    def {w}_{v}_handle(self, value, retries={3 + k}, label="{w}-{k}"):\n'
        f'        """Handle {v} values for the {w} service, retrying when the backend is busy."""\n'
        f"        if value is None:\n            raise ValueError('missing {v}')\n"
        f"        return [value] * retries + [label] * {k + 2}\n\n"
        for k, v in enumerate(WORDS)
    )
    return f'"""The {w} service."""\n\n{w.upper()}_LIMIT = {100 + n}\n{w.upper()}_MODE = "{w}-mode"\n\n\nclass {w.capitalize()}Service:\n{methods}'


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w6000")
    tar, sha = write_fixture_tarball(tmp, {f"Lib/{w}.py": big_module(n) for n, w in enumerate(WORDS)})
    manifest = build_corpus(seed=1, tarball=tar, sha256=sha, window=1500, rhos=(10,), n_gold=16, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    return manifest, build_questions(manifest, seed=1, episodes=1, runs_dir=tmp / "runs")


def test_l_and_b4_episodes_of_eight_questions_fold_and_compact_at_w6000(tmp_path, corpus):
    manifest, rows = corpus
    client, guard, _ = offline_client(tmp_path, run_id="w", pricing=TABLE, budget_usd="0.75")
    try:
        summary = run_pilot(manifest=manifest, rows=rows, arms=["B4", "L"], rho=10, model=HAIKU, window=6000, episodes=1, per_episode=8, smoke=False,
                            run_dir=tmp_path, run_id="w", pricing=TABLE, estimator=window_estimator(load_calibration(HAIKU)), client=client,
                            live=False, budget_usd="0.75")  # fmt: skip
    finally:
        guard.close()
    assert summary["window"] == 6000 and summary["estimator"]["estimator"] == "calib-v2" and not summary["incomplete"]
    results = load_run(tmp_path)[1]
    assert len([r for r in results if r["arm"] == "L"]) == 8 == len([r for r in results if r["arm"] == "B4"])
    assert sum(r["compactions"] for r in results if r["arm"] == "B4") >= 1
    assert sum(r["folds"] for r in results if r["arm"] == "L") >= 1
