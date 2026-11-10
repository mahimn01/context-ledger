"""M4 item 6: predicted cache reads per call, the same way for every arm."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from context_ledger.anthropic.cache import CachePredictor, block_tokens
from context_ledger.anthropic.request import cache_control
from context_ledger.eval.corpus import build_corpus, corpus_texts
from context_ledger.eval.questions import build_questions
from context_ledger.eval.report import build_report
from context_ledger.eval.runner import offline_client, run_pilot
from context_ledger.eval.score import score_run
from context_ledger.pricing import PricingTable
from context_ledger.tokens import BYTES4_V0
from pilot_fixture import write_fixture_tarball

BP = cache_control("5m")


def text(t: str, mark: bool = False) -> dict:
    return {"type": "text", "text": t, **({"cache_control": BP} if mark else {})}


def b3(system: str, context: str, question: str) -> dict:
    """A B3 request: system and retrieved context carry breakpoints, the question does not."""
    return {"model": "m", "max_tokens": 10, "system": [text(system, True)],
            "messages": [{"role": "user", "content": [text(context, True), text(question)]}]}  # fmt: skip


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def total(request: dict) -> int:
    return sum(block_tokens(request, BYTES4_V0))


def test_b3_changing_context_predicts_zero_when_the_system_alone_is_too_short():
    cache = CachePredictor(BYTES4_V0, 512)
    assert cache.predict(b3("short system", "context one " * 300, "q1")) == 0
    assert cache.predict(b3("short system", "context two " * 300, "q2")) == 0  # the prefix changed


def test_b3_reads_only_the_unchanged_system_prefix():
    cache = CachePredictor(BYTES4_V0, 512)
    system = "s" * 4000
    first = b3(system, "context one " * 300, "q1")
    assert cache.predict(first) == 0
    second = b3(system, "context two " * 300, "q2")
    assert cache.predict(second) == block_tokens(second, BYTES4_V0)[0]


def test_repeated_identical_prefix_predicts_the_prefix_size():
    cache = CachePredictor(BYTES4_V0, 512)
    request = b3("s" * 4000, "same context " * 300, "q")
    assert cache.predict(request) == 0
    again = b3("s" * 4000, "same context " * 300, "another question")
    assert cache.predict(again) == sum(block_tokens(again, BYTES4_V0)[:2])  # system and context, not the question


def conversation(n: int, mark_last: bool = True) -> dict:
    content = [text(f"turn {k} " * 60, mark=(mark_last and k == n - 1)) for k in range(n)]
    return {"model": "m", "max_tokens": 10, "system": [text("s" * 3000, True)], "messages": [{"role": "user", "content": content}]}


def test_growing_conversation_reads_the_previous_call_and_ttl_expires():
    clock = Clock()
    cache = CachePredictor(BYTES4_V0, 512, clock=clock)
    cache.predict(conversation(3))
    nxt = conversation(5)
    assert cache.predict(nxt) == sum(block_tokens(nxt, BYTES4_V0)[:4])  # through the block the last call marked
    clock.now += 299
    assert cache.predict(conversation(6)) > 0  # a read refreshes the TTL
    clock.now += 301
    assert cache.predict(conversation(7)) == 0  # everything expired after 5 minutes


def test_lookback_is_twenty_blocks():
    cache = CachePredictor(BYTES4_V0, 1)
    cache.predict(conversation(3))
    near = conversation(23)  # the cached prefix ends 20 blocks before this request's breakpoint
    assert cache.predict(near) > 0
    cache = CachePredictor(BYTES4_V0, 1)
    cache.predict(conversation(3))
    assert cache.predict(conversation(25)) == block_tokens(conversation(25), BYTES4_V0)[0]  # only the system breakpoint


def test_prediction_never_exceeds_the_counted_prompt():
    cache = CachePredictor(BYTES4_V0, 1)
    request = b3("s" * 4000, "c" * 4000, "q")
    cache.predict(request)
    assert cache.predict(request, prompt_total=500) == 500


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(st.integers(1, 30), st.booleans(), st.integers(0, 3000)), min_size=1, max_size=8), st.integers(0, 600))
def test_prediction_is_bounded_by_the_prompt(calls, min_tokens):
    cache = CachePredictor(BYTES4_V0, min_tokens)
    for n, mark, counted in calls:
        request = conversation(n, mark)
        predicted = cache.predict(request, counted if counted else None)
        assert 0 <= predicted <= total(request)
        assert counted == 0 or predicted <= counted


def test_offline_run_predictions_are_per_call_bounded_and_summed(tmp_path):
    tar, sha = write_fixture_tarball(tmp_path)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache", progress=lambda m: None)
    rows = build_questions(manifest, seed=9, episodes=2, runs_dir=tmp_path / "runs")
    table, run_dir = PricingTable.verified_default(), tmp_path / "run"
    client, guard, _ = offline_client(run_dir, run_id="c", pricing=table, budget_usd="0.75")
    try:
        run_pilot(manifest=manifest, rows=rows, arms=["B3", "B4", "L"], rho=10, model="claude-haiku-5-5", window=1500, episodes=1, per_episode=8,
                  smoke=False, run_dir=run_dir, run_id="c", pricing=table, estimator=BYTES4_V0, client=client, live=False, budget_usd="0.75")  # fmt: skip
    finally:
        guard.close()
    score_run(run_dir, rows, corpus_texts(manifest))
    report = build_report(run_dir, resamples=50)
    calls = report["calls"]
    assert calls and all(c["predicted_cache_read"] <= c["predicted_prompt"] for c in calls)
    assert any(c["predicted_cache_read"] > 0 for c in calls)
    for cell, m in report["cells"].items():
        arm = cell.split("|")[0]
        mine = [c for c in calls if c["arm"] == arm]
        t = m["predicted_vs_reported"]
        assert t["calls"] == len(mine) and t["predicted_cache_read"] == sum(c["predicted_cache_read"] for c in mine)
        assert t["reported_cache_read"] == sum(c["reported_cache_read"] for c in mine)
        assert t["predicted_prompt"] == sum(c["predicted_prompt"] for c in mine if c["predicted_prompt"] is not None)
    assert "totals per cell" in (run_dir / "report.md").read_text()
