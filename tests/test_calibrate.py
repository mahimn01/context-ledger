"""A.2: the calibrator recovers a known synthetic rule, reports errors exactly and writes no text."""

from __future__ import annotations

import json
import math
from fractions import Fraction

import pytest

from context_ledger.eval.calibrate import (
    CalibrationError,
    CountTokensCounter,
    FakeTokenCounter,
    calibrate,
    draw_sample,
    error_stats,
    ols,
    percentile,
    render_report,
    synthetic_rule,
    write_calibration,
)
from context_ledger.tokens import CLASSES, calibration_estimator, load_calibration

RULES = {
    "python": (Fraction(10, 33), Fraction(7)),
    "prose": (Fraction(1, 4), Fraction(3)),
    "json": (Fraction(2, 5), Fraction(11)),
    "other": (Fraction(2, 7), Fraction(5)),
}


def corpus_files() -> list[tuple[str, str]]:
    """A small synthetic corpus: python modules and prose documents of varied sizes."""
    files = []
    for n in range(6):
        body = "".join(f"def func_{n}_{k}(x, limit={k}):\n    \"\"\"Return x times {k}.\"\"\"\n    return x * {k}\n\n" for k in range(40 + 60 * n))
        files.append((f"Lib/mod{n}.py", f"import os\n\nLIMIT = {n}\n\n" + body))
    for n in range(5):
        files.append((f"Doc/guide{n}.rst", "".join(f"Paragraph {k} explains the option number {k} in plain words.\n" for k in range(80 + 70 * n))))
    files.append(("Misc/NEWS.txt", "".join(f"- Fixed issue {k}.\n" for k in range(500))))
    return files


@pytest.fixture(scope="module")
def sample():
    return draw_sample(corpus_files(), seed=7)


def test_sample_is_stratified_seeded_and_sized(sample):
    assert [sum(s.cls == c for s in sample) for c in CLASSES] == [16, 16, 16, 16]
    assert [sum(s.held_out for s in sample if s.cls == c) for c in CLASSES] == [4, 4, 4, 4]
    assert all(150 <= s.size <= 30 * 1024 for s in sample)
    assert all(s.text.startswith('{"content"') for s in sample if s.cls == "json")
    assert [s.text for s in draw_sample(corpus_files(), seed=7)] == [s.text for s in sample]
    assert [s.text for s in draw_sample(corpus_files(), seed=8)] != [s.text for s in sample]
    with pytest.raises(CalibrationError):
        draw_sample([("Lib/a.py", "x = 1\n")], seed=1)


def known_counter(sample) -> FakeTokenCounter:
    classes = {s.text: s.cls for s in sample}
    return FakeTokenCounter(synthetic_rule(RULES, lambda t: classes.get(t, "other")), overhead=9)


def test_fit_recovers_the_synthetic_coefficients(sample):
    counter = known_counter(sample)
    result = calibrate(sample, counter, model="claude-haiku-5-5", seed=7, created="2026-10-31")
    x_cost = math.ceil(RULES["other"][0] * 1 + RULES["other"][1])  # "x" is counted with the other rule
    assert result["overhead_tokens"] == 9 + x_cost and counter.calls == 3 + 64
    for cls, (a, b) in RULES.items():
        fit = result["coefficients"][cls]
        assert abs(Fraction(fit["a"]) - a) <= a / 200, cls  # within 0.5%
        assert abs(Fraction(fit["b"]) - (b - x_cost)) <= 2, cls  # ceil adds at most one token
        assert Fraction(fit["a_hi"]) >= Fraction(fit["a"])
    errors = result["errors"]
    held = errors["calib_v1_held_out"]
    assert all(Fraction(held[c]["mape"]) < 1 for c in CLASSES)  # the fit is near exact
    assert Fraction(errors["bytes4_all"]["json"]["bias"]) < -30  # bytes/4 undercounts a 2/5 rule
    assert errors["envelope_covers_held_out"].endswith("/16")
    assert "bytes/4 error" in render_report(result)


def test_error_report_is_exact():
    stats = error_stats([(110, 100), (90, 100), (50, 40)])
    assert stats == {"n": 3, "mape": Fraction(10 + 10 + 25, 3), "bias": Fraction(10 - 10 + 25, 3), "max_ape": Fraction(25)}
    assert error_stats([(5, 0)])["n"] == 0
    assert ols([(1, 3), (2, 5), (3, 7)]) == (Fraction(2), Fraction(1))
    assert percentile([Fraction(v) for v in range(1, 21)], 95) == 19 and percentile([Fraction(1)], 95) == 1
    with pytest.raises(CalibrationError):
        ols([(4, 1), (4, 2)])


def test_output_holds_no_text_and_loads_as_calib_v1(sample, tmp_path):
    result = calibrate(sample, known_counter(sample), model="claude-haiku-5-5", seed=7, created="2026-10-31")
    path = write_calibration(result, tmp_path)
    assert path.name == "claude-haiku-5-5-20261031.json"
    raw = path.read_text()
    for s in sample:
        for line in s.text.split("\n"):
            if len(line) >= 24:
                assert line not in raw
    data = json.loads(raw)
    assert {"samples", "coefficients", "errors", "overhead_tokens", "counter"} <= set(data)
    assert all(set(item) == {"cls", "bytes", "sha256", "tokens", "held_out"} for item in data["samples"])
    assert data["counter"] == "fake"
    with pytest.raises(Exception, match="fake"):
        load_calibration("claude-haiku-5-5", tmp_path)
    estimator = load_calibration("claude-haiku-5-5", tmp_path, allow_fake=True)
    assert estimator == calibration_estimator(data, allow_fake=True) and estimator.model == "claude-haiku-5-5"


def test_count_tokens_counter_reads_input_tokens():
    class Messages:
        def __init__(self):
            self.bodies = []

        def count_tokens(self, **body):
            self.bodies.append(body)
            return {"input_tokens": 42}

    class Client:
        messages = Messages()

    client = Client()
    assert CountTokensCounter(client, "claude-haiku-5-5").count_message("hello") == 42
    assert client.messages.bodies == [{"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "hello"}]}]
