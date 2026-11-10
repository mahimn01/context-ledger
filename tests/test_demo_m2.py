"""F.1: the M2 demo tells its whole story, deterministically; the cost CLI reads its ledger."""

from __future__ import annotations

import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest

from context_ledger import cli
from context_ledger.ids import parse_frac
from context_ledger.model import EVENT_TYPES
from context_ledger.store import Ledger

ROOT = Path(__file__).resolve().parents[1]
M1_EVENT_TYPES = {"source_registered", "source_updated", "span_registered", "segment_appended", "fold", "evict", "call", "refetch", "claim"}


def load_demo():
    spec = importlib.util.spec_from_file_location("run_demo_m2", ROOT / "demo" / "run_demo_m2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo():
    return load_demo()


@pytest.fixture(scope="module")
def story(demo, tmp_path_factory):
    db = tmp_path_factory.mktemp("demo_m2") / "demo_m2.sqlite"
    return demo.main(db_path=db), db


def leaves(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from leaves(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from leaves(value)
    else:
        yield obj


def test_story(story, capsys):
    result, _ = story
    assert result["ok"] is True and result["violations"] == []
    # Turn 1: search, re-fetch, grounded through numeric normalization ("37" vs 37.0).
    assert result["answers"]["Q1"]["statuses"] == ["grounded"]
    # The backstop refuses discretionary folds that do not pay back, then folds.
    first_fold = result["folds"][0]
    assert result["rejected_folds"] and result["rejected_folds"][0]["call"] < first_fold["call"]
    assert all(r["J_usd"].startswith("-$") and r["zone"] == "discretionary" for r in result["rejected_folds"])
    assert first_fold["zone"] == "discretionary" and parse_frac(first_fold["J"]) > 0 and first_fold["delta"] > 0
    assert all(after < before for before, after in result["window_before_after"])
    assert len(result["window_before_after"]) == len(result["folds"])
    # A bloated model fold is refused without touching the ledger beyond its result.
    assert result["bloated_fold"]["refused"] and result["bloated_fold"]["text"].startswith("error: fold_would_grow_window delta=-")
    # A claim on summary-only evidence is routed to ctx_refetch and re-answered, grounded.
    assert result["answers"]["Q12"]["statuses"] == ["needs_refetch", "grounded"] and result["refetch_routes"] == 1
    assert result["answers"]["Q13"]["reason"] == "model_abstained"
    # After the source edit: one span STALE, one MOVED.
    assert any(line.endswith(" STALE") for line in result["source_edit"])
    assert any(" MOVED @" in line for line in result["source_edit"])
    # Tampered copies are caught with exactly the intended codes.
    assert result["tamper"] == {"freed_tokens+1": ["P12"], "cost_pusd+1": ["P16"]}
    assert {"pricing_set", "alias_assigned", "model_call", "answer_emitted", "fold"} <= set(result["events"])


def test_cost_report_holds_ints_and_strings_only(story):
    report = story[0]["cost_report"]
    assert all(isinstance(v, (int, str)) and not isinstance(v, bool) for v in leaves(report))
    assert report["totals"]["cost_pusd"] == sum(row["cost_pusd"] for row in report["rows"]) > 0
    assert report["totals"]["cost_usd"].startswith("$") and len(report["totals"]["cost_usd"].split(".")[1]) == 6
    assert 0 < report["cache_hit_bp"] < 10000 and Fraction(*map(int, report["cache_hit"].split("/"))) > 0
    assert report["folds"] and all(f["realized_estimate_usd"] and f["predicted_usd"] for f in report["folds"])
    assert len(report["rows"]) == len(story[0]["calls"])


def test_demo_is_deterministic(demo, capsys):
    assert demo.main()["head"] == demo.main()["head"]


def test_cost_cli_and_check_on_the_demo_ledger(story, capsys):
    _, db = story
    capsys.readouterr()
    assert cli.main(["cost", str(db)]) == 0
    out = capsys.readouterr().out
    assert "cache hit ratio" in out and "predicted saving" in out and out.count("claude-haiku-5-5") >= 10
    assert cli.main(["cost", str(db), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["cost_pusd"] > 0
    assert cli.main(["check", str(db)]) == 0
    assert "0 violations" in capsys.readouterr().out


def test_payloads_of_every_new_event_type_hold_no_float(story):
    """The M2 event types (and M1 types with new fields) never carry a float."""
    _, db = story
    with Ledger(db) as ledger:
        events = ledger.events()
    new_types = EVENT_TYPES - M1_EVENT_TYPES - {"tokenizer_set"}  # M3's tokenizer_set: see test_tokens.py
    assert new_types == {"alias_assigned", "pricing_set", "answer_emitted", "model_call"}
    assert new_types <= {e.type for e in events}
    for event in events:
        assert not any(isinstance(v, float) for v in leaves(event.payload)), event.type
