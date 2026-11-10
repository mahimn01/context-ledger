"""BUILD-V2b item 3: corpus_read's token cap is min(MAX_READ_TOKENS, W // 4), the same for B4 and
L (B3 has no tools; its chunk filling is unchanged), and run.json records it."""

from __future__ import annotations

import json

import pytest

from context_ledger.eval.corpus import CorpusIndex, lines_of
from context_ledger.eval.tools import MAX_READ_LINES, MAX_READ_TOKENS, LedgerCorpusTools, PlainCorpusTools, read_cap
from context_ledger.pricing import PricingTable
from test_one_w_measure import HAIKU, session, v3

TABLE = PricingTable.verified_default()
PATH = "Lib/spool.py"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from context_ledger.eval.corpus import build_corpus
    from context_ledger.eval.questions import build_questions
    from pilot_fixture import write_fixture_tarball

    tmp = tmp_path_factory.mktemp("read-cap")
    tar, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    return manifest, build_questions(manifest, seed=9, episodes=2, runs_dir=tmp / "runs")


@pytest.fixture(scope="module")
def index(corpus):
    index = CorpusIndex(corpus[0]["cache_dir"], 10)
    yield index
    index.close()


def test_the_cap_is_a_quarter_of_the_window_at_most_2000():
    assert [read_cap(w) for w in (1200, 3000, 6000, 8000, 100_000)] == [300, 750, 1500, 2000, 2000] and MAX_READ_TOKENS == 2000


def whole_file(index) -> dict:
    return {"path": PATH, "start_line": 1, "end_line": index.text(PATH).count("\n") + 1}


def test_b4_reads_at_most_the_cap(index):
    count = v3().count
    assert count(index.text(PATH), "python") > 750  # a whole-file read needs more than a quarter of W = 3000
    text, is_error = PlainCorpusTools(index, count, read_cap(3000))("corpus_read", whole_file(index))
    first, _, rest = text.partition("\n")
    end = int(first.rsplit("-", 1)[1].rstrip("]"))
    assert not is_error and "(truncated; continue at line" in text
    assert count(lines_of(index.text(PATH), 1, end), "python") <= 750
    full, _ = PlainCorpusTools(index, count, read_cap(100_000))("corpus_read", whole_file(index))
    assert "truncated" not in full and len(full) > len(text)


def read_in_l(index, window: int):
    s = session(100_000)
    s.append_raw("user", "What is SPOOL_LIMIT?")
    s.append_blocks([{"type": "tool_use", "id": "toolu_r1", "name": "corpus_read", "input": whole_file(index)}])
    return s, LedgerCorpusTools(index, read_cap(window))(s, "corpus_read", whole_file(index), turn_id="q1", tool_use_id="toolu_r1")


def test_l_reads_at_most_the_same_cap(index):
    s, outcome = read_in_l(index, 3000)
    content = sum(s.counter.count(s.blobs.get(seg.text_hash).decode(), "python") for seg in s.state.segments if seg.ref_kind == "span")
    assert not outcome.is_error and "truncated, continue at line" in outcome.text and outcome.appended and content <= 750
    _, full = read_in_l(index, 100_000)
    assert "truncated" not in full.text and len(full.appended) > len(outcome.appended)


def test_the_line_cap_is_unchanged():
    assert MAX_READ_LINES == 200


def test_run_json_records_the_cap_and_every_read_of_b4_and_l_keeps_it(tmp_path, corpus):
    from context_ledger.eval.runner import offline_client, run_pilot
    from context_ledger.store import Ledger, SqliteBlobStore

    manifest, rows = corpus
    window = 1200
    client, guard, _ = offline_client(tmp_path, run_id="c", pricing=TABLE, budget_usd="0.75")
    try:
        summary = run_pilot(manifest=manifest, rows=rows, arms=["B3", "B4", "L"], rho=10, model=HAIKU, window=window, episodes=1, per_episode=8,
                            smoke=False, run_dir=tmp_path, run_id="c", pricing=TABLE, estimator=v3(), client=client, live=False, budget_usd="0.75")  # fmt: skip
    finally:
        guard.close()
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["read_cap_tokens"] == summary["read_cap_tokens"] == 300
    count = v3().count
    reads = [e for e in map(json.loads, (tmp_path / "B4" / "episode-01.jsonl").read_text().splitlines()) if e["type"] == "tool" and e["name"] == "corpus_read"]
    assert reads and all(count(e["result"].split("\n", 1)[1].split("\n(truncated", 1)[0], "python") <= 300 for e in reads if not e["is_error"])
    db = tmp_path / "L" / "episode-01.sqlite"
    events, blobs = Ledger(db).events(), SqliteBlobStore(db)
    segments = [e.payload["segment"] for e in events if e.type == "segment_appended"]
    reads = {b["id"] for seg in segments if seg.get("content_type") == "blocks" for b in json.loads(blobs.get(seg["text_hash"])) if b.get("name") == "corpus_read"}
    per_result: dict[str, int] = {}
    for seg in segments:
        if seg["ref_kind"] == "span" and seg.get("tool_use_id") in reads:
            per_result[seg["tool_use_id"]] = per_result.get(seg["tool_use_id"], 0) + count(blobs.get(seg["text_hash"]).decode(), "python")
    assert per_result and max(per_result.values()) <= 300
