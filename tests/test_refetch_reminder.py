"""BUILD-V2 item 3: before an abstention stands, L checks the ledger once per question for
spans held only behind summary lines that name the question's identifiers, and reminds the
model to refetch them. Also the clp-v2-pilot-1 prompt and the per-question result keys."""

from __future__ import annotations

import json

import pytest

from context_ledger.anthropic.client import LedgerClient
from context_ledger.anthropic.fake import FakeAnthropicServer
from context_ledger.eval.prompts import PILOT_PROMPT_V2, PILOT_PROMPT_VERSION, PROMPTS, prompt_sha256, system_prompt
from context_ledger.invariants import check
from context_ledger.model import SummarySentence
from context_ledger.prompts import SYSTEM_PROMPT, answer_text
from context_ledger.tools import question_identifiers, refetch_candidates
from m2util import MODULE, MODULE_PATH, reply, text, tool_use, v1_session

QUESTION = "What does `inventory.client.backoff_delay` return at most?"


def setup():
    """retrying and backoff_delay folded behind one summary line; DEFAULT_TIMEOUT verbatim."""
    s = v1_session()
    s.append_raw("system", SYSTEM_PROMPT, pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    segs = {n: s.append_span(spans[n].span_id) for n in ["retrying", "backoff_delay", "DEFAULT_TIMEOUT"]}
    s.call()
    s.fold([segs["retrying"].segment_id, segs["backoff_delay"].segment_id],
           [SummarySentence("a retry decorator and its backoff", (spans["retrying"].span_id, spans["backoff_delay"].span_id))])  # fmt: skip
    return s, spans


def turn(s, script, question=QUESTION, max_rounds=8, remind=True):
    server = FakeAnthropicServer(script, pricing=s.pricing)
    wrapper = LedgerClient(s, server.as_fake_client(), model="claude-haiku-5-5", max_tokens=512, remind_refetch=remind)
    return wrapper.run_turn(question, turn_id="t1", max_rounds=max_rounds), server


def last_user_text(request) -> str:
    content = request["messages"][-1]["content"]
    return content if isinstance(content, str) else [b["text"] for b in content if b.get("type") == "text"][-1]


ABSTAIN = reply(text(answer_text([], abstain=True)))


def grounded(alias):
    return reply(text(answer_text([{"text": "`backoff_delay` returns at most `300`", "citations": [{"ref": alias, "quote": "return min(2 ** attempt, 300)"}]}])))


def test_question_identifiers_take_dotted_names_and_their_last_component():
    assert question_identifiers(QUESTION) == ["inventory.client.backoff_delay", "backoff_delay"]
    assert question_identifiers("What is the default value of parameter `errors` of `bz2_codec.lotus`?") == ["errors", "bz2_codec.lotus", "lotus"]
    assert question_identifiers("What is the weather like?") == []


def test_candidates_are_summary_only_spans_naming_an_identifier():
    s, spans = setup()
    # backoff_delay by its name; retrying because its text calls backoff_delay (BM25 ranks it second)
    assert refetch_candidates(s, QUESTION) == [spans["backoff_delay"].span_id, spans["retrying"].span_id]
    assert refetch_candidates(s, "Where is `MAX_RETRIES` used?") == [spans["retrying"].span_id]  # named in retrying's text
    assert refetch_candidates(s, "What is `DEFAULT_TIMEOUT`?") == []  # held verbatim, not folded
    assert refetch_candidates(s, "What is `USER_AGENT`?") == []  # registered but never read


def test_abstain_reminder_refetch_then_grounded_answer():
    s, spans = setup()
    alias, other = (s.state.aliases.alias_of(spans[n].span_id) for n in ("backoff_delay", "retrying"))
    script = [ABSTAIN, reply(tool_use("toolu_1", "ctx_refetch", refs=[alias])), grounded(alias)]
    result, server = turn(s, script)
    assert last_user_text(server.requests[1]) == (
        f"Before abstaining: you no longer hold these verbatim, and they may answer the question: "
        f"{alias} (inventory/client.py#backoff_delay), {other} (inventory/client.py#retrying). Call ctx_refetch on them and answer again, "
        "or abstain again if they do not answer it."
    )
    assert (result.reason, result.abstained, result.reminders, result.refetches) == ("answered", False, 1, 1)
    assert [c.status for c in result.claims] == ["abstain", "grounded"]
    assert check(s.events, s.blobs) == []


def test_no_reminder_when_nothing_folded_matches():
    s, _ = setup()
    result, server = turn(s, [ABSTAIN], question="What is the value of `USER_AGENT`?")
    assert (result.reason, result.reminders, len(server.requests)) == ("model_abstained", 0, 1)
    assert check(s.events, s.blobs) == []


def test_the_reminder_is_sent_at_most_once():
    s, _ = setup()
    result, server = turn(s, [ABSTAIN, ABSTAIN, ABSTAIN])
    assert (result.reason, result.reminders, len(server.requests)) == ("model_abstained", 1, 2)
    assert check(s.events, s.blobs) == []


def test_reminder_before_an_ungrounded_abstention():
    s, spans = setup()
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    folded = s.state.aliases.alias_of(spans["backoff_delay"].span_id)
    wrong = reply(text(answer_text([{"text": "`backoff_delay` returns at most `600`", "citations": [{"ref": alias, "quote": "DEFAULT_TIMEOUT = 37.0"}]}])))
    script = [wrong, wrong, reply(tool_use("toolu_1", "ctx_refetch", refs=[folded])), grounded(folded)]
    result, server = turn(s, script)
    assert last_user_text(server.requests[1]).startswith("Verification: claim 1 was not accepted")
    assert last_user_text(server.requests[2]).startswith("Before abstaining:")
    assert (result.reason, result.reminders, result.refetches) == ("answered", 1, 1)
    assert check(s.events, s.blobs) == []


def test_no_reminder_without_a_round_left_or_when_off():
    s, _ = setup()
    result, _ = turn(s, [ABSTAIN], max_rounds=1)
    assert (result.reason, result.reminders) == ("model_abstained", 0)
    s, _ = setup()
    result, server = turn(s, [ABSTAIN], remind=False)
    assert (result.reason, result.reminders, len(server.requests)) == ("model_abstained", 0, 1)


def test_prompt_v2_1_is_the_default_and_only_l_changes():
    assert PILOT_PROMPT_VERSION == "clp-v2-pilot-1"
    sentence = "Before answering UNANSWERABLE, check ctx_search and ctx_refetch for folded material."
    for arm in ("B3", "B4"):
        assert system_prompt(arm) == system_prompt(arm, PILOT_PROMPT_V2)
    assert system_prompt("L") == system_prompt("L", PILOT_PROMPT_V2).replace(
        "call ctx_refetch and answer again.\n", f"call ctx_refetch and answer again.\n{sentence}\n"
    )
    assert PROMPTS[PILOT_PROMPT_VERSION].frame == PROMPTS[PILOT_PROMPT_V2].frame


def test_the_m4_grid_prompts_stay_reproducible():
    """The sha256 of each arm's clp-m4-pilot-v2 prompt as recorded in runs/grid-haiku-m4."""
    recorded = {
        "B3": "c91a7b4f9d3099f00100caeb2cbae40d5ad47b3006250be709134b36f661f786",
        "B4": "ef8eb83a4ad6030f21ffe3f09970a7c7f370977e1be96a6b354bd34dc0c13289",
        "L": "ec1dc0409f5316c5c75678023c4a5104a9237c1e41a3b7ea1198adcad58818d5",
    }
    assert {arm: prompt_sha256(arm, PILOT_PROMPT_V2) for arm in recorded} == recorded


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    from context_ledger.eval.corpus import build_corpus
    from context_ledger.eval.questions import build_questions
    from pilot_fixture import write_fixture_tarball

    tmp = tmp_path_factory.mktemp("reminder")
    tar, sha = write_fixture_tarball(tmp)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp / "runs", cache_dir=tmp / "cache", progress=lambda m: None)
    return manifest, build_questions(manifest, seed=9, episodes=2, runs_dir=tmp / "runs")


def test_results_carry_the_reminder_keys_for_l_and_null_for_b3_b4(tmp_path, pilot):
    from test_arms import run

    summary, _ = run(tmp_path, pilot)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert summary["l_mechanism"] == "l-v3"
    for row in rows:
        keys = (row["refetch_reminders"], row["refetches"], row["verdict_reasons"])
        if row["arm"] == "L":
            assert row["refetch_reminders"] in (0, 1) and isinstance(row["refetches"], int) and isinstance(row["verdict_reasons"], list)
        else:
            assert keys == (None, None, None)
