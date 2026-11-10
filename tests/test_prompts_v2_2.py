"""BUILD-V2b item 4: prompt clp-v2-pilot-2, a neutral abstention rule, selectable with --prompt;
clp-v2-pilot-1 stays the default and byte-for-byte what the v2a pilot sent."""

from __future__ import annotations

import json

from context_ledger.eval import cli
from context_ledger.eval.prompts import PILOT_PROMPT_V2_1, PILOT_PROMPT_V2_2, PILOT_PROMPT_VERSION, PROMPTS, TOOL_TEXT_V2_1, prompt_sha256, system_prompt

ARMS = ("B3", "B4", "L")
# clp-v2-pilot-1 as recorded in runs/pilot-v2a-haiku/rho-10/run.json (prompts.sha256)
V2_1_SHA256 = {
    "B3": "c91a7b4f9d3099f00100caeb2cbae40d5ad47b3006250be709134b36f661f786",
    "B4": "ef8eb83a4ad6030f21ffe3f09970a7c7f370977e1be96a6b354bd34dc0c13289",
    "L": "9e8912cf5a2197231b36a18a66694f0482cb30dff432f3c86595fab62ca7e9fc",
}
V2_2_SHA256 = {
    "B3": "25364aca2fb3859268991c5d731232fbcba04d354adbc56759e9bfb5764bd88a",
    "B4": "f04ec1c5eeaef36bf6b55815fe3c76ce1ae9f8e367f1c4c5155aba1515eda2b9",
    "L": "5c04b81465023f7afd199618120f268e2324a6c570c9dd822aadc95f3ade0ee0",
}
DELETED = "If you are not sure, abstain. Never guess."
OLD = "If the text shown does not answer the question, or you are not sure, write"
NEW = "If the corpus does not contain the answer, write"


def test_both_versions_are_pinned_and_the_default_is_unchanged():
    assert PILOT_PROMPT_VERSION == PILOT_PROMPT_V2_1 == "clp-v2-pilot-1" and PILOT_PROMPT_V2_2 == "clp-v2-pilot-2"
    assert {arm: prompt_sha256(arm, PILOT_PROMPT_V2_1) for arm in ARMS} == V2_1_SHA256
    assert {arm: prompt_sha256(arm, PILOT_PROMPT_V2_2) for arm in ARMS} == V2_2_SHA256


def test_v2_2_is_v2_1_with_exactly_the_two_edits_in_every_arm():
    for arm in ARMS:
        old, new = system_prompt(arm, PILOT_PROMPT_V2_1), system_prompt(arm, PILOT_PROMPT_V2_2)
        assert old.count(DELETED) == old.count(OLD) == 1 and DELETED not in new and OLD not in new and new.count(NEW) == 1
        assert new == old.replace(DELETED + "\n\n", "").replace(OLD, NEW)
        assert "\n\n\n" not in new and "ANSWER: UNANSWERABLE and <claims>[]</claims>." in new
    assert TOOL_TEXT_V2_1["L"].endswith("Before answering UNANSWERABLE, check ctx_search and ctx_refetch for folded material.")
    assert TOOL_TEXT_V2_1["L"] in system_prompt("L", PILOT_PROMPT_V2_2)
    assert PROMPTS[PILOT_PROMPT_V2_2].tools == PROMPTS[PILOT_PROMPT_V2_1].tools


def test_the_frame_is_identical_across_arms_in_both_versions():
    for version in (PILOT_PROMPT_V2_1, PILOT_PROMPT_V2_2):
        frames = {system_prompt(arm, version).replace(PROMPTS[version].tools[arm], "{tools}") for arm in ARMS}
        assert frames == {PROMPTS[version].frame.replace("{{", "{").replace("}}", "}")}


def test_the_answer_format_and_parser_are_shared():
    a, b = PROMPTS[PILOT_PROMPT_V2_1], PROMPTS[PILOT_PROMPT_V2_2]
    assert (a.parse, a.answer_text, a.reformat) == (b.parse, b.answer_text, b.reformat)
    assert b.parse("ANSWER: UNANSWERABLE\n<claims>[]</claims>").abstain


def test_v2_2_is_selectable_and_recorded_in_run_json(tmp_path):
    from context_ledger.eval.corpus import build_corpus
    from context_ledger.eval.questions import build_questions
    from context_ledger.eval.runner import offline_client, run_pilot
    from context_ledger.pricing import PricingTable
    from context_ledger.tokens import BYTES4_V0
    from pilot_fixture import write_fixture_tarball

    assert cli.build_parser().parse_args(["run", "--budget", "1", "--prompt", PILOT_PROMPT_V2_2]).prompt == PILOT_PROMPT_V2_2
    defaults = cli.build_parser().parse_args(["run", "--budget", "1"])  # BUILD-V3: with no --prompt, the condition picks it
    assert (defaults.prompt, defaults.condition) == (None, "allowed") and PILOT_PROMPT_VERSION == PILOT_PROMPT_V2_1
    tar, sha = write_fixture_tarball(tmp_path)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache", progress=lambda m: None)
    rows = build_questions(manifest, seed=9, episodes=1, runs_dir=tmp_path / "runs")
    table = PricingTable.verified_default()
    client, guard, server = offline_client(tmp_path / "run", run_id="p", pricing=table, budget_usd="0.75")
    try:
        run_pilot(manifest=manifest, rows=rows, arms=["B3"], rho=10, model="claude-haiku-5-5", window=3000, episodes=1, per_episode=2, smoke=False,
                  run_dir=tmp_path / "run", run_id="p", pricing=table, estimator=BYTES4_V0, client=client, live=False, budget_usd="0.75",
                  prompt_version=PILOT_PROMPT_V2_2)  # fmt: skip
    finally:
        guard.close()
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["prompts"]["version"] == PILOT_PROMPT_V2_2 and run["prompts"]["sha256"] == V2_2_SHA256
    assert server.requests and all(r["system"][0]["text"] == system_prompt("B3", PILOT_PROMPT_V2_2) for r in server.requests)
