"""M4 item 2: the pilot-v2 answer format (an ANSWER: line and plain claims), with v1 kept."""

from __future__ import annotations

import pytest

from context_ledger.eval.prompts import (
    PILOT_PROMPT_V1,
    PILOT_PROMPT_V2,
    PILOT_PROMPT_VERSION,
    PROMPTS,
    parse_pilot_answer_v2,
    pilot_answer_text_v2,
    prompt_sha256,
    system_prompt,
)
from context_ledger.prompts import AnswerFormatError

ARMS = ("B3", "B4", "L")
# The sha256 of each arm's v1 prompt as recorded in the 2026-10-31 live smoke runs' run.json.
SMOKE_V1_SHA256 = {
    "B3": "1196476ad836b76f97aa6f74987aed54fd1cff549392950c53e9afdb19b1d7cc",
    "B4": "e7fbe548f7c824d6c10edaae827daafa85a2b1b805cb495fe527ceb37d6ddbde",
    "L": "73cf1d0dfcfb17caacf7e810b5c6045bcd6772fc0bd7f4747f32f28be7100081",
}


def test_v2_is_the_default_and_v1_is_kept_byte_for_byte():
    # BUILD-V2 made clp-v2-pilot-1 the default (tests/test_prompts_v2_1.py); M4's v2 is kept.
    assert PILOT_PROMPT_V2 == "clp-m4-pilot-v2" and PILOT_PROMPT_VERSION == "clp-v2-pilot-1"
    assert {arm: prompt_sha256(arm, PILOT_PROMPT_V1) for arm in ARMS} == SMOKE_V1_SHA256
    assert all(prompt_sha256(arm, PILOT_PROMPT_V2) != SMOKE_V1_SHA256[arm] for arm in ARMS)


def test_v2_answer_format_text_is_identical_across_arms():
    prompts = {arm: system_prompt(arm) for arm in ARMS}
    framing = {p.split("\n\n", 1)[0] + p.rsplit("If you are not sure", 1)[1] for p in prompts.values()}
    assert len(framing) == 1
    text = next(iter(prompts.values()))
    assert "ANSWER: UNANSWERABLE" in text and "<claims>" in text and "verbatim defining line" in text


def test_v2_parser_reads_the_answer_line_and_claims():
    text = "I read the def line.\nANSWER: `'igoore'`\n<claims>" + '[{"text": "`m.f` parameter `p` defaults to `\'igoore\'`", "citations": [{"ref": "@3", "quote": "def f(p=\'igoore\'):"}]}]' + "</claims>"
    parsed = parse_pilot_answer_v2(text)
    assert (parsed.answer, parsed.abstain) == ("'igoore'", False)
    assert parsed.claims[0].citations[0].ref == "@3" and parsed.claims[0].citations[0].quote == "def f(p='igoore'):"
    assert parse_pilot_answer_v2("ANSWER: ValueError, KeyError\n<claims>[]</claims>").answer == "ValueError, KeyError"
    for abstain in ("ANSWER: UNANSWERABLE", "ANSWER: unanswerable\n<claims>[]</claims>", 'ANSWER: UNANSWERABLE\n<claims>[{"text": "x", "citations": []}]</claims>'):
        parsed = parse_pilot_answer_v2(abstain)
        assert parsed.abstain and parsed.answer is None and parsed.claims == ()


@pytest.mark.parametrize(
    "bad",
    [
        "no answer line",
        "ANSWER: 3\nANSWER: 4\n<claims>[]</claims>",
        "ANSWER: 3",  # an answer needs its claims block
        "ANSWER:   \n<claims>[]</claims>",
        "<claims>[]</claims>\nANSWER: 3",
        "ANSWER: 3\n<claims>[{]</claims>",
        "ANSWER: 3\n<claims>{}</claims>",
        "ANSWER: 3\n<claims>[]</claims><claims>[]</claims>",
        'ANSWER: 3\n<claims>[{"text": "x"}]</claims>',
        'ANSWER: 3\n<claims>[{"text": "x", "citations": [{"ref": 3}]}]</claims>',
    ],
)
def test_v2_parser_is_strict(bad):
    with pytest.raises(AnswerFormatError):
        parse_pilot_answer_v2(bad)


def test_formatters_round_trip_through_their_parsers():
    claims = [{"text": "`m.f` raises `ValueError` directly", "citations": [{"ref": "Lib/m.py:1-9", "quote": "raise ValueError('x')"}]}]
    for version, prompt in PROMPTS.items():
        assert prompt.parse(prompt.answer_text(None, [], True)).abstain
        parsed = prompt.parse(prompt.answer_text(["ValueError"], claims))
        assert not parsed.abstain and parsed.claims[0].text == claims[0]["text"]
    assert parse_pilot_answer_v2(pilot_answer_text_v2(["ValueError", "KeyError"], claims)).answer == "ValueError, KeyError"
