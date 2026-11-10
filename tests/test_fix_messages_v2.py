"""BUILD-V2 item 2: informative fix messages, tolerance for a ref written as a whole header,
and no egress for a non-abstaining answer without claims."""

from __future__ import annotations

from context_ledger.anthropic.client import LedgerClient
from context_ledger.anthropic.fake import FakeAnthropicServer
from context_ledger.invariants import check
from context_ledger.model import Citation
from context_ledger.prompts import SYSTEM_PROMPT, answer_text
from m2util import MODULE, MODULE_PATH, reply, text, v1_session


def setup():
    s = v1_session()
    s.append_raw("system", SYSTEM_PROMPT, pinned=True)
    src = s.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
    spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(src, MODULE_PATH)}
    for name in ["DEFAULT_TIMEOUT", "MAX_RETRIES", "backoff_delay"]:
        s.append_span(spans[name].span_id)
    return s, spans


def turn(s, script, question="What is the default timeout?"):
    server = FakeAnthropicServer(script, pricing=s.pricing)
    wrapper = LedgerClient(s, server.as_fake_client(), model="claude-haiku-5-5", max_tokens=512, tools=[])
    return wrapper.run_turn(question, turn_id="t1"), server


def user_texts(request) -> list[str]:
    out = []
    for m in request["messages"]:
        if m["role"] == "user":
            blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
            out += [b["text"] for b in blocks if b.get("type") == "text"]
    return out


def answer(*claims) -> str:
    return answer_text([{"text": t, "citations": [{"ref": r, "quote": q} for r, q in cites]} for t, *cites in claims])


def test_value_not_covered_message_names_the_offending_tokens():
    s, spans = setup()
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    bad = answer(("`DEFAULT_TIMEOUT` is 9 seconds, see `Lib/x.py`", (alias, "DEFAULT_TIMEOUT = 37.0")))
    good = answer(("`DEFAULT_TIMEOUT` is 37 seconds", (alias, "DEFAULT_TIMEOUT = 37.0")))
    result, server = turn(s, [reply(text(bad)), reply(text(good))])
    message = user_texts(server.requests[-1])[-1]
    assert message == (
        "Verification: claim 1 was not accepted: `9` and `Lib/x.py` do not appear in your quotes. Quote a verbatim line that "
        "contains them, or remove them from the claim text, and answer again; or abstain."
    )
    assert result.reason == "answered" and check(s.events, s.blobs) == []


def test_identifier_and_quote_messages():
    s, spans = setup()
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    for claim, expected in [
        (("`client.TIMEOUTS` is 37", (alias, "DEFAULT_TIMEOUT = 37.0")), "`client.TIMEOUTS` does not appear in your quotes, and no cited span is named so"),
        (("`DEFAULT_TIMEOUT` is 37", (alias, "DEFAULT_TIMEOUT = 38.0 seconds")), "the quote is not verbatim in the cited text (or shorter than 20 characters)"),
    ]:
        s2, _ = setup()
        _, server = turn(s2, [reply(text(answer(claim))), reply(text(answer_text([], abstain=True)))])
        assert expected in user_texts(server.requests[-1])[-1]


def test_unknown_ref_message_suggests_the_alias_from_the_header():
    s, spans = setup()
    bad = answer(("`DEFAULT_TIMEOUT` is 37", ("inventory/client.py:6-6", "DEFAULT_TIMEOUT = 37.0")))
    _, server = turn(s, [reply(text(bad)), reply(text(answer_text([], abstain=True)))])
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    assert user_texts(server.requests[-1])[-1] == (
        f"Verification: claim 1 was not accepted: ref `inventory/client.py:6-6` is not a ref shown to you; cite the @alias from the "
        f"header, e.g. {alias}. Answer again; or abstain."
    )


def test_a_ref_written_as_the_whole_header_resolves_to_its_alias():
    """The real M4 case: ref "@6 Lib/email/contentmanager.py#set_text_content" (the header text)."""
    s, spans = setup()
    s.call()
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    for ref in (f"{alias} inventory/client.py#DEFAULT_TIMEOUT", f"[{alias} inventory/client.py#DEFAULT_TIMEOUT]", f"[{alias}]"):
        claim, verdict = s.claim("`DEFAULT_TIMEOUT` is 37", [Citation(ref, "DEFAULT_TIMEOUT = 37.0")], turn_id=ref)
        assert claim.status == "grounded", (ref, verdict)
        assert (claim.citations[0].ref, claim.citations[0].alias) == (spans["DEFAULT_TIMEOUT"].span_id, alias)
    for ref in ("@ZZ9 inventory/client.py#DEFAULT_TIMEOUT", "[@ZZ9 x]", "see @0 for this", "inventory/client.py#DEFAULT_TIMEOUT"):
        claim, verdict = s.claim("`DEFAULT_TIMEOUT` is 37", [Citation(ref, "DEFAULT_TIMEOUT = 37.0")], turn_id=ref)
        assert (claim.status, verdict.reason) == ("ungrounded", "unknown_ref"), ref
        assert s.unknown_refs(claim) == [ref]
    assert check(s.events, s.blobs) == []


def test_header_ref_end_to_end_is_accepted_without_a_fix_round():
    s, spans = setup()
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    result, server = turn(s, [reply(text(answer((f"`DEFAULT_TIMEOUT` is 37", (f"{alias} inventory/client.py#DEFAULT_TIMEOUT", "DEFAULT_TIMEOUT = 37.0")))))])
    assert result.reason == "answered" and len(server.requests) == 1
    assert check(s.events, s.blobs) == []


def test_an_answer_without_claims_does_not_leave_and_is_asked_for_claims():
    s, spans = setup()
    alias = s.state.aliases.alias_of(spans["DEFAULT_TIMEOUT"].span_id)
    empty = answer_text([], abstain=False)
    good = answer(("`DEFAULT_TIMEOUT` is 37", (alias, "DEFAULT_TIMEOUT = 37.0")))
    result, server = turn(s, [reply(text(empty)), reply(text(good))])
    flagged = [c for c in result.claims if c.verdict["reason"] == "no_claims"]
    assert len(flagged) == 1 and flagged[0].status == "ungrounded"
    assert user_texts(server.requests[-1])[-1].startswith("Verification: your answer has no claims")
    assert "at least one claim" in user_texts(server.requests[-1])[-1] and "verbatim" in user_texts(server.requests[-1])[-1]
    assert result.reason == "answered" and len(result.emitted) == 1
    emitted = [e.payload for e in s.events if e.type == "answer_emitted"]
    assert emitted[-1]["claim_ids"] and not emitted[-1]["abstained"]
    assert check(s.events, s.blobs) == []


def test_two_answers_without_claims_abstain_after_one_fix_round():
    s, _ = setup()
    empty = answer_text([], abstain=False)
    result, server = turn(s, [reply(text(empty)), reply(text(empty))])
    assert (result.abstained, result.reason, len(server.requests)) == (True, "ungrounded", 2)
    assert check(s.events, s.blobs) == []
