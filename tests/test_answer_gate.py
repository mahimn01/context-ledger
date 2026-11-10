"""BUILD-V3 item 1: the answer gate. The target parsed from the question text, positive controls per
template (a correct, correctly cited answer passes), the failure reasons, staleness of notes and of
code spans, "any failing claim blocks", property tests over targets and their one-character edits,
the client's fix messages, gate_blocked and the forced condition's fallback, and a sentinel test that
the gate sees the question text and never a gold or metadata field."""

from __future__ import annotations

import json

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from context_ledger.anthropic.client import FORCED_MESSAGE, LedgerClient
from context_ledger.anthropic.fake import FakeAnthropicServer
from context_ledger.eval.prompts import parse_pilot_answer_v2, pilot_answer_text_v2
from context_ledger.invariants import check
from context_ledger.model import Citation
from context_ledger.session import GateError, RefetchObligationError, UngroundedClaimError
from context_ledger.verify.gate import Target, answer_gate, asked_target
from context_ledger.verify.tier1 import IDENTIFIER
from m2util import reply, text
from test_bounded_floor import session as l_session

MODULES = {
    "Lib/spool.py": (
        "SPOOL_LIMIT = 127\n"
        "SPOOL_MODE = 'strict'\n\n\n"
        'def drain_spool_step0(value, retries=3, label="spool-0", scale=1.5):\n'
        "    if value is None:\n"
        '        raise ValueError("spool step 0 needs a value")\n'
        "    if retries < 0:\n"
        '        raise KeyError("retries must not be negative")\n'
        "    return value\n"
    ),
    "Lib/bz2.py": "from _compression import BUFFER_SIZE\n\n\nclass BZ2File:\n    def readline(self, size=BUFFER_SIZE):\n        return size\n",
    "Lib/_compression.py": "BUFFER_SIZE = 8192\nDEFAULT_SIZE = BUFFER_SIZE\n",
    "Lib/garden/tools.py": "def dahlia_mint(bed, depth=2):\n    return bed\n",
    "Lib/poplib.py": (
        "class error_proto(Exception):\n    pass\n\n\nclass POP3:\n    def apop(self, user, secret):\n        if not secret:\n"
        "            raise error_proto('-ERR APOP not supported by server')\n        return user\n"
    ),
    "Lib/re/_constants.py": "MAGIC = 57585100\nSRE_FLAG_DOTALL = 16\n",
    "Lib/xml/dom/__init__.py": "INVALID_STATE_ERR              = 45\nNOT_SUPPORTED_ERR              = 6\n",
}
NOTES = [
    "Deployment note: in this deployment, the retry limit of service orion-relay is 48213.",
    "Deployment note: in this deployment, the retry limit of service orion-relayd is 51234.",
    "Update: the retry limit of service orion-relay is now 70531.",
    "Deployment note: in this deployment, the retry delay of service orion-relay is 6042.",
]


def world():
    """An L-like session holding every module's symbols and the notes as verbatim spans."""
    s = l_session(200_000, "v2c")
    spans = {}
    for path, body in MODULES.items():
        source = s.register_source(path, body, media_type="text/x-python")
        for span in s.register_python_spans(source, path):
            s.append_span(span.span_id, role="user")
            spans[(path, span.anchor.qualname)] = span.span_id
    for pos, note in enumerate(NOTES):
        source = s.register_source(f"note:{pos}", note)
        span = s.register_text_span(source, 0, len(note))
        s.append_span(span.span_id, role="user")
        spans[f"note:{pos}"] = span.span_id
    return s, spans


def ask(s, question: str, claims: list[tuple[str, list[tuple[str, str]]]], turn: str = "t"):
    """Ask `question`, call, record the claims (text, [(span id, quote)]); return them."""
    s.append_raw("user", question)
    s.call()
    out = []
    for claim_text, cites in claims:
        claim, _ = s.claim(claim_text, [Citation(s.state.aliases.alias_of(ref) or ref, quote) for ref, quote in cites], turn_id=turn)
        out.append(claim)
    return out


def gate(s, question, answer, claims):
    return answer_gate(question, answer, claims, s.state, s.blobs)


# 1a. The asked target, from the question text


def test_targets_are_parsed_from_the_question_text_alone():
    assert asked_target("What is the value of the module constant `stat.ST_ATIME`?") == Target("code", "stat.ST_ATIME")
    default = asked_target("What is the default value of parameter `size` of `bz2.BZ2File.readline`?")
    assert (default.name, default.secondary, default.question, default.label()) == ("bz2.BZ2File.readline", ("size",), "value", "bz2.BZ2File.readline parameter size")
    hop = asked_target("What value does parameter `p` of `mod.f` default to? Resolve any named constant to its literal value.")
    assert (hop.name, hop.secondary, hop.question) == ("mod.f", ("p",), "multihop")
    assert asked_target("Which exception classes does `poplib.POP3.apop` raise directly (in its own body)?").question == "raises"
    assert asked_target("Which module defines the function `dahlia_mint`?") == Target("code", "dahlia_mint", question="locate")
    session = asked_target("What is the current shard key of service sadr-router?")
    assert (session.kind, session.name, session.attribute, session.label()) == ("session", "sadr-router", "shard key", "the shard key of service sadr-router")
    assert asked_target("What is the weather like?") is None
    assert asked_target("What about `not an identifier` and `3.5`?") is None


def test_no_target_passes_vacuously():
    s, spans = world()
    [claim] = ask(s, "What is the weather like?", [("It is the limit.", [(spans[("Lib/spool.py", "SPOOL_LIMIT")], "SPOOL_LIMIT = 127")])])
    verdict = gate(s, "What is the weather like?", "127", [claim])
    assert verdict.passed and verdict.reason == "no_target" and verdict.target is None


# Positive controls: a correct, correctly cited answer passes, one per template


POSITIVE = [
    ("constant", "What is the value of the module constant `spool.SPOOL_LIMIT`?", "127",
     [("`spool.SPOOL_LIMIT` is `127`", [(("Lib/spool.py", "SPOOL_LIMIT"), "SPOOL_LIMIT = 127")])]),
    ("constant (string)", "What is the value of the module constant `spool.SPOOL_MODE`?", "'strict'",
     [("`spool.SPOOL_MODE` is `'strict'`", [(("Lib/spool.py", "SPOOL_MODE"), "SPOOL_MODE = 'strict'")])]),
    ("default", "What is the default value of parameter `retries` of `spool.drain_spool_step0`?", "3",
     [("`spool.drain_spool_step0` parameter `retries` defaults to `3`",
       [(("Lib/spool.py", "drain_spool_step0"), 'def drain_spool_step0(value, retries=3, label="spool-0", scale=1.5):')])]),
    ("default_ref (multi-hop)", "What value does parameter `size` of `bz2.BZ2File.readline` default to? Resolve any named constant to its literal value.", "8192",
     [("`bz2.BZ2File.readline` parameter `size` defaults to `BUFFER_SIZE`, which is `8192`",
       [(("Lib/bz2.py", "BZ2File.readline"), "def readline(self, size=BUFFER_SIZE):"), (("Lib/_compression.py", "BUFFER_SIZE"), "BUFFER_SIZE = 8192")])]),
    ("constant_ref (multi-hop)", "What is the value of the module constant `_compression.DEFAULT_SIZE`? Resolve any named constant to its literal value.", "8192",
     [("`_compression.DEFAULT_SIZE` is `BUFFER_SIZE`, which is `8192`",
       [(("Lib/_compression.py", "DEFAULT_SIZE"), "DEFAULT_SIZE = BUFFER_SIZE"), (("Lib/_compression.py", "BUFFER_SIZE"), "BUFFER_SIZE = 8192")])]),
    ("locate", "Which module defines the function `dahlia_mint`?", "garden.tools",
     [("`dahlia_mint` is defined in `garden.tools`", [(("Lib/garden/tools.py", "dahlia_mint"), "def dahlia_mint(bed, depth=2):")])]),
    ("raises", "Which exception classes does `poplib.POP3.apop` raise directly (in its own body)?", "error_proto",
     [("`poplib.POP3.apop` raises `error_proto` directly", [(("Lib/poplib.py", "POP3.apop"), "raise error_proto('-ERR APOP not supported by server')")])]),
    ("raises (two, two quotes)", "Which exception classes does `spool.drain_spool_step0` raise directly (in its own body)?", "KeyError, ValueError",
     [("`spool.drain_spool_step0` raises `ValueError` and `KeyError` directly",
       [(("Lib/spool.py", "drain_spool_step0"), 'raise ValueError("spool step 0 needs a value")'),
        (("Lib/spool.py", "drain_spool_step0"), 'raise KeyError("retries must not be negative")')])]),
    ("session", "What is the current retry delay of service orion-relay?", "6042",
     [("the retry delay of service `orion-relay` is `6042`", [("note:3", NOTES[3])])]),
    ("session, superseded, from the latest note", "What is the current retry limit of service orion-relay?", "70531",
     [("the retry limit of service `orion-relay` is now `70531`", [("note:2", NOTES[2])])]),
]


@pytest.mark.parametrize(("name", "question", "answer", "claims"), POSITIVE, ids=[p[0] for p in POSITIVE])
def test_a_correct_correctly_cited_answer_passes(name, question, answer, claims):
    s, spans = world()
    made = ask(s, question, [(t, [(spans[k], q) for k, q in cites]) for t, cites in claims])
    assert all(c.status == "grounded" for c in made)
    verdict = gate(s, question, answer, made)
    assert verdict.passed and verdict.reason == "anchored" and verdict.target == asked_target(question).label()
    event = s.emit_answer("t", [c.claim_id for c in made], question=question, answer=answer)
    assert event.payload["gate"] == verdict.to_dict() and check(s.events, s.blobs) == []


def test_the_module_prefix_witnesses_a_text_span_of_the_file():
    s, spans = world()
    source = next(src for src in s.state.sources.values() if src.uri == "Lib/re/_constants.py")
    body = MODULES["Lib/re/_constants.py"]
    span = s.register_text_span(source, body.index("SRE_FLAG_DOTALL"), len(body) - 1)
    s.append_span(span.span_id, role="user")
    question = "What is the value of the module constant `re._constants.SRE_FLAG_DOTALL`?"
    [claim] = ask(s, question, [("`re._constants.SRE_FLAG_DOTALL` is `16`", [(span.span_id, "SRE_FLAG_DOTALL = 16")])])
    assert gate(s, question, "16", [claim]).passed
    other = "What is the value of the module constant `re._parser.SRE_FLAG_DOTALL`?"  # another module's constant of that name
    assert gate(s, other, "16", [claim]).reason == "target_not_witnessed"


# The failure reasons


def test_value_not_with_target_and_the_hop_rule():
    s, spans = world()
    question = "What is the value of the module constant `spool.SPOOL_LIMIT`?"
    [claim] = ask(s, question, [("`spool.SPOOL_LIMIT` is `127`", [(spans[("Lib/spool.py", "SPOOL_LIMIT")], "SPOOL_LIMIT = 127")])])
    verdict = gate(s, question, "128", [claim])
    assert (verdict.passed, verdict.reason, verdict.detail) == (False, "value_not_with_target", ("128",))
    hop = "What value does parameter `size` of `bz2.BZ2File.readline` default to? Resolve any named constant to its literal value."
    [first] = ask(s, hop, [("`bz2.BZ2File.readline` parameter `size` defaults to `BUFFER_SIZE`", [(spans[("Lib/bz2.py", "BZ2File.readline")], "def readline(self, size=BUFFER_SIZE):")])], "h")
    assert gate(s, hop, "8192", [first]).reason == "value_not_with_target"  # no second hop cited
    [magic] = ask(s, hop, [("`re._constants.MAGIC` is `57585100`", [(spans[("Lib/re/_constants.py", "MAGIC")], "MAGIC = 57585100")])], "h2")
    assert gate(s, hop, "57585100", [first, magic]).reason == "value_not_with_target"  # MAGIC is not the default's name


def test_a_missing_parameter_and_a_wrong_module_are_refused():
    s, spans = world()
    question = "What is the default value of parameter `depth` of `spool.drain_spool_step0`?"
    [claim] = ask(s, question, [("`spool.drain_spool_step0` parameter `retries` defaults to `3`",
                                 [(spans[("Lib/spool.py", "drain_spool_step0")], 'def drain_spool_step0(value, retries=3, label="spool-0", scale=1.5):')])])  # fmt: skip
    verdict = gate(s, question, "3", [claim])
    assert (verdict.reason, verdict.detail) == ("target_not_witnessed", ("depth",))
    locate = "Which module defines the function `dahlia_mint`?"
    [where] = ask(s, locate, [("`dahlia_mint` is defined in `garden.tools`", [(spans[("Lib/garden/tools.py", "dahlia_mint")], "def dahlia_mint(bed, depth=2):")])], "l")
    assert gate(s, locate, "garden.beds", [where]).reason == "value_not_with_target"
    assert gate(s, locate, "Lib/garden/tools.py", [where]).passed


def test_look_alikes_never_witness():
    s, spans = world()
    invalid = (spans[("Lib/xml/dom/__init__.py", "INVALID_STATE_ERR")], "INVALID_STATE_ERR              = 45")
    [claim] = ask(s, "q", [("`xml.dom.INVALID_STATE_ERR` has the value 45", [invalid])])
    assert gate(s, "What is the value of the module constant `xml.dom.INALID_STATE_ERR`?", "45", [claim]).reason == "target_not_witnessed"
    assert gate(s, "What is the value of the module constant `xml.dom.INVALID_STATE_ERR`?", "45", [claim]).passed
    assert gate(s, "What is the value of the module constant `stat.INVALID_STATE_ERR`?", "45", [claim]).reason == "target_not_witnessed"
    [relayd] = ask(s, "q2", [("the retry limit of service `orion-relayd` is `51234`", [(spans["note:1"], NOTES[1])])], "t2")
    assert gate(s, "What is the current retry limit of service orion-relay?", "51234", [relayd]).reason == "target_not_witnessed"
    assert gate(s, "What is the current retry limit of service orion-relayd?", "51234", [relayd]).passed
    assert gate(s, "What is the current retry budget of service orion-relayd?", "51234", [relayd]).reason == "target_not_witnessed"


# 1d. Staleness


def test_a_cited_note_updated_later_is_stale_and_names_the_update():
    s, spans = world()
    question = "What is the current retry limit of service orion-relay?"
    [old] = ask(s, question, [("the retry limit of service `orion-relay` is `48213`", [(spans["note:0"], NOTES[0])])])
    verdict = gate(s, question, "48213", [old])
    assert (verdict.reason, verdict.stale) == ("stale_superseded", ((spans["note:0"], spans["note:2"]),))
    assert gate(s, "What is the current retry delay of service orion-relay?", "6042", ask(s, "q", [("x", [(spans["note:3"], NOTES[3])])], "t3")).passed
    # the update of orion-relay does not make orion-relayd's note stale (another service token)
    assert gate(s, "What is the current retry limit of service orion-relayd?", "51234", ask(s, "q", [("y", [(spans["note:1"], NOTES[1])])], "t4")).passed


def test_a_code_span_whose_source_changed_is_stale_by_kappa_and_must_be_refetched():
    s, spans = world()
    limit = spans[("Lib/spool.py", "SPOOL_LIMIT")]
    s.update_source(s.state.spans[limit].source_id, MODULES["Lib/spool.py"].replace("127", "131"))
    [claim] = ask(s, "What is the value of the module constant `spool.SPOOL_LIMIT`?", [("`spool.SPOOL_LIMIT` is `127`", [(limit, "SPOOL_LIMIT = 127")])])
    assert claim.status == "needs_refetch" and claim.verdict["reason"] == "stale" and claim.verdict["obligations"] == [limit]
    with pytest.raises(RefetchObligationError):
        s.emit_answer("t", [claim.claim_id], question="What is the value of the module constant `spool.SPOOL_LIMIT`?", answer="127")


# 1c. Any failing claim blocks; only a passing superseding claim clears it


def test_a_withdrawn_failure_blocks_but_a_passing_superseding_claim_clears():
    s, spans = world()
    question = "What is the value of the module constant `spool.SPOOL_LIMIT`?"
    good = (spans[("Lib/spool.py", "SPOOL_LIMIT")], "SPOOL_LIMIT = 127")
    bad, other = ask(s, question, [("`spool.SPOOL_LIMIT` is `999`", [good]), ("`spool.SPOOL_LIMIT` is `998`", [good])])
    assert bad.status == other.status == "ungrounded"
    fixed, _ = s.claim("`spool.SPOOL_LIMIT` is `127`", [Citation(s.state.aliases.alias_of(good[0]), good[1])], turn_id="t", supersedes=bad.claim_id)
    s.abstain("withdrawn", turn_id="t", supersedes=other.claim_id, reason="withdrawn")  # v2c's client withdrew left-over claims
    with pytest.raises(UngroundedClaimError) as blocked:
        s.emit_answer("t", [fixed.claim_id], question=question, answer="127")
    assert blocked.value.claim_ids == [other.claim_id] and [c.claim_id for c in s.cleared_failures("t")] == [bad.claim_id]
    s.emit_answer("t", [], abstained=True)  # an abstention still may leave


# Property tests

_MODULE = st.from_regex(r"[a-z][a-z0-9_]{0,6}", fullmatch=True)
_CONSTANT = st.from_regex(r"[A-Z][A-Z0-9_]{1,10}", fullmatch=True)


def _one_module_world(module: str, name: str, value: int):
    s = l_session(100_000, "v2c")
    path = f"Lib/{module}.py"
    source = s.register_source(path, f"{name} = {value}\n", media_type="text/x-python")
    [span] = s.register_python_spans(source, path)
    s.append_span(span.span_id, role="user")
    s.append_raw("user", "q")
    s.call()
    claim, _ = s.claim("the quoted line", [Citation(s.state.aliases.alias_of(span.span_id), f"{name} = {value}")], turn_id="t")
    return s, claim


@settings(max_examples=60, deadline=None)
@given(module=_MODULE, name=_CONSTANT, other=st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,10}", fullmatch=True), asked_module=_MODULE,
       value=st.integers(0, 10**6))  # fmt: skip
def test_the_gate_never_passes_a_target_absent_as_a_whole_token_or_an_anchored_run(module, name, other, asked_module, value):
    assume(other != name and not (asked_module == module and other == name))
    s, claim = _one_module_world(module, name, value)
    target = f"{asked_module}.{other}"
    assume(target not in (f"{module}.{name}", name))
    verdict = gate(s, f"What is the value of the module constant `{target}`?", str(value), [claim])
    assert not verdict.passed and verdict.reason == "target_not_witnessed"


def _edits(word: str, alphabet: str, data) -> str:
    """One character of `word` substituted, deleted, or inserted (from `alphabet`)."""
    kind = data.draw(st.sampled_from(["sub", "del", "ins"]))
    i = data.draw(st.integers(0, len(word) - (kind != "ins")))
    if kind == "del":
        return word[:i] + word[i + 1 :]
    c = data.draw(st.sampled_from(alphabet))
    return word[:i] + c + word[i + (kind == "sub") :]


@settings(max_examples=150, deadline=None)
@given(module=_MODULE, name=_CONSTANT, value=st.integers(0, 10**6), data=st.data())
def test_one_character_edits_of_a_code_target_never_pass(module, name, value, data):
    s, claim = _one_module_world(module, name, value)
    target = f"{module}.{name}"
    assert gate(s, f"What is the value of the module constant `{target}`?", str(value), [claim]).passed
    edited = _edits(target, "abcxyzABCXYZ019_.", data)
    assume(edited != target and IDENTIFIER.fullmatch(edited))  # an edit that leaves no identifier leaves no target (no_target)
    assert not gate(s, f"What is the value of the module constant `{edited}`?", str(value), [claim]).passed


@settings(max_examples=150, deadline=None)
@given(data=st.data())
def test_one_character_edits_of_a_session_target_never_pass(data):
    s, spans = world()
    [claim] = ask(s, "q", [("the retry limit of service `orion-relay` is now `70531`", [(spans["note:2"], NOTES[2])])])
    service, attribute = "orion-relay", "retry limit"
    if data.draw(st.booleans()):
        service = _edits(service, "abdeoryz019-_", data)
    else:
        attribute = _edits(attribute, "abcelmrtyz", data)
    assume((service, attribute) != ("orion-relay", "retry limit") and service and attribute and attribute[0] != " ")
    question = f"What is the current {attribute} of service {service}?"
    assume(asked_target(question) is not None and asked_target(question).kind == "session")
    assert not gate(s, question, "70531", [claim]).passed


# The client: fix messages, gate_blocked, and the forced condition


def _answer(value: str, alias: str, quote: str, claim_text: str = "The module constant `NOT_SUPPORTED_ERR` in `xml.dom` is defined with the value 6.") -> dict:
    return reply(text(pilot_answer_text_v2(value, [{"text": claim_text, "citations": [{"ref": alias, "quote": quote}]}])))


UNANSWERABLE = reply(text(pilot_answer_text_v2(None, [], abstain=True)))
ST_ATIME = "What is the value of the module constant `stat.ST_ATIME`?"


def _turn(s, script, *, forced=False, max_rounds=6):
    server = FakeAnthropicServer(script, pricing=s.pricing)
    wrapper = LedgerClient(s, server.as_fake_client(), model="claude-haiku-5-5", max_tokens=512, parse=parse_pilot_answer_v2, gate=True, forced=forced)
    return wrapper.run_turn(ST_ATIME, turn_id="t", max_rounds=max_rounds), server


def _last_user(request) -> str:
    content = request["messages"][-1]["content"]
    return content if isinstance(content, str) else [b["text"] for b in content if b.get("type") == "text"][-1]


def test_a_gate_refusal_names_the_asked_target_then_the_model_may_abstain():
    s, spans = world()
    alias = s.state.aliases.alias_of(spans[("Lib/xml/dom/__init__.py", "NOT_SUPPORTED_ERR")])
    result, server = _turn(s, [_answer("6", alias, "NOT_SUPPORTED_ERR              = 6"), UNANSWERABLE])
    message = _last_user(server.requests[1])
    assert message.startswith("Verification: the question asks about `stat.ST_ATIME`") and "about something else" in message
    assert "ctx_search" in message and "ctx_refetch" in message and message.endswith("; or abstain.")
    assert (result.reason, result.abstained, [g["reason"] for g in result.gates]) == ("model_abstained", True, ["target_not_witnessed"])
    assert check(s.events, s.blobs) == []


def test_the_round_bound_ends_a_gate_blocked_turn_with_gate_blocked():
    s, spans = world()
    alias = s.state.aliases.alias_of(spans[("Lib/xml/dom/__init__.py", "NOT_SUPPORTED_ERR")])
    result, _ = _turn(s, [_answer("6", alias, "NOT_SUPPORTED_ERR              = 6")] * 3, max_rounds=3)
    assert (result.reason, result.abstained, result.bound_hit, result.fallback) == ("gate_blocked", True, True, None)
    assert [g["reason"] for g in result.gates] == ["target_not_witnessed"] * 3


def test_forced_asks_once_for_a_best_guess_then_falls_back_to_the_last_blocked_candidate():
    s, spans = world()
    alias = s.state.aliases.alias_of(spans[("Lib/xml/dom/__init__.py", "NOT_SUPPORTED_ERR")])
    wrong = _answer("6", alias, "NOT_SUPPORTED_ERR              = 6")
    result, server = _turn(s, [UNANSWERABLE, wrong, wrong], forced=True, max_rounds=3)
    assert _last_user(server.requests[1]) == FORCED_MESSAGE
    assert "abstain" not in _last_user(server.requests[2])  # the forced condition never offers to abstain
    assert (result.reason, result.abstained, result.bound_hit, result.forced_violation) == ("forced_blocked", True, True, False)
    parsed, claims = result.fallback
    assert parsed.answer == "6" and claims[0].status == "grounded"
    assert any(e.type == "answer_emitted" and e.payload["abstained"] for e in s.events)  # the ledger records an abstention
    assert check(s.events, s.blobs) == []


def test_forced_with_no_candidate_is_a_violation():
    s, _ = world()
    result, server = _turn(s, [UNANSWERABLE, UNANSWERABLE], forced=True)
    assert (result.reason, result.abstained, result.fallback, result.forced_violation) == ("model_abstained", True, None, True)
    assert len(server.requests) == 2


def test_forced_keeps_fixing_a_failed_claim_until_the_bound():
    s, spans = world()
    alias = s.state.aliases.alias_of(spans[("Lib/xml/dom/__init__.py", "NOT_SUPPORTED_ERR")])
    bad = _answer("6", alias, "NOT_SUPPORTED_ERR              = 6", "The value is 7.")  # Tier 1: 7 is in no quote
    allowed, _ = _turn(s, [bad] * 4, max_rounds=4)
    assert allowed.reason == "ungrounded" and allowed.fallback is None  # one fix, then abstain
    s2, spans2 = world()
    forced, server = _turn(s2, [bad] * 4, forced=True, max_rounds=4)
    assert forced.reason == "forced_blocked" and forced.fallback[0].answer == "6" and len(server.requests) == 4


# The gate sees the question text only


SENTINEL = "SENTINEL-GOLD-7f3a"


def test_the_gate_never_sees_a_gold_or_metadata_field(tmp_path, monkeypatch):
    import context_ledger.session as session_module
    from context_ledger.eval.corpus import build_corpus
    from context_ledger.eval.questions import build_questions
    from context_ledger.eval.runner import load_run, offline_client, run_pilot
    from context_ledger.pricing import PricingTable
    from context_ledger.store import Ledger, SqliteBlobStore
    from context_ledger.tokens import BYTES4_V0
    from pilot_fixture import write_fixture_tarball

    tar, sha = write_fixture_tarball(tmp_path)
    manifest = build_corpus(seed=3, tarball=tar, sha256=sha, window=300, rhos=(10,), n_gold=4, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache",
                            progress=lambda m: None)  # fmt: skip
    rows = build_questions(manifest, seed=9, episodes=1, runs_dir=tmp_path / "runs")
    hidden = {"answer": SENTINEL, "upstream_answer": SENTINEL, "subject": SENTINEL, "gold_path": SENTINEL, "gold_lines": [SENTINEL],
              "gold_paths": [SENTINEL], "gold_lines_all": [[SENTINEL]], "attribute": SENTINEL, "confusers": {"10": SENTINEL},
              "superseded": SENTINEL, "notes_of": [SENTINEL], "template": SENTINEL, "near": SENTINEL}  # fmt: skip
    rows = [{**r, **hidden} for r in rows]
    texts = {r["text"] for r in rows}
    seen = []
    real = session_module.answer_gate

    def spy(question_text, answer, claims, state, blobs):
        seen.append(question_text)
        assert question_text in texts and SENTINEL not in json.dumps([question_text, str(answer), [c.to_dict() for c in claims]])
        return real(question_text, answer, claims, state, blobs)

    monkeypatch.setattr(session_module, "answer_gate", spy)
    table = PricingTable.verified_default()
    client, guard, server = offline_client(tmp_path / "run", run_id="s", pricing=table, budget_usd="0.75")
    try:
        run_pilot(manifest=manifest, rows=rows, arms=["L"], rho=10, model="claude-haiku-5-5", window=3000, episodes=1, per_episode=8, smoke=False,
                  run_dir=tmp_path / "run", run_id="s", pricing=table, estimator=BYTES4_V0, client=client, live=False, budget_usd="0.75")  # fmt: skip
    finally:
        guard.close()
    assert seen  # the gate ran
    assert SENTINEL not in json.dumps(server.requests)  # nor did the model see any of it
    db = tmp_path / "run" / "L" / "episode-01.sqlite"
    events, blobs = Ledger(db).events(), SqliteBlobStore(db)
    assert SENTINEL not in json.dumps([e.payload for e in events])
    results = load_run(tmp_path / "run")[1]
    assert all(r["gate_target"] is not None for r in results)  # every v1 question has a target
