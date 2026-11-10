"""BUILD-V3 item 1e: each of L's 11 pilot-v2c errors, rebuilt offline and blocked by the answer gate.

Every case below was read once, read-only, from runs/pilot-v2c-haiku/rho-{10,1000}/L/episode-1{5,6}.sqlite
(opened with mode=ro&immutable=1) and frozen here: the question text, the answer L emitted, every claim of
the turn (text, status, verdict reason, which claim it superseded) with its citations and quotes, the full
text of every cited span, and the episode's notes in the order the ledger registered them. The tests never
read runs/. Each case is rebuilt as a small ledger with the same notes, spans, claims and quotes (a fold or
a refetch where the pilot had one), its claims are checked to verify exactly as they did in the pilot, and
then:

- with the gate (the question text passed to emit_answer, as L does since BUILD-V3) the answer is refused
  with GateError and the expected reason, recording nothing;
- without the gate (v2c's egress) the same answer is emitted, as it was in the pilot.

What let the three "failed claim, then grounded claim" cases out (v2s15q19 at both rho, v2s15q05, v2s15q06):
supersession. In each, a later claim that passed Tier 1 superseded the failed one (it had dropped the
misspelled name the fix message flagged, or cited the stale note instead of the fold); none was let out by
`abstained=True` skipping the ungrounded check (an abstaining answer never carries an answer value). A
superseding claim that passes still clears a failure; the gate is what now refuses those answers.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from context_ledger.fold import extractive_summary_v1
from context_ledger.invariants import check
from context_ledger.model import Citation
from context_ledger.session import GateError
from context_ledger.verify.gate import asked_target
from test_bounded_floor import session as l_session

EPISODE_NOTES = {15: {0: 'Deployment note: in this deployment, the queue size of service phecda-proxy is 34771.',
      1: 'Deployment note: in this deployment, the queue limit of service phecda-relayd is 68281.',
      2: 'Deployment note: in this deployment, the queue limit of service phecda-relay is 52107.',
      3: 'Update: the queue limit of service phecda-relayd is now 39852.',
      7: 'Deployment note: in this deployment, the queue depth of service phecda-relay is 94181.',
      15: 'Deployment note: in this deployment, the queue depth of service phecda-relayd is 29326.',
      18: 'Deployment note: in this deployment, the queue limit of service phecda-proxy is 17593.',
      21: 'Update: the queue limit of service phecda-proxy is now 26155.'},
 16: {0: 'Deployment note: in this deployment, the shard label of service sadr-gateway is hb51r.',
      3: 'Deployment note: in this deployment, the shard label of service sadr-router is jp78w.',
      5: 'Update: the shard label of service sadr-router is now qc46r.',
      6: 'Deployment note: in this deployment, the shard key of service sadr-gatewayd is vb76r.',
      17: 'Deployment note: in this deployment, the shard tag of service sadr-gatewayd is mn17b.'}}

CASES = [{'qid': 'v2s15q18',
  'rho': 10,
  'kind': 'cross-wired',
  'question': 'What is the value of the module constant `stat.ST_ATIME`?',
  'answer': '6',
  'claims': [{'text': 'The module constant `NOT_SUPPORTED_ERR` in `xml.dom` is defined with the value 6.',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': None,
              'citations': [{'path': 'Lib/xml/dom/__init__.py',
                             'qualname': 'NOT_SUPPORTED_ERR',
                             'span': 'NOT_SUPPORTED_ERR              = 6',
                             'quote': 'NOT_SUPPORTED_ERR              = 6'}]}],
  'expect': 'target_not_witnessed'},
 {'qid': 'v2s15q19',
  'rho': 10,
  'kind': 'confusable',
  'question': 'What is the value of the module constant `xml.dom.INALID_STATE_ERR`?',
  'answer': '45',
  'claims': [{'text': "`xml.dom.INVALID_STATE_ERR` (the correctly spelled name; the query's `INALID_STATE_ERR` does not exist) has the value 45",
              'status': 'ungrounded',
              'reason': 'identifier_not_anchored',
              'supersedes': None,
              'citations': [{'path': 'Lib/xml/dom/__init__.py',
                             'qualname': 'INVALID_STATE_ERR',
                             'span': 'INVALID_STATE_ERR              = 45',
                             'quote': 'INVALID_STATE_ERR              = 45'}]},
             {'text': '`xml.dom.INVALID_STATE_ERR` has the value 45',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': 0,
              'citations': [{'path': 'Lib/xml/dom/__init__.py',
                             'qualname': 'INVALID_STATE_ERR',
                             'span': 'INVALID_STATE_ERR              = 45',
                             'quote': 'INVALID_STATE_ERR              = 45'}]}],
  'expect': 'target_not_witnessed'},
 {'qid': 'v2s15q21',
  'rho': 10,
  'kind': 'cross-wired',
  'question': 'Which exception classes does `poplib.POP3.apop` raise directly (in its own body)?',
  'answer': '57585100',
  'claims': [{'text': 'abstained before a refetch reminder', 'status': 'abstain', 'reason': 'model_abstained', 'supersedes': None, 'citations': []},
             {'text': '`Lib/re/_constants.py` defines `MAGIC` as 57585100',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': None,
              'citations': [{'path': 'Lib/re/_constants.py', 'qualname': 'MAGIC', 'span': 'MAGIC = 57585100', 'quote': 'MAGIC = 57585100'}]}],
  'expect': 'target_not_witnessed'},
 {'qid': 'v2s16q08',
  'rho': 10,
  'kind': 'cross-wired',
  'question': 'What is the value of the module constant `token.ENDMARKER`?',
  'answer': '7',
  'claims': [{'text': '`bisect.onyx_larch` parameter `lo` defaults to `7`',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': None,
              'citations': [{'path': 'Lib/bisect.py',
                             'qualname': 'onyx_larch',
                             'span': 'def onyx_larch(a, x, lo=7, hi=None, *, key=None):\n'
                                     '    """Insert item x in list a, and keep it sorted assuming a is sorted.\n'
                                     '\n'
                                     '    If x is already in a, insert it to the left of the leftmost x.\n'
                                     '\n'
                                     '    Optional args lo (default 0) and hi (default len(a)) bound the\n'
                                     '    slice of a to be searched.\n'
                                     '\n'
                                     '    A custom key function can be supplied to customize the sort order.\n'
                                     '    """\n'
                                     '\n'
                                     '    if key is None:\n'
                                     '        lo = breeze_apron(a, x, lo, hi)\n'
                                     '    else:\n'
                                     '        lo = breeze_apron(a, key(x), lo, hi, key=key)\n'
                                     '    a.insert(lo, x)',
                             'quote': 'def onyx_larch(a, x, lo=7, hi=None, *, key=None):'}]}],
  'expect': 'target_not_witnessed'},
 {'qid': 'v2s16q14',
  'rho': 10,
  'kind': 'cross-wired',
  'question': 'What is the current shard key of service sadr-router?',
  'answer': '84',
  'claims': [{'text': 'The module constant `token.MINUS` is 84 in Lib/token.py',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': None,
              'citations': [{'path': 'Lib/token.py', 'qualname': 'MINUS', 'span': 'MINUS = 84', 'quote': 'MINUS = 84'}]}],
  'expect': 'target_not_witnessed'},
 {'qid': 'v2s16q15',
  'rho': 10,
  'kind': 'cross-wired',
  'question': 'What is the current shard label of service sadr-router?',
  'answer': '84',
  'claims': [{'text': 'The module constant `token.MINUS` is defined as 84 in Lib/token.py',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': None,
              'citations': [{'path': 'Lib/token.py', 'qualname': 'MINUS', 'span': 'MINUS = 84', 'quote': 'MINUS = 84'}]}],
  'expect': 'target_not_witnessed'},
 {'qid': 'v2s16q17',
  'rho': 10,
  'kind': 'cross-wired',
  'question': 'What is the value of the module constant `xml.dom.SYNTX_ERR`?',
  'answer': "'mn17b'",
  'claims': [{'text': 'In this deployment, the shard tag of service sadr-gatewayd is mn17b, per the most recent deployment note.',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': None,
              'citations': [{'note': 'note:17', 'quote': 'Deployment note: in this deployment, the shard tag of service sadr-gatewayd is mn17b.'}]}],
  'expect': 'target_not_witnessed'},
 {'qid': 'v2s15q05',
  'rho': 1000,
  'kind': 'stale',
  'question': 'What is the current queue limit of service phecda-relayd?',
  'answer': '68281',
  'claims': [{'text': "The current queue limit of service phecda-relayd is 39852, per the user's latest update.",
              'status': 'needs_refetch',
              'reason': 'summary_only',
              'supersedes': None,
              'citations': [{'fold': True, 'quote': 'Update: the queue limit of service phecda-relayd is now 39852.'}]},
             {'text': "The current queue limit of service phecda-relayd is 39852, per the user's latest update; this supersedes the earlier "
                      'deployment note value of 68281.',
              'status': 'ungrounded',
              'reason': 'value_not_covered',
              'supersedes': 0,
              'citations': [{'fold': True, 'quote': 'Update: the queue limit of service phecda-relayd is now 39852.'},
                            {'note': 'note:1', 'quote': 'Deployment note: in this deployment, the queue limit of service phecda-relayd is 68281.'}]},
             {'text': 'The queue limit of service phecda-relayd is 68281 in this deployment',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': 1,
              'citations': [{'note': 'note:1', 'quote': 'Deployment note: in this deployment, the queue limit of service phecda-relayd is 68281.'}]}],
  'expect': 'stale_superseded',
  'fold': 'note:3'},
 {'qid': 'v2s15q06',
  'rho': 1000,
  'kind': 'stale',
  'question': 'What is the current queue limit of service phecda-relay?',
  'answer': '68281',
  'claims': [{'text': 'The current queue limit of service phecda-relayd is 68281 in this deployment.',
              'status': 'ungrounded',
              'reason': 'unknown_ref',
              'supersedes': None,
              'citations': [{'ref': 'note:1#0-87',
                             'quote': 'Deployment note: in this deployment, the queue limit of service phecda-relayd is 68281.'}]},
             {'text': 'The current queue limit of service phecda-relayd is 68281 in this deployment.',
              'status': 'needs_refetch',
              'reason': 'summary_only',
              'supersedes': 0,
              'citations': [{'note': 'note:1', 'quote': 'Deployment note: in this deployment, the queue limit of service phecda-relayd is 68281.'}]},
             {'text': 'The current queue limit of service phecda-relayd is 68281 in this deployment.',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': 1,
              'citations': [{'note': 'note:1', 'quote': 'Deployment note: in this deployment, the queue limit of service phecda-relayd is 68281.'}]}],
  'expect': 'stale_superseded',
  'fold': 'note:1',
  'refetch_before': {2: 'note:1'}},
 {'qid': 'v2s15q19',
  'rho': 1000,
  'kind': 'confusable',
  'question': 'What is the value of the module constant `xml.dom.INALID_STATE_ERR`?',
  'answer': '45',
  'claims': [{'text': 'The module constant `INVALID_STATE_ERR` in `xml.dom` (Lib/xml/dom/__init__.py) has the value 45. The spelling '
                      '`INALID_STATE_ERR` from the question does not appear in this definition.',
              'status': 'ungrounded',
              'reason': 'identifier_not_anchored',
              'supersedes': None,
              'citations': [{'path': 'Lib/xml/dom/__init__.py',
                             'qualname': 'INVALID_STATE_ERR',
                             'span': 'INVALID_STATE_ERR              = 45',
                             'quote': 'INVALID_STATE_ERR              = 45'}]},
             {'text': 'The module constant `INVALID_STATE_ERR` in `xml.dom` (Lib/xml/dom/__init__.py) has the value 45.',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': 0,
              'citations': [{'path': 'Lib/xml/dom/__init__.py',
                             'qualname': 'INVALID_STATE_ERR',
                             'span': 'INVALID_STATE_ERR              = 45',
                             'quote': 'INVALID_STATE_ERR              = 45'}]}],
  'expect': 'target_not_witnessed'},
 {'qid': 'v2s16q14',
  'rho': 1000,
  'kind': 'confusable',
  'question': 'What is the current shard key of service sadr-router?',
  'answer': "'qc46r'",
  'claims': [{'text': 'The current shard label of service sadr-router is qc46r, per the latest update note (superseding the earlier value jp78w).',
              'status': 'grounded',
              'reason': 'quotes_verified',
              'supersedes': None,
              'citations': [{'note': 'note:5', 'quote': 'Update: the shard label of service sadr-router is now qc46r.'}]}],
  'expect': 'target_not_witnessed'}]


def notes_of(case: dict) -> list[tuple[str, str]]:
    """The episode's notes the ledger held at the question: its own note and every earlier one."""
    episode, pos = int(case["qid"][3:5]), int(case["qid"][6:])
    return [(f"note:{p}", text) for p, text in EPISODE_NOTES[episode].items() if p <= pos]


def rebuild(case: dict, notes: list[tuple[str, str]] | None = None):
    """The case's ledger: its notes as verbatim user spans, the cited code spans, the pilot's fold,
    the question and a call, then its claims in order (a refetch and a call before a claim where
    the pilot refetched). Returns the session and the claims."""
    s = l_session(100_000, "v2c")
    note_spans = {}
    for uri, text in notes if notes is not None else notes_of(case):
        source = s.register_source(uri, text)
        span = s.register_text_span(source, 0, len(text))
        s.append_span(span.span_id, role="user")
        note_spans[uri] = span.span_id
    texts: dict[str, list[str]] = defaultdict(list)
    for claim in case["claims"]:
        for c in claim["citations"]:
            if "path" in c and c["span"] not in texts[c["path"]]:
                texts[c["path"]].append(c["span"])
    code = {}
    for path, parts in texts.items():
        source = s.register_source(path, "\n\n".join(parts) + "\n", media_type="text/x-python")
        spans = {sp.anchor.qualname: sp for sp in s.register_python_spans(source, path)}
        for claim in case["claims"]:
            for c in claim["citations"]:
                if c.get("path") == path and (path, c["qualname"]) not in code:
                    span = spans[c["qualname"]]
                    assert s.blobs.get(span.text_hash).decode() == c["span"]  # the same span text as in the pilot
                    s.append_span(span.span_id, role="user")
                    code[(path, c["qualname"])] = span.span_id
    fold_ref = None
    if case.get("fold"):
        victim = next(seg.segment_id for seg in s.state.segments if seg.ref == note_spans[case["fold"]])
        record = s.fold([victim], extractive_summary_v1(s.state, [victim], s.blobs, rules="v2c", digest=True))
        fold_ref = s.state.aliases.alias_of(f"fold:{record.fold_id}") or f"fold:{record.fold_id}"
    s.append_raw("user", case["question"])
    s.call()

    def ref(c: dict) -> str:
        if "path" in c:
            return s.state.aliases.alias_of(code[(c["path"], c["qualname"])])
        if "note" in c:
            return s.state.aliases.alias_of(note_spans[c["note"]])
        return fold_ref if c.get("fold") else c["ref"]

    claims = []
    for k, item in enumerate(case["claims"]):
        if k in case.get("refetch_before", {}):
            s.refetch(note_spans[case["refetch_before"][k]])
            s.call()
        if item["status"] == "abstain":
            claims.append(s.abstain(item["text"], turn_id=case["qid"], reason=item["reason"]))
            continue
        supersedes = claims[item["supersedes"]].claim_id if item["supersedes"] is not None else None
        claims.append(s.claim(item["text"], [Citation(ref(c), c["quote"]) for c in item["citations"]], turn_id=case["qid"], supersedes=supersedes)[0])
    return s, claims


def test_the_eleven_cases_are_the_pilots_errors():
    assert len(CASES) == 11
    assert sorted((c["kind"], c["qid"], c["rho"]) for c in CASES) == sorted(
        [("cross-wired", q, 10) for q in ("v2s15q18", "v2s15q21", "v2s16q08", "v2s16q14", "v2s16q15", "v2s16q17")]
        + [("confusable", "v2s15q19", 10), ("confusable", "v2s15q19", 1000), ("confusable", "v2s16q14", 1000)]
        + [("stale", "v2s15q05", 1000), ("stale", "v2s15q06", 1000)]
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['qid']}-rho{c['rho']}-{c['kind']}")
def test_the_gate_blocks_each_v2c_error(case):
    s, claims = rebuild(case)
    assert [(c.status, (c.verdict or {}).get("reason")) for c in claims] == [(i["status"], i["reason"]) for i in case["claims"]]
    final = claims[-1]
    before = len(s.events)
    with pytest.raises(GateError) as refused:
        s.emit_answer(case["qid"], [final.claim_id], question=case["question"], answer=case["answer"])
    verdict = refused.value.verdict
    assert not verdict.passed and verdict.reason == case["expect"] and len(s.events) == before
    assert verdict.target == asked_target(case["question"]).label()
    if case["expect"] == "stale_superseded":  # note:1 (68281) was updated by note:3 (39852)
        [(old, new)] = verdict.stale
        assert s.state.sources[s.state.spans[old].source_id].uri == "note:1" and s.state.sources[s.state.spans[new].source_id].uri == "note:3"
    else:
        assert verdict.detail == (asked_target(case["question"]).name,)
    event = s.emit_answer(case["qid"], [final.claim_id])  # v2c's egress let it out
    assert event.payload == {"turn_id": case["qid"], "claim_ids": [final.claim_id], "abstained": False}
    assert check(s.events, s.blobs) == []


@pytest.mark.parametrize("case", [c for c in CASES if any(i["status"] in ("ungrounded", "needs_refetch") for i in c["claims"])],
                         ids=lambda c: f"{c['qid']}-rho{c['rho']}")
def test_failed_claims_were_let_out_by_supersession_not_by_abstaining(case):
    s, claims = rebuild(case)
    failed = [c for c in claims if c.status in ("ungrounded", "needs_refetch")]
    assert [c.claim_id for c in s.cleared_failures(case["qid"])] == [c.claim_id for c in failed]  # each cleared by a passing claim
    assert s.unresolved_claims(case["qid"]) == []


def test_phecda_relayd_never_witnesses_phecda_relay_even_without_the_update():
    case = next(c for c in CASES if c["qid"] == "v2s15q06")
    notes = [n for n in notes_of(case) if n[0] != "note:3"]  # no later note: note:1 is not stale
    s, claims = rebuild({**case, "fold": None, "refetch_before": {}, "claims": [{**case["claims"][-1], "supersedes": None}]}, notes)
    with pytest.raises(GateError) as refused:
        s.emit_answer(case["qid"], [claims[-1].claim_id], question=case["question"], answer=case["answer"])
    assert refused.value.verdict.reason == "target_not_witnessed" and refused.value.verdict.target == "the queue limit of service phecda-relay"
