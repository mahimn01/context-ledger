"""BUILD-V2 item 1: Tier 1 witnesses the source path of a passing citation and the module
prefix of its span, from the ledger's own registration. The positive cases are real claims
that L's egress check blocked in the M4 grid (runs/grid-*-m4, L ledgers), each of which the
grader would have scored correct."""

from __future__ import annotations

import json

import pytest

from context_ledger.fold import BackstopPolicy
from context_ledger.model import Citation, Claim
from context_ledger.session import Session
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.verify import Verdict, strip_source_paths, tier1
from context_ledger.window import replay

STORE_KV = 'STORE_KV = "REDLACE INTO Dict (key, value) VALUES (CAST(? AS BLOB), CAST(? AS BLOB))"'
SOURCES = {
    "Lib/token.py": "ENDMARKER = 0\nNAME = 1\nINDENT = 4\nDEDENT = 5\n",
    "Lib/re/_constants.py": "SRE_FLAG_UNICODE = 32\nSRE_FLAG_VERBOSE = 94\n",
    "Lib/stat.py": "ST_UID   = 4\nST_GID   = 7\n",
    "Lib/dbm/sqlite3.py": f'BUILD_TABLE = "CREATE TABLE"\n{STORE_KV}\n',
    "Lib/email/utils.py": "ST_GID = 7\n",
}


class World:
    """Sessions over SOURCES; `show` appends spans verbatim and makes a call."""

    def __init__(self) -> None:
        self.blobs = MemoryBlobStore()
        self.session = Session(Ledger(":memory:"), self.blobs, 100_000, BackstopPolicy(budget_tokens=100_000))
        self.sources = {p: self.session.register_source(p, t, media_type="text/x-python") for p, t in SOURCES.items()}
        self.spans = {}
        for path, source in self.sources.items():
            for span in self.session.register_python_spans(source, path):
                self.spans[(path, span.anchor.qualname)] = span

    def text_span(self, path: str, line: str):
        text = SOURCES[path]
        start = text.index(line)
        return self.session.register_text_span(self.sources[path], start, start + len(line))

    def show(self, *span_ids: str) -> World:
        for span_id in span_ids:
            self.session.append_span(span_id)
        self.session.call()
        return self

    def verdict(self, text: str, *cites: tuple[str, str]) -> Verdict:
        claim = Claim("c", self.session.state.calls[-1], text, tuple(Citation(s, q) for s, q in cites), "ungrounded")
        return tier1(claim, replay(self.session.events), self.blobs)


@pytest.fixture
def world():
    return World()


def _py(world: World, path: str, name: str) -> str:
    return world.spans[(path, name)].span_id


REAL = [
    # (claim text, source path, symbol, quote), as recorded in the M4 L ledgers
    ("`token.DEDENT` is defined in `Lib/token.py` with the value 5", "Lib/token.py", "DEDENT", "DEDENT = 5"),
    ("The module constant `SRE_FLAG_VERBOSE` in `Lib/re/_constants.py` has the value 94.", "Lib/re/_constants.py", "SRE_FLAG_VERBOSE", "SRE_FLAG_VERBOSE = 94"),
    ("`stat.ST_GID` is defined as 7 in Lib/stat.py", "Lib/stat.py", "ST_GID", "ST_GID   = 7"),
    ("The module constant `stat.ST_GID` is defined in `Lib/stat.py` with the value 7.", "Lib/stat.py", "ST_GID", "ST_GID   = 7"),
    (
        '`dbm.sqlite3.STORE_KV` is assigned the string "REDLACE INTO Dict (key, value) VALUES (CAST(? AS BLOB), CAST(? AS BLOB))" in Lib/dbm/sqlite3.py.',
        "Lib/dbm/sqlite3.py",
        "STORE_KV",
        STORE_KV,
    ),
]


@pytest.mark.parametrize("text,path,name,quote", REAL)
def test_the_real_m4_blocked_claims_now_pass(world, text, path, name, quote):
    span = _py(world, path, name)
    world.show(span)
    verdict = world.verdict(text, (span, quote))
    assert verdict.passed and verdict.reason == "quotes_verified", verdict


@pytest.mark.parametrize("text,path,name,quote", REAL)
def test_the_real_claims_pass_with_text_spans_too(world, text, path, name, quote):
    """Module-level lines read as text spans have no Python anchor; the source path and the
    module prefix still witness, because they come from the source the span was registered from."""
    span = world.text_span(path, quote).span_id
    world.show(span)
    assert world.verdict(text, (span, quote)).passed


def test_a_wrong_value_still_fails_with_the_value_as_detail(world):
    span = _py(world, "Lib/token.py", "DEDENT")
    world.show(span)
    verdict = world.verdict("`token.DEDENT` is defined in `Lib/token.py` with the value 6", (span, "DEDENT = 5"))
    assert (verdict.passed, verdict.reason, verdict.detail) == (False, "value_not_covered", ("6",))
    assert verdict.to_dict()["detail"] == ["6"]


def test_a_wrong_string_still_fails(world):
    span = _py(world, "Lib/dbm/sqlite3.py", "STORE_KV")
    world.show(span)
    claim = '`dbm.sqlite3.STORE_KV` is the string "REPLACE INTO Dict (key, value) VALUES (CAST(? AS BLOB), CAST(? AS BLOB))" in Lib/dbm/sqlite3.py'
    verdict = world.verdict(claim, (span, STORE_KV))
    assert verdict.reason == "value_not_covered"
    assert verdict.detail == ("REPLACE INTO Dict (key, value) VALUES (CAST(? AS BLOB), CAST(? AS BLOB))",)


def test_a_path_that_is_not_the_cited_source_still_fails(world):
    span = _py(world, "Lib/stat.py", "ST_GID")
    world.show(span)
    backticked = world.verdict("`stat.ST_GID` is defined in `Lib/other.py` with the value 7", (span, "ST_GID   = 7"))
    assert (backticked.reason, backticked.detail) == ("value_not_covered", ("Lib/other.py",))
    bare = world.verdict("`stat.ST_GID` is defined as 7 in Lib/other.py", (span, "ST_GID   = 7"))
    assert (bare.reason, bare.detail) == ("identifier_not_anchored", ("other.py",))
    other_dir = world.verdict("`stat.ST_GID` is defined as 7 in Lib/os/stat.py", (span, "ST_GID   = 7"))
    assert not other_dir.passed  # stat.py is the file name, but Lib/os/stat.py is another file


def test_another_module_prefix_fails(world):
    span = _py(world, "Lib/stat.py", "ST_GID")
    world.show(span)
    verdict = world.verdict("`email.ST_GID` is defined as 7", (span, "ST_GID   = 7"))
    assert (verdict.reason, verdict.detail) == ("identifier_not_anchored", ("email.ST_GID",))


def test_the_prefix_must_be_trailing_whole_components(world):
    span = world.text_span("Lib/dbm/sqlite3.py", STORE_KV).span_id
    world.show(span)
    for name in ("sqlite3.STORE_KV", "dbm.sqlite3.STORE_KV"):
        assert world.verdict(f"`{name}` is the string shown", (span, STORE_KV)).passed, name
    for name in ("db.sqlite3.STORE_KV", "dbm.STORE_KV", "bm.sqlite3.STORE_KV", "sqlite.STORE_KV"):
        verdict = world.verdict(f"`{name}` is the string shown", (span, STORE_KV))
        assert (verdict.reason, verdict.detail) == ("identifier_not_anchored", (name,)), name


def test_the_suffix_after_the_prefix_must_be_witnessed_by_the_same_citation(world):
    stat, email = _py(world, "Lib/stat.py", "ST_GID"), _py(world, "Lib/email/utils.py", "ST_GID")
    world.show(stat, email)
    # stat.ST_UID: the module prefix is right, but no passing quote or anchor names ST_UID
    verdict = world.verdict("`stat.ST_UID` is defined as 7", (stat, "ST_GID   = 7"))
    assert (verdict.reason, verdict.detail) == ("identifier_not_anchored", ("stat.ST_UID",))
    # utils.ST_GID passes through the email/utils.py citation, not through the stat.py one
    both = world.verdict("`utils.ST_GID` and `stat.ST_GID` are 7", (stat, "ST_GID   = 7"), (email, "ST_GID = 7"))
    assert both.passed


def test_detail_lists_every_offending_token_in_claim_order(world):
    span = _py(world, "Lib/token.py", "DEDENT")
    world.show(span)
    verdict = world.verdict("`token.DEDENT` is 9, see `Lib/x.py`, or 9 again, or 11", (span, "DEDENT = 5"))
    assert verdict.detail == ("9", "Lib/x.py", "11")
    names = world.verdict("`tokens.DEDENT` is 5 and `foo.bar` too", (span, "DEDENT = 5"))
    assert (names.reason, names.detail) == ("identifier_not_anchored", ("tokens.DEDENT", "foo.bar"))


def test_strip_source_paths_takes_whole_component_suffixes_only():
    paths = ["Lib/re/_constants.py"]
    assert strip_source_paths("in `Lib/re/_constants.py` and re/_constants.py and _constants.py.", paths).split() == ["in", "and", "and", "."]
    kept = "in Lib/other/_constants.py or my_constants.py or _constants.pyc or `x/_constants.py`"
    assert strip_source_paths(kept, paths) == kept


def test_old_verdicts_without_detail_read_and_hash_as_before():
    old = {"tier": 1, "passed": False, "action": "refetch", "reason": "value_not_covered", "support": ["s"]}
    verdict = Verdict.from_dict(old)
    assert verdict.detail == () and verdict.to_dict() == old
    assert json.loads(json.dumps(Verdict(1, False, "refetch", "x", ("s",), detail=("a",)).to_dict()))["detail"] == ["a"]


def test_the_detail_is_written_into_the_claim_event(world):
    span = _py(world, "Lib/token.py", "DEDENT")
    world.show(span)
    claim, _ = world.session.claim("`token.DEDENT` is 6", [Citation(span, "DEDENT = 5")])
    event = world.session.events[-1].payload["claim"]
    assert claim.status == "ungrounded" and event["verdict"]["detail"] == ["6"]
