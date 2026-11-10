"""BUILD-V4 items 3-7 and 10: L4 (l-v4) and B5 offline, with scripted models.

Pre-resolution puts the asked target's exact span (or the address-miss certificate, or the latest
note) in the window from the question TEXT alone; a gate refusal is repaired, on each reason, at
most twice; forced, the fallback goes through repair and the gate and leaves only as
unverified_guess; search is capped per question; the question, not its note, is the turn's
protected first user segment; B5 has the same pre-resolution and tools but no gate and no repair;
no gold or metadata field of a question reaches either arm."""

from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from context_ledger.anthropic.client_v4 import LAST_ROUND
from context_ledger.eval import cli
from context_ledger.eval.address_tools import MAX_SEARCHES
from context_ledger.eval.arms import ArmConfig, CallLog
from context_ledger.eval.arms_v4 import ARMS_V4, B5Episode, pre_resolve
from context_ledger.eval.cli import run_estimator
from context_ledger.eval.corpus import CorpusIndex, load_manifest
from context_ledger.eval.fakemodel import SPAN_HEADER, ScriptedPilotModel, _pieces, _text_blocks
from context_ledger.eval.prompts import CONDITION_PROMPTS_V4
from context_ledger.eval.runner import measure_tool_definitions, offline_client, v4_tools
from context_ledger.index import INDEX_FILE, AddressIndex, NoteTable
from context_ledger.invariants import check
from context_ledger.pricing import PricingTable
from context_ledger.store import Ledger, SqliteBlobStore
from pilot_fixture import v2_fixture_files, write_fixture_tarball

MODEL = "claude-haiku-5-5"
SENTINEL = "SENTINEL-GOLD-7f3a"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    work = tmp_path_factory.mktemp("l4")
    tar, sha = write_fixture_tarball(work, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    assert e("index", "build") == 0
    return work, load_manifest(None, work / "runs" / "corpus")


def value(manifest: dict, path: str, name: str) -> str:
    text = (Path(manifest["cache_dir"]) / "files" / path).read_text()
    return re.search(rf"^{name} = (.+)$", text, re.M).group(1).strip()


def q(pos: int, text: str, note: str | None = None, **gold) -> dict:
    return {"qid": f"t01q{pos:02d}", "episode": 1, "pos": pos, "text": text, "note": None if note is None else {"text": note}, **gold}


def const(module: str, name: str) -> str:
    return f"What is the value of the module constant `{module}.{name}`?"


def run_arm(corpus, tmp_path, arm: str, questions: list[dict], responder, condition: str = "allowed", rho: int = 10, episode_calls: int = 90):
    work, manifest = corpus
    run_dir = tmp_path / f"{arm}-{condition}"
    run_dir.mkdir()
    pricing = PricingTable.verified_default()
    client, guard, _ = offline_client(run_dir, run_id=run_dir.name, pricing=pricing, budget_usd="2", responder=responder)
    estimator, definitions, _ = run_estimator(MODEL, False)
    index, address = CorpusIndex(manifest["cache_dir"], rho), AddressIndex(Path(manifest["cache_dir"]) / INDEX_FILE, rho)
    try:
        definitions = {**(definitions or {}), **measure_tool_definitions(client, MODEL, v4_tools([arm]))}
        cfg = ArmConfig(model=MODEL, window=3000, pricing=pricing, estimator=estimator, run_dir=run_dir, prompt_version=CONDITION_PROMPTS_V4[condition],
                        tool_definitions=definitions, condition=condition, address=address, max_calls_episode=episode_calls)  # fmt: skip
        results = ARMS_V4[arm](questions, index, client, cfg, CallLog(run_dir / "calls.jsonl"))
    finally:
        index.close()
        address.close()
        guard.close()
    return {r.qid: r for r in results}, run_dir


def question_of(request: dict) -> str:
    texts = [t for m in request["messages"] if m["role"] == "user" for k, t in _text_blocks(m["content"]) if k == "text"]
    return next(t for t in reversed(texts) if t.startswith(("What ", "Which ", "Question: ")))


def last_text(request: dict) -> str:
    texts = [t for k, t in _text_blocks(request["messages"][-1]["content"]) if k == "text"]
    return texts[-1] if texts else ""


def answer(value: str, claim: str, ref: str, quote: str) -> dict:
    text = f"ANSWER: {value}\n<claims>" + json.dumps([{"text": claim, "citations": [{"ref": ref, "quote": quote}]}]) + "</claims>"
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


def tool(name: str, args: dict, n: int) -> dict:
    return {"content": [{"type": "tool_use", "id": f"toolu_t_{n:05d}", "name": name, "input": args}], "stop_reason": "tool_use"}


class WrongValue(ScriptedPilotModel):
    """A true, quoted claim under a wrong ANSWER line, the first `times` answers to each question."""

    def __init__(self, times: int = 99) -> None:
        super().__init__()
        self.times, self.done = times, Counter()

    def __call__(self, request):
        out = super().__call__(request)
        text = out["content"][0].get("text", "") if out["stop_reason"] == "end_turn" else ""
        key = question_of(request) if request.get("messages") else ""
        if text.startswith("ANSWER: ") and "UNANSWERABLE" not in text.split("\n", 1)[0] and self.done[key] < self.times:
            self.done[key] += 1
            out = {**out, "content": [{"type": "text", "text": re.sub(r"^ANSWER: .*$", "ANSWER: 987654321", text, count=1, flags=re.M)}]}
        return out


class OtherEntityOnce(ScriptedPilotModel):
    """Asked about `asked`, first answers with a TRUE claim about `other` (a span it can see),
    as the v2c cross-wired errors did; then the scripted model."""

    def __init__(self, asked: str, other: str) -> None:
        super().__init__()
        self.asked, self.other, self.used = asked, other, False

    def __call__(self, request):
        if not self.used and f"`{self.asked}`" in question_of(request):
            for piece in [p for m in request["messages"] for _, t in _text_blocks(m["content"]) for p in _pieces(t)]:
                line = re.search(rf"^{self.other.rsplit('.', 1)[1]} = (\S+)$", piece["body"], re.M)
                if piece["kind"] == "span" and line:
                    self.used = True
                    return answer(line.group(1), f"`{self.other}` is `{line.group(1)}`", piece["ref"], line.group(0))
        return super().__call__(request)


class StaleOnce(ScriptedPilotModel):
    """On a session probe, first cites the OLDEST note on the pair (still verbatim); then the
    scripted model, which answers from the latest."""

    used = False

    def __call__(self, request):
        if not self.used and question_of(request).startswith("What is the current"):
            notes = [p for m in request["messages"] for _, t in _text_blocks(m["content"]) for p in _pieces(t) if (p["path"] or "").startswith("note:")]
            first = min(notes, key=lambda p: int(p["path"].split(":")[1]))
            line = re.search(r"the (.+?) of service (\S+) is (\d+)\.", first["body"])
            self.used = True
            return answer(line.group(3), f"the {line.group(1)} of service `{line.group(2)}` is `{line.group(3)}`", first["ref"], line.group(0))
        return super().__call__(request)


def test_pre_resolution_reads_the_exact_span_and_names_its_address(corpus, tmp_path):
    work, manifest = corpus
    results, run_dir = run_arm(corpus, tmp_path, "L4", [q(0, const("spool", "SPOOL_LIMIT"))], ScriptedPilotModel())
    r = results["t01q00"]
    assert r.reason == "answered" and r.answer == value(manifest, "Lib/spool.py", "SPOOL_LIMIT") and r.calls == 1  # no tool call was needed
    assert r.preresolved["status"] == "exact" and r.preresolved["addresses"][0].startswith("Lib/spool.py:")
    events, blobs = Ledger(run_dir / "L4" / "episode-01.sqlite").events(), SqliteBlobStore(run_dir / "L4" / "episode-01.sqlite")
    users = [blobs.get(e.payload["segment"]["text_hash"]).decode() for e in events if e.type == "segment_appended" and e.payload["segment"]["role"] == "user"]
    assert users[0] == const("spool", "SPOOL_LIMIT")  # the question first
    assert any(u.startswith("SPOOL_LIMIT = ") for u in users)  # the pre-resolved span, verbatim
    assert any(u.startswith("Index: `spool.SPOOL_LIMIT` is at Lib/spool.py:") for u in users)
    assert check(events, blobs) == []


def test_a_miss_gets_the_address_miss_certificate(corpus, tmp_path):
    results, run_dir = run_arm(corpus, tmp_path, "L4", [q(0, const("spool", "SPOOL_LIMTI"))], ScriptedPilotModel())
    r = results["t01q00"]
    assert r.preresolved["status"] == "miss" and "spool.SPOOL_LIMIT" in r.preresolved["near"]
    assert r.abstained and r.reason == "model_abstained"
    blobs = SqliteBlobStore(run_dir / "L4" / "episode-01.sqlite")
    events = Ledger(run_dir / "L4" / "episode-01.sqlite").events()
    texts = [blobs.get(e.payload["segment"]["text_hash"]).decode() for e in events if e.type == "segment_appended" and e.payload["segment"]["role"] == "user"]
    certificate = next(t for t in texts if t.startswith("Index: no exact address for `spool.SPOOL_LIMTI`"))
    assert "Nearest DIFFERENT identifiers (none of them is the one asked): `spool.SPOOL_LIMIT`" in certificate


def test_target_not_witnessed_is_repaired_with_the_anchored_span(corpus, tmp_path):
    work, manifest = corpus
    questions = [q(0, const("harbor", "HARBOR_LIMIT")), q(1, const("spool", "SPOOL_LIMIT"))]
    results, run_dir = run_arm(corpus, tmp_path, "L4", questions, OtherEntityOnce("spool.SPOOL_LIMIT", "harbor.HARBOR_LIMIT"))
    r = results["t01q01"]
    assert r.gate_reasons == ["target_not_witnessed", "anchored"] and r.repairs == 1 and r.repair_reasons == ["target_not_witnessed"]
    assert r.reason == "answered" and r.answer == value(manifest, "Lib/spool.py", "SPOOL_LIMIT")
    assert value(manifest, "Lib/harbor.py", "HARBOR_LIMIT") in r.blocked_answers


def test_value_not_with_target_is_repaired_once_then_answered(corpus, tmp_path):
    work, manifest = corpus
    results, _ = run_arm(corpus, tmp_path, "L4", [q(0, const("garden", "GARDEN_RATIO"))], WrongValue(times=1))
    r = results["t01q00"]
    assert r.gate_reasons == ["value_not_with_target", "anchored"] and r.repair_reasons == ["value_not_with_target"]
    assert r.reason == "answered" and r.answer == value(manifest, "Lib/garden.py", "GARDEN_RATIO") and "987654321" in r.blocked_answers


def test_stale_superseded_is_repaired_with_the_latest_note(corpus, tmp_path):
    questions = [
        q(0, const("spool", "SPOOL_LIMIT"), "Deployment note: in this deployment, the retry limit of service orion-relay is 48213."),
        q(1, const("garden", "GARDEN_LIMIT"), "Update: the retry limit of service orion-relay is now 70531."),
        q(2, "What is the current retry limit of service orion-relay?"),
    ]
    results, _ = run_arm(corpus, tmp_path, "L4", questions, StaleOnce())
    r = results["t01q02"]
    assert r.preresolved["status"] == "note" and r.preresolved["chain"] == ["note:0", "note:1"]
    assert r.gate_reasons == ["stale_superseded", "anchored"] and r.repair_reasons == ["stale_superseded"]
    assert r.reason == "answered" and r.answer == "70531" and "48213" in r.blocked_answers


def test_at_most_two_repairs_then_v3s_gate_message_and_the_bound(corpus, tmp_path):
    results, run_dir = run_arm(corpus, tmp_path, "L4", [q(0, const("garden", "GARDEN_RATIO"))], WrongValue())
    r = results["t01q00"]
    assert r.repairs == 2 and r.repair_reasons == ["value_not_with_target"] * 2
    assert r.abstained and r.answer is None and len(r.gate_reasons) >= 3 and set(r.gate_reasons) == {"value_not_with_target"}
    blobs = SqliteBlobStore(run_dir / "L4" / "episode-01.sqlite")
    events = Ledger(run_dir / "L4" / "episode-01.sqlite").events()
    texts = [blobs.get(e.payload["segment"]["text_hash"]).decode() for e in events if e.type == "segment_appended" and e.payload["segment"]["role"] == "user"]
    assert sum(t.startswith("Repair (value_not_with_target)") for t in texts) == 2
    assert sum(t.startswith("Verification: your answer `987654321`") for t in texts) == len(r.gate_reasons) - 2  # later refusals: v3's message, no repair
    assert all(e.payload["abstained"] for e in events if e.type == "answer_emitted")  # nothing un-gated left


class ForcedLastRound(ScriptedPilotModel):
    """Forced: a wrong ANSWER line first; then status calls until the harness's last-round repair;
    then `final` ("wrong" or "right")."""

    def __init__(self, final: str) -> None:
        super().__init__()
        self.final, self.n, self.wrong_once = final, 0, False

    def __call__(self, request):
        self.n += 1
        out = super().__call__(request)
        if LAST_ROUND in last_text(request):
            if self.final == "right":
                return out
        elif self.wrong_once:
            return tool("ctx_status", {}, self.n)
        self.wrong_once = True
        text = out["content"][0]["text"]
        return {**out, "content": [{"type": "text", "text": re.sub(r"^ANSWER: .*$", "ANSWER: 987654321", text, count=1, flags=re.M)}]}


def test_forced_fallback_goes_through_repair_and_the_gate_before_it_is_an_unverified_guess(corpus, tmp_path):
    work, manifest = corpus
    results, _ = run_arm(corpus, tmp_path, "L4", [q(0, const("garden", "GARDEN_RATIO"))], ForcedLastRound("wrong"), condition="forced")
    r = results["t01q00"]
    assert r.repair_reasons == ["value_not_with_target", "final"] and r.calls == 10
    assert r.reason == "unverified_guess" and r.unverified_guess and r.forced_fallback and not r.abstained and r.answer == "987654321"
    results, _ = run_arm(corpus, tmp_path / "right", "L4", [q(0, const("garden", "GARDEN_RATIO"))], ForcedLastRound("right"), condition="forced") if (tmp_path / "right").mkdir() is None else None
    r = results["t01q00"]
    assert r.repair_reasons == ["value_not_with_target", "final"] and r.reason == "answered" and not r.unverified_guess
    assert r.answer == value(manifest, "Lib/garden.py", "GARDEN_RATIO") and r.gate_reasons[-1] == "anchored"


class Unanswerable(ScriptedPilotModel):
    def __call__(self, request):
        return self._reply("ANSWER: UNANSWERABLE\n<claims>[]</claims>")


def test_forced_with_no_candidate_is_forced_noanswer(corpus, tmp_path):
    results, _ = run_arm(corpus, tmp_path, "L4", [q(0, const("garden", "GARDEN_RATIO"))], Unanswerable(), condition="forced")
    r = results["t01q00"]
    assert r.reason == "forced_noanswer" and r.forced_violation and r.abstained and r.answer is None and not r.unverified_guess
    assert r.stop_reason == "model_abstained"  # v3's reason, kept for comparisons with L


class LateWrong(ScriptedPilotModel):
    """Forced: status calls for 8 rounds, then a wrong ANSWER line in round 9 (refused and
    repaired), then whatever the last round brings."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def __call__(self, request):
        self.n += 1
        out = super().__call__(request)
        if self.n <= 8:
            return tool("ctx_status", {}, self.n)
        text = out["content"][0]["text"]
        return {**out, "content": [{"type": "text", "text": re.sub(r"^ANSWER: .*$", "ANSWER: 987654321", text, count=1, flags=re.M)}]}


def test_the_last_round_repair_never_stacks_on_an_unanswered_repair(corpus, tmp_path):
    results, run_dir = run_arm(corpus, tmp_path, "L4", [q(0, const("garden", "GARDEN_RATIO"))], LateWrong(), condition="forced")
    r = results["t01q00"]
    assert r.repair_reasons == ["value_not_with_target"] and r.calls == 10  # round 9's repair; then no "final" on top of it
    db = run_dir / "L4" / "episode-01.sqlite"
    events, blobs = Ledger(db).events(), SqliteBlobStore(db)
    texts = [blobs.get(e.payload["segment"]["text_hash"]).decode() for e in events if e.type == "segment_appended" and e.payload["segment"]["role"] == "user"]
    assert texts.count(LAST_ROUND) == 1 and not any(t.endswith(LAST_ROUND) and t.startswith("Repair") for t in texts)  # said once, alone
    assert r.reason == "unverified_guess" and r.answer == "987654321"


class Looker(ScriptedPilotModel):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def __call__(self, request):
        self.n += 1
        return tool("lookup", {"name": f"spool.NAME_{self.n}"}, self.n)


def test_b5_keeps_the_episode_call_bound(corpus, tmp_path):
    questions = [q(0, const("garden", "GARDEN_RATIO")), q(1, const("spool", "SPOOL_LIMIT"))]
    results, _ = run_arm(corpus, tmp_path, "B5", questions, Looker(), episode_calls=15)
    assert (results["t01q00"].calls, results["t01q01"].calls) == (10, 5) and all(r.reason == "bound_hit" for r in results.values())


class ReadBig(ScriptedPilotModel):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def __call__(self, request):
        self.n += 1
        if self.n == 1:
            return tool("read", {"address": "Lib/bigmod.py:4-47"}, self.n)
        return super().__call__(request)


def test_b5_abstains_with_budget_exceeded_instead_of_sending_more_than_w(big_corpus, tmp_path):
    results, run_dir = run_arm(big_corpus, tmp_path, "B5", [q(0, "What is the default value of parameter `limit` of `bigmod.big`?")], ReadBig(), rho=20)
    r = results["t01q00"]
    assert r.reason == "budget_exceeded" and r.abstained and r.calls == 1


class Searcher(ScriptedPilotModel):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def __call__(self, request):
        self.n += 1
        if self.n <= MAX_SEARCHES + 1:
            return tool("search", {"query": f"limit {self.n}"}, self.n)
        return self._reply("ANSWER: UNANSWERABLE\n<claims>[]</claims>")


def test_search_is_capped_per_question(corpus, tmp_path):
    results, run_dir = run_arm(corpus, tmp_path, "L4", [q(0, const("garden", "GARDEN_RATIO"))], Searcher())
    r = results["t01q00"]
    assert r.tool_counts == {"search": MAX_SEARCHES + 1}
    blobs = SqliteBlobStore(run_dir / "L4" / "episode-01.sqlite")
    events = Ledger(run_dir / "L4" / "episode-01.sqlite").events()
    tool_texts = [blobs.get(e.payload["segment"]["text_hash"]).decode() for e in events if e.type == "segment_appended" and e.payload["segment"]["role"] == "tool"]
    assert sum(t.startswith("error: search_limit") for t in tool_texts) == 1


def test_the_question_not_its_note_is_the_protected_current_question(corpus, tmp_path):
    from context_ledger.backstop import PayoffModel
    from context_ledger.window import replay

    questions = [q(0, const("spool", "SPOOL_LIMIT"), "Deployment note: in this deployment, the retry limit of service orion-relay is 48213.")]
    _, run_dir = run_arm(corpus, tmp_path, "L4", questions, ScriptedPilotModel())
    db = run_dir / "L4" / "episode-01.sqlite"
    events, blobs = Ledger(db).events(), SqliteBlobStore(db)
    first_call = next(n for n, e in enumerate(events) if e.type == "call")
    before = events[:first_call]
    state = replay(before)
    from context_ledger.fold import BackstopPolicyV1
    from context_ledger.tokens import estimator_for

    policy = BackstopPolicyV1(budget_tokens=3000, model=MODEL, token_counter=estimator_for(state).name, rules="v2c")
    model = PayoffModel(state, before, blobs, policy, PricingTable.from_dict(state.pricing), estimator_for(state))
    current = next(s for s in state.segments if s.segment_id == model.current_question)
    assert blobs.get(current.text_hash).decode() == const("spool", "SPOOL_LIMIT")


def test_no_gold_or_metadata_field_reaches_l4_or_b5(corpus, tmp_path):
    """Sentinel test: every gold and metadata field of the questions holds a sentinel; no request
    either arm sends, and no byte of L4's ledger or B5's transcript, contains it."""
    gold = {k: SENTINEL for k in ("answer", "upstream_answer", "subject", "gold_path", "attribute", "template", "superseded", "probe_of")}
    gold |= {"gold_lines": [SENTINEL], "gold_paths": [SENTINEL], "gold_lines_all": [[SENTINEL]], "confusers": {"10": SENTINEL}, "notes_of": [SENTINEL]}
    questions = [q(0, const("spool", "SPOOL_LIMIT"), "Deployment note: in this deployment, the retry limit of service orion-relay is 48213.", **gold),
                 q(1, const("spool", "SPOOL_LIMTI"), **gold), q(2, "What is the current retry limit of service orion-relay?", **gold)]  # fmt: skip
    seen: list[str] = []

    class Recording(ScriptedPilotModel):
        def __call__(self, request):
            seen.append(json.dumps(request))
            return super().__call__(request)

    for arm in ("L4", "B5"):
        _, run_dir = run_arm(corpus, tmp_path, arm, questions, Recording())
        artifacts = [p.read_bytes() for p in run_dir.rglob("*") if p.is_file() and p.name != "fake-spend.jsonl"]
        assert not any(SENTINEL.encode() in a for a in artifacts)
    assert seen and not any(SENTINEL in s for s in seen)
    assert list(inspect.signature(pre_resolve).parameters) == ["question_text", "address", "notes"]


def test_b5_has_the_pre_resolution_and_tools_but_no_gate_and_no_repair(corpus, tmp_path):
    work, manifest = corpus
    results, run_dir = run_arm(corpus, tmp_path, "B5", [q(0, const("garden", "GARDEN_RATIO")), q(1, const("spool", "SPOOL_LIMTI"))], WrongValue())
    r = results["t01q00"]
    assert r.answer == "987654321" and r.reason == "answered" and r.calls == 1  # emitted as given: nothing checks it
    assert r.gate_reason is None and r.gate_reasons is None and r.repairs is None and r.preresolved["status"] == "exact"
    assert results["t01q01"].preresolved["status"] == "miss"
    source = inspect.getsource(B5Episode)
    assert "emit_answer" not in source and "answer_gate" not in source and "repair" not in source
    transcript = [json.loads(line) for line in (run_dir / "B5" / "episode-01.jsonl").read_text().splitlines()]
    assert [t["type"] for t in transcript][:2] == ["question", "response"]


def test_pre_resolve_reads_the_question_text_only(corpus):
    work, manifest = corpus
    address = AddressIndex(Path(manifest["cache_dir"]) / INDEX_FILE, 10)
    notes = NoteTable()
    notes.add("note:3", "Update: the shard key of service sadr-router is now qc46r.")
    assert pre_resolve("What is the current shard key of service sadr-router?", address, notes).chain == ("note:3",)
    assert pre_resolve("What is the current shard key of service sadr-routerd?", address, notes).status == "no_note"
    assert pre_resolve("What is the value of the module constant `spool.SPOOL_LIMIT`?", address, notes).status == "exact"
    assert pre_resolve("How are you?", address, notes).status == "no_target"


def big_module() -> str:
    body = "".join(f"    total = total + {k} * value  # step {k} of the long computation, kept for the fixture\n" for k in range(40))
    return f'"""A module with one long function."""\n\n\ndef big(value, limit=123, mode="fast"):\n    """Sum many steps."""\n    total = 0\n{body}    return min(total, limit)\n'


@pytest.fixture(scope="module")
def big_corpus(tmp_path_factory):
    work = tmp_path_factory.mktemp("l4big")
    tar, sha = write_fixture_tarball(work, {**v2_fixture_files(), "Lib/bigmod.py": big_module()})
    e = lambda *argv: cli.main(["--workdir", str(work), *argv])  # noqa: E731
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0",
             "--tarball", str(tar), "--sha256", sha) == 0  # fmt: skip
    assert e("index", "build") == 0
    return work, load_manifest(None, work / "runs" / "corpus")


def test_a_target_longer_than_the_read_cap_is_pre_resolved_whole_so_it_anchors(big_corpus, tmp_path):
    """A def over corpus_read's cap (W // 4) would come as a text span of its first lines, which
    Tier 1 and the gate cannot anchor to `bigmod.big`; within W // 2 the harness reads
    its anchored span whole (L4 and B5 alike)."""
    from context_ledger.eval.cli import run_estimator
    from context_ledger.tokens import source_class

    work, manifest = big_corpus
    text = (Path(manifest["cache_dir"]) / "files" / "Lib/bigmod.py").read_text()
    estimator, _, _ = run_estimator(MODEL, False)
    size = estimator.count(text, source_class("Lib/bigmod.py"))
    assert 3000 // 4 < size <= 3000 // 2  # over the read cap, within the pre-resolution cap
    limit = re.search(r"limit=(\d+)", text).group(1)
    question = "What is the default value of parameter `limit` of `bigmod.big`?"
    results, run_dir = run_arm(big_corpus, tmp_path, "L4", [q(0, question)], ScriptedPilotModel(), rho=20)
    r = results["t01q00"]
    assert r.reason == "answered" and r.answer == limit and r.calls == 1 and r.gate_reason == "anchored" and r.claim_statuses == ["grounded"]
    events = Ledger(run_dir / "L4" / "episode-01.sqlite").events()
    spans = {e.payload["span"]["span_id"]: e.payload["span"] for e in events if e.type == "span_registered"}
    held = [spans[e.payload["segment"]["ref"]] for e in events if e.type == "segment_appended" and e.payload["segment"]["ref_kind"] == "span"]
    assert [s["anchor"].get("qualname") for s in held] == ["big"]
    (tmp_path / "b5").mkdir()
    results, run_dir = run_arm(big_corpus, tmp_path / "b5", "B5", [q(0, question)], ScriptedPilotModel(), rho=20)
    assert results["t01q00"].answer == limit
    first = json.loads((run_dir / "B5" / "episode-01.jsonl").read_text().splitlines()[0])
    assert first["preresolved"]["status"] == "exact"
