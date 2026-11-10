"""BUILD-V4B decision 3: arm L4-nogate (mechanism l-v4-nogate) is L4 with the answer gate off, and
therefore no gate-triggered repair. On fake-model fixtures it differs from L4 only there: where
the gate passes, both send the same requests and write the same rows (arm, mechanism, gate label
and gate verdicts aside); where the gate would refuse, L4 repairs and L4-nogate emits; Tier 0/1
and the forced handling (the last candidate emitted as unverified_guess) are the same."""

from __future__ import annotations

import inspect
import json
import re

from context_ledger.anthropic.client_v4 import LedgerClientV4
from context_ledger.eval import cli
from context_ledger.eval.arms_v4 import ARMS_V4, L4_NOGATE_MECHANISM, L4NoGateEpisode
from context_ledger.eval.fakemodel import ScriptedPilotModel
from context_ledger.invariants import check
from context_ledger.store import Ledger, SqliteBlobStore
from test_l4 import OtherEntityOnce, WrongValue, answer, const, corpus, q, run_arm, value  # noqa: F401  (corpus is a fixture)

GATE_FIELDS = {"arm", "mechanism", "gate", "gate_reasons", "gate_reason"}


class Recording(ScriptedPilotModel):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[str] = []

    def __call__(self, request):
        self.requests.append(json.dumps(request, sort_keys=True))
        return super().__call__(request)


def questions():
    return [q(0, const("spool", "SPOOL_LIMIT"), "Deployment note: in this deployment, the retry limit of service orion-relay is 48213."),
            q(1, const("garden", "GARDEN_RATIO"), "Update: the retry limit of service orion-relay is now 70531."),
            q(2, const("spool", "SPOOL_LIMTI")), q(3, "What is the current retry limit of service orion-relay?"),
            q(4, const("harbor", "HARBOR_LIMIT"))]  # fmt: skip


def test_where_the_gate_passes_l4_nogate_is_l4(corpus, tmp_path):
    rows, requests = {}, {}
    for arm in ("L4", "L4-nogate"):
        model = Recording()
        results, run_dir = run_arm(corpus, tmp_path, arm, questions(), model)
        rows[arm], requests[arm] = results, model.requests
        events, blobs = Ledger(run_dir / arm / "episode-01.sqlite").events(), SqliteBlobStore(run_dir / arm / "episode-01.sqlite")
        assert check(events, blobs) == []
    assert requests["L4"] == requests["L4-nogate"] and requests["L4"]  # the same prompts, tools, pre-resolution and window
    for qid, r in rows["L4"].items():
        n = rows["L4-nogate"][qid]
        assert {k: v for k, v in r.to_dict().items() if k not in GATE_FIELDS} == {k: v for k, v in n.to_dict().items() if k not in GATE_FIELDS}
        assert (r.gate, n.gate, n.mechanism, n.arm) == ("on", "off", L4_NOGATE_MECHANISM, "L4-nogate")
        assert not n.gate_reasons and n.repairs == 0
    assert any(r.gate_reasons for r in rows["L4"].values())  # L4's gate did run on these answers, and passed them


def test_where_the_gate_refuses_l4_repairs_and_l4_nogate_emits(corpus, tmp_path):
    work, manifest = corpus
    asked = [q(0, const("harbor", "HARBOR_LIMIT")), q(1, const("spool", "SPOOL_LIMIT"))]
    l4, _ = run_arm(corpus, tmp_path, "L4", asked, OtherEntityOnce("spool.SPOOL_LIMIT", "harbor.HARBOR_LIMIT"))
    nogate, run_dir = run_arm(corpus, tmp_path, "L4-nogate", asked, OtherEntityOnce("spool.SPOOL_LIMIT", "harbor.HARBOR_LIMIT"))
    other = value(manifest, "Lib/harbor.py", "HARBOR_LIMIT")
    assert l4["t01q01"].repair_reasons == ["target_not_witnessed"] and l4["t01q01"].answer == value(manifest, "Lib/spool.py", "SPOOL_LIMIT")
    r = nogate["t01q01"]
    assert r.answer == other and r.reason == "answered" and r.repairs == 0 and r.repair_reasons == [] and r.calls == 1  # a true quote about another entity
    events = Ledger(run_dir / "L4-nogate" / "episode-01.sqlite").events()
    assert [e for e in events if e.type == "answer_emitted" and not e.payload["abstained"] and "gate" in e.payload] == []  # no gate verdict recorded
    wrong, _ = run_arm(corpus, tmp_path / "w", "L4-nogate", [q(0, const("garden", "GARDEN_RATIO"))], WrongValue(times=1)) if (tmp_path / "w").mkdir() is None else None
    assert wrong["t01q00"].answer == "987654321" and wrong["t01q00"].repairs == 0  # value_not_with_target is the gate's: not checked


class Ungrounded(ScriptedPilotModel):
    """Always answers with a quote that is in no span: Tier 1 blocks every candidate."""

    def __call__(self, request):
        return answer("4242", "the limit is `4242`", "@1", "THIS QUOTE IS IN NO SPAN 4242")


def test_tier_one_and_the_forced_handling_are_l4s(corpus, tmp_path):
    for condition in ("allowed", "forced"):
        out = {}
        for arm in ("L4", "L4-nogate"):
            results, _ = run_arm(corpus, tmp_path, arm, [q(0, const("garden", "GARDEN_RATIO"))], Ungrounded(), condition=condition)
            out[arm] = results["t01q00"]
        a, b = out["L4"], out["L4-nogate"]
        assert {k: v for k, v in a.to_dict().items() if k not in GATE_FIELDS} == {k: v for k, v in b.to_dict().items() if k not in GATE_FIELDS}
        if condition == "forced":  # the last candidate, still blocked by Tier 1, is emitted as an unverified guess in both
            assert b.reason == "unverified_guess" and b.unverified_guess and b.answer == "4242" and b.repair_reasons == ["final"]
        else:
            assert b.abstained and b.answer is None


def test_only_the_gate_switch_differs():
    assert (L4NoGateEpisode.ARM, L4NoGateEpisode.GATE, L4NoGateEpisode.MECHANISM) == ("L4-nogate", False, "l-v4-nogate")
    assert set(vars(L4NoGateEpisode)) - {"__module__", "__doc__", "__qualname__", "__firstlineno__", "__static_attributes__"} == {"ARM", "GATE", "MECHANISM"}
    source = inspect.getsource(LedgerClientV4)
    assert source.count("emit_answer(") == 1 and 'if self.gate else {}' in source


def test_a_run_records_the_arm_its_mechanism_and_l4s_prompt(tmp_path, capsys):
    from pilot_fixture import v2_fixture_files, write_fixture_tarball

    tar, sha = write_fixture_tarball(tmp_path, v2_fixture_files())
    e = lambda *argv: cli.main(["--workdir", str(tmp_path), *argv])  # noqa: E731
    mix = "anchors=5,probes=3,multihop=2,unanswerable=2,notes=4,supersessions=1,session=3,session_unanswerable=1"
    assert e("corpus", "build", "--seed", "3", "--window", "400", "--rho", "10,20", "--version", "v2", "--estimator", "bytes4-v0", "--tarball", str(tar), "--sha256", sha) == 0
    assert e("index", "build") == 0
    assert e("questions", "build", "--version", "v3", "--seed", "2", "--episodes", "1", "--questions-per-episode", "12", "--mix", mix, "--probe-gap", "4") == 0
    assert e("run", "--questions", "v3", "--arms", "L4,L4-nogate", "--episodes", "1", "--window", "3000", "--budget", "2", "--run-id", "ng",
             "--max-calls-episode", "260") == 0  # fmt: skip
    run = json.loads((tmp_path / "runs" / "ng" / "run.json").read_text())
    assert run["v4"]["l4_nogate_mechanism"] == "l-v4-nogate" and run["max_calls_episode"] == 260 and "L4" in run["prompts"]["sha256"]
    results = [json.loads(line) for line in (tmp_path / "runs" / "ng" / "results.jsonl").read_text().splitlines()]
    assert {(r["arm"], r["gate"], r["mechanism"]) for r in results} == {("L4", "on", "l-v4"), ("L4-nogate", "off", "l-v4-nogate")}
    assert (tmp_path / "runs" / "ng" / "L4-nogate" / "episode-01.sqlite").exists()
    capsys.readouterr()
    assert e("quick", "ng") == 0
    assert "L4-nogate" in capsys.readouterr().out
    assert "L4-nogate" in ARMS_V4 and re.fullmatch(r"[\w-]+", "L4-nogate")
