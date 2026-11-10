"""F.1 and 0.3: the eval CLI offline end to end, and proof that every live path goes through the
gate and that nothing builds a real client without --live."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from context_ledger.eval import cli
from context_ledger.live import entry, factory
from context_ledger.live.gate import LiveRefused
from context_ledger.live.keys import EnvironmentKeyProvider
from context_ledger.live.spend import SpendGuard, read_spend
from context_ledger.pricing import PICO_PER_USD, PricingTable
from context_ledger.tokens import BYTES4_V0
from pilot_fixture import write_fixture_tarball


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    work = tmp_path_factory.mktemp("cli")
    tar, sha = write_fixture_tarball(work)
    assert cli.main(["--workdir", str(work), "corpus", "build", "--seed", "3", "--estimator", "bytes4-v0", "--tarball", str(tar),
                     "--sha256", sha, "--window", "300", "--rho", "10"]) == 0  # fmt: skip
    assert cli.main(["--workdir", str(work), "questions", "build", "--seed", "9", "--episodes", "2"]) == 0
    return work, tar, sha


def e(work, *args) -> int:
    return cli.main(["--workdir", str(work), *args])


def test_offline_smoke_run_score_report(workdir, capsys, monkeypatch):
    work, _, _ = workdir
    import anthropic

    def no_client(*a, **k):
        raise AssertionError("an SDK client was constructed without --live")

    monkeypatch.setattr(anthropic.Anthropic, "__init__", no_client)
    monkeypatch.setattr(cli, "open_live", lambda *a, **k: pytest.fail("open_live without --live"))
    capsys.readouterr()
    assert e(work, "run", "--arms", "B3,B4,L", "--rho", "10", "--episodes", "1", "--questions-per-episode", "3", "--model", "claude-haiku-5-5",
             "--budget", "0.75", "--smoke", "--window", "1500", "--run-id", "smoke") == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "complete" in out and "leakscan: pass" in out
    run = json.loads((work / "runs" / "smoke" / "run.json").read_text())
    assert run["live"] is False and run["leakscan"] == "pass" and run["smoke"] and run["questions"]["B4"] == run["questions"]["L"]
    assert len(run["questions"]["B3"]) == 3 and len(run["questions"]["L"]) == 4  # the fold probe is the 4th question for B4 and L
    assert (work / "runs" / "smoke" / "fake-spend.jsonl").exists() and not (work / ".clp-spend").exists()
    assert e(work, "score", "smoke") == 0 and e(work, "report", "smoke", "--resamples", "100") == 0
    assert "## SpendGuard" in (work / "runs" / "smoke" / "report.md").read_text()
    assert e(work, "spend", "verify", "--spend-file", str(work / "runs" / "smoke" / "fake-spend.jsonl")) == 0
    assert e(work, "spend", "status", "--spend-file", str(work / "runs" / "smoke" / "fake-spend.jsonl")) == 0
    assert e(work, "leakscan") == 0


def fitted_envelope(monkeypatch) -> None:
    """Stand in for a fitted envelope file (BUILD-V2), so a live run gets as far as the gate."""
    monkeypatch.setattr(cli, "run_estimator", lambda model, live, calibration=None, directory=None: (BYTES4_V0, {"B3": 0, "B4": 0, "L": 0}, []))


def test_every_live_command_goes_through_open_live(workdir, monkeypatch):
    work, tar, sha = workdir
    seen = []

    def gate(args, environ, **kw):
        seen.append(args.command)
        raise LiveRefused("test")

    monkeypatch.setattr(cli, "open_live", gate)
    fitted_envelope(monkeypatch)  # BUILD-V2: without one, a live run refuses before the gate
    assert e(work, "run", "--live", "--budget", "0.75", "--smoke", "--window", "1500") == 2
    assert e(work, "calibrate", "--live", "--tarball", str(tar), "--sha256", sha) == 2
    assert seen == ["run", "calibrate"]


def test_the_real_gate_refuses_inside_claude_code_before_any_key_or_client(workdir, monkeypatch, capsys):
    work, tar, sha = workdir
    monkeypatch.setattr(factory, "make_live_client", lambda s: pytest.fail("client built"))
    monkeypatch.setattr(entry, "make_live_client", lambda s: pytest.fail("client built"))
    monkeypatch.setattr(EnvironmentKeyProvider, "get", lambda self: pytest.fail("key read"))
    monkeypatch.setenv("CLAUDECODE", "1")
    fitted_envelope(monkeypatch)  # BUILD-V2: without one, a live run refuses before the gate
    capsys.readouterr()
    assert e(work, "run", "--live", "--budget", "0.75", "--smoke") == 2
    assert e(work, "calibrate", "--live", "--tarball", str(tar), "--sha256", sha) == 2
    err = capsys.readouterr().err
    assert err.count("live mode refused: CLAUDECODE") == 2
    assert not (work / ".clp-spend").exists()  # refused before the SpendGuard was even opened


def test_offline_calibration_needs_the_fake_counter_and_never_writes_calibration(workdir):
    work, tar, sha = workdir
    repo_cal = Path(cli.ROOT) / "calibration"
    before = {f: f.read_bytes() for f in repo_cal.glob("*.json")}
    assert e(work, "calibrate", "--tarball", str(tar), "--sha256", sha) == 2
    out = work / "cal"
    assert e(work, "calibrate", "--counter", "fake", "--tarball", str(tar), "--sha256", sha, "--out", str(out)) == 0
    data = json.loads(next(out.glob("claude-haiku-5-5-*.json")).read_text())
    assert data["counter"] == "fake"
    assert {f: f.read_bytes() for f in repo_cal.glob("*.json")} == before  # the repo's calibration is untouched
    assert all(json.loads(v)["counter"] != "fake" for v in before.values())


def test_grid_refuses_when_the_budget_is_short(workdir, capsys):
    work, _, _ = workdir
    table = PricingTable.verified_default()
    with SpendGuard(work / ".clp-spend" / "spend.jsonl", run_id="earlier", pricing=table) as guard:
        for _ in range(300):  # about $120 of the $150 hard cap assumed spent at Sonnet's worst case
            r = guard.reserve(model="claude-sonnet-5-5", request_sha256="0" * 64, prompt_tokens_upper=90_000, max_tokens=4096)
            guard.assume_spent(r, "test")
    capsys.readouterr()
    assert e(work, "run", "--grid", "--budget", "1", "--episodes", "12", "--key", "aperture") == 2  # BUILD-KEY: the ledger filled above
    assert "x 1.2 is more than" in capsys.readouterr().err
    assert e(work, "plan", "--window", "16000", "--key", "aperture") == 0 and "DOES NOT FIT" in capsys.readouterr().out  # the W = 16,000 prior
    assert e(work, "spend", "resolve") == 2


def test_spend_status_reports_open_reservations(workdir, tmp_path, capsys):
    table = PricingTable.verified_default()
    path = tmp_path / "spend.jsonl"
    guard = SpendGuard(path, run_id="crash", pricing=table)
    guard.reserve(model="claude-haiku-5-5", request_sha256="0" * 64, prompt_tokens_upper=100, max_tokens=10)
    guard.close()
    capsys.readouterr()
    assert cli.main(["spend", "status", "--spend-file", str(path)]) == 0 and "open reservations [1]" in capsys.readouterr().out
    assert cli.main(["spend", "resolve", "--assume-spent", "--spend-file", str(path)]) == 0
    assert "resolved 1 open reservations" in capsys.readouterr().out
    assert cli.main(["spend", "status", "--spend-file", str(path)]) == 0 and "open reservations" not in capsys.readouterr().out


def test_arms_measure_the_window_with_calib_v2_and_the_guard_keeps_the_envelope(workdir, monkeypatch):
    """M4 item 5: with a live calibration in calibration/, every arm runs on calib-v2 while the
    live SpendGuard's counting fallback gets the conservative calib-v1 estimator. BUILD-V2
    (minimal update): the arms now run on calib-v3, calib-v2's content slopes plus the envelope
    (offline: the default envelope when no envelope file exists)."""
    from context_ledger.tokens import CalibEstimator, MarginalCalibEstimator

    work, _, _ = workdir
    assert e(work, "run", "--arms", "B3", "--episodes", "1", "--questions-per-episode", "1", "--budget", "0.1", "--window", "1500", "--run-id", "w2") == 0
    assert json.loads((work / "runs" / "w2" / "run.json").read_text())["estimator"]["estimator"] == "calib-v3"
    fitted_envelope(monkeypatch)
    seen = {}

    def gate(args, environ, **kw):
        seen.update(kw)
        raise LiveRefused("test")

    monkeypatch.setattr(cli, "open_live", gate)
    assert e(work, "run", "--live", "--budget", "0.75", "--smoke", "--window", "1500") == 2
    assert type(seen["estimator"]) is CalibEstimator and not isinstance(seen["estimator"], MarginalCalibEstimator)


def test_run_resumes_by_run_id_and_refuses_another_config(workdir, capsys):
    work, _, _ = workdir
    args = ["run", "--arms", "B3,L", "--episodes", "1", "--questions-per-episode", "3", "--budget", "0.2", "--window", "1500", "--run-id", "again"]
    capsys.readouterr()
    assert e(work, *args) == 0
    out = capsys.readouterr().out
    assert out.count("progress: run again rho 10 claude-haiku-5-5") == 2 and "sk-" not in out
    assert e(work, *args) == 0 and "progress:" not in capsys.readouterr().out  # everything was complete: nothing reruns
    assert e(work, *args[:-4], "--window", "1600", "--run-id", "again") == 2
    assert "config hash" in capsys.readouterr().err


def test_analyze_scores_with_the_one_scorer_and_writes_every_output(workdir, tmp_path, capsys):
    work, _, _ = workdir
    assert e(work, "run", "--arms", "B3,B4,L", "--episodes", "2", "--questions-per-episode", "8", "--budget", "0.5", "--window", "1500", "--run-id", "an") == 0
    capsys.readouterr()
    assert e(work, "analyze", "an", "--out", str(tmp_path / "out"), "--resamples", "200") == 0
    out = capsys.readouterr().out
    assert "H1-B3" in out and "H4" in out and "wrote cells.csv" in out
    for name in ("cells.csv", "tests.csv", "tests.json", "exclusions.csv", "figure_data.csv"):
        assert (tmp_path / "out" / name).exists(), name
    tests = json.loads((tmp_path / "out" / "tests.json").read_text())["tests"]
    assert {t["test"] for t in tests} >= {"H1-B3", "H1-B4", "H2", "H2-McNemar", "H3", "H3-NI", "H4"}
    assert all(t["verdict"] == "not run" for t in tests if t["test"] in ("H1-B3", "H1-B4", "H2", "H3"))  # one rho only: no slope
    assert e(work, "analyze", "nothing-here", "--out", str(tmp_path / "none")) == 2


def test_every_live_command_takes_the_key_profile_and_use_reserve(workdir, monkeypatch):
    """BUILD-KEY: run (smoke included), calibrate and calibrate --envelope read the --key's
    environment credential entry (default research) and pass --use-reserve on; open_live takes the ledger from
    that provider's profile."""
    work, tar, sha = workdir
    seen = []

    def gate(args, environ, **kw):
        assert isinstance(kw["keys"], EnvironmentKeyProvider) and "spend_path" not in kw
        seen.append((args.command, kw["keys"].profile, kw["use_reserve"]))
        raise LiveRefused("test")

    monkeypatch.setattr(cli, "open_live", gate)
    fitted_envelope(monkeypatch)
    smoke = ("run", "--live", "--budget", "0.75", "--smoke", "--window", "1500")
    assert e(work, *smoke) == 2
    assert e(work, *smoke, "--key", "aperture") == 2
    assert e(work, *smoke, "--key", "research", "--use-reserve") == 2
    assert e(work, "calibrate", "--live", "--tarball", str(tar), "--sha256", sha) == 2
    assert e(work, "calibrate", "--live", "--key", "aperture", "--tarball", str(tar), "--sha256", sha) == 2
    assert e(work, "calibrate", "--live", "--envelope") == 2
    assert e(work, "calibrate", "--live", "--envelope", "--key", "aperture", "--use-reserve") == 2
    assert seen == [("run", "research", False), ("run", "aperture", False), ("run", "research", True), ("calibrate", "research", False),
                    ("calibrate", "aperture", False), ("calibrate", "research", False), ("calibrate", "aperture", True)]  # fmt: skip
    with pytest.raises(SystemExit):
        e(work, *smoke, "--key", "other")


def test_a_run_records_its_key_profile_and_its_caps(workdir, capsys):
    work, _, _ = workdir
    base = ("run", "--arms", "B3", "--episodes", "1", "--questions-per-episode", "1", "--window", "1500")
    capsys.readouterr()
    assert e(work, *base, "--budget", "0.1", "--run-id", "key-aperture", "--key", "aperture") == 0
    assert "(aperture key ledger: " in capsys.readouterr().out
    assert e(work, *base, "--budget", "0.1", "--run-id", "key-research") == 0
    assert "of the $200 hard cap, $180 usable without --use-reserve)" in capsys.readouterr().out
    for run_id, profile, cap in (("key-aperture", "aperture", 150), ("key-research", "research", 180)):
        run_dir = work / "runs" / run_id
        run = json.loads((run_dir / "run.json").read_text())
        assert run["key_profile"] == run["spend"]["key_profile"] == profile and run["spend"]["global_cap_pusd"] == cap * PICO_PER_USD
        for name in ("results.jsonl", "calls.jsonl"):
            assert {json.loads(line)["key_profile"] for line in (run_dir / name).read_text().splitlines()} == {profile}
    # the key is not part of the configuration: a run resumes under the other key, and each row keeps the key that paid for it
    assert e(work, *base, "--budget", "0.1", "--run-id", "key-aperture") == 0 and "progress:" not in capsys.readouterr().out
    assert json.loads((work / "runs" / "key-aperture" / "run.json").read_text())["key_profile"] == "research"
    assert {json.loads(line)["key_profile"] for line in (work / "runs" / "key-aperture" / "results.jsonl").read_text().splitlines()} == {"aperture"}
    assert e(work, *base, "--budget", "180.01", "--run-id", "key-over") == 2  # offline too: research's reserve is held back
    assert "only --use-reserve releases" in capsys.readouterr().err
    assert e(work, *base, "--budget", "180.01", "--run-id", "key-over", "--use-reserve") == 0
    assert "--use-reserve: the research key's reserve is released for run key-over" in capsys.readouterr().err
    run = json.loads((work / "runs" / "key-over" / "run.json").read_text())
    assert run["spend"]["use_reserve"] is True and run["spend"]["global_cap_pusd"] == 200 * PICO_PER_USD
    assert read_spend(work / "runs" / "key-over" / "fake-spend.jsonl")[0]["kind"] == "use_reserve"
