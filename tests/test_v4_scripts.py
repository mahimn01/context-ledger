"""BUILD-V4B decision 7: what scripts/v4/pilot.sh and grid.sh must do before anything is billed. Every
live run is on --key research at 260 calls per episode; exit status is the run's (zsh
${pipestatus[1]}); INCOMPLETE runs are flagged; the stop rule halts before a run whose --budget
would take the COMBINED spend (spent + outstanding, every key) past the research key's usable $180;
each run logs to runs/v4/<id>.log; the pilot's caps sum to at most $15; the grid is parameterised
and ordered by episode halves; extend reads the snapshot; the grid checks the questions file."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {name: (ROOT / "scripts" / "v4" / name).read_text() for name in ("pilot.sh", "grid.sh")}


def test_both_parse_as_zsh():
    for name in SCRIPTS:
        assert subprocess.run(["zsh", "-n", str(ROOT / "scripts" / "v4" / name)]).returncode == 0


def test_every_live_run_is_on_the_research_key_at_260_calls_with_its_own_log_and_status():
    for name, text in SCRIPTS.items():
        assert 'MAXCALLS=${MAXCALLS:-260}' in text and "--max-calls-episode $MAXCALLS" in text
        live = [line for line in text.splitlines() if "--live" in line]
        assert live and all("--key research" in line for line in live)
        assert "--use-reserve" not in text
        assert "tee runs/v4/$" in text and "rc=${pipestatus[1]}" in text and "grep -q INCOMPLETE" in text and "FAILED/INCOMPLETE" in text


def test_the_stop_rule_counts_the_combined_spend_against_180():
    # the pilot (already run) kept the stricter combined rule
    text = SCRIPTS["pilot.sh"]
    assert "USABLE=180" in text and "/^combined spent/" in text and "/^combined outstanding/" in text
    assert re.search(r'echo "\$s \+ \$[45] > \$USABLE" \| bc -l', text)


def test_the_grid_stop_rule_counts_the_research_key_spend_against_180():
    text = SCRIPTS["grid.sh"]
    assert "USABLE=180" in text and "/^research spent/" in text and "/^research outstanding/" in text
    assert "s=$(research)" in text
    assert re.search(r'echo "\$s \+ \$4 > \$USABLE" \| bc -l', text)  # research spent + outstanding + this run's cap


def test_the_pilot_runs_every_arm_on_v3_episodes_11_12_within_15_dollars():
    text = SCRIPTS["pilot.sh"]
    assert "ARMS=B3,B4,B5,L,L4,L4-nogate" in text and "--questions v3" in text and "--episodes 11-12" in text and "10,1000,10000" in text
    cap = float(re.search(r"^CAP=([0-9.]+)$", text, re.M).group(1))
    assert 2 * cap <= 15 and "for cond in allowed forced" in text
    assert text.index("corpus snapshot") < text.index("corpus extend") and "plan --v4 --max-calls-episode $MAXCALLS --key research --measured" in text


def test_the_grid_is_parameterised_ordered_by_halves_and_checks_its_questions():
    text = SCRIPTS["grid.sh"]
    for var in ("MODEL", "EPISODES", "RHOS", "ARMS"):
        assert f"{var}=${{{var}:-" in text
    assert "for eps in $HALVES; do for rho in ${(s:,:)RHOS}; do for cond in allowed forced" in text
    assert "--questions v4" in text and "QSHA=" in text and "questions overlap" in text and "--episodes 13" in text


# BUILD-V4FIX: scripts/v4/rerun-sonnet.sh, the PREREG-v4-AMENDMENT-1 rerun

RERUN = ROOT / "scripts" / "v4" / "rerun-sonnet.sh"
STUB = r'''#!/bin/zsh
# a stand-in for ".venv/bin/python -m context_ledger.eval": logs each command; spend lives in ./stub-spend
shift 2
echo "$*" >> stub-calls.log
case $1 in
  spend) echo "research spent \$$(cat stub-spend)"; echo "research outstanding \$0";;
  plan) echo "  custom grid                                                     \$1.000000   \$1.200000";;
  run) i=${@[(i)--run-id]}; id=${@[i+1]}; j=${@[(i)--budget]}; b=${@[j+1]}
       echo "$(cat stub-spend) + $b / 2" | bc -l > stub-spend.new; mv stub-spend.new stub-spend; mkdir -p runs/$id
       if [ -f stub-fail ] && grep -qx $id stub-fail; then echo '{"incomplete": true}' > runs/$id/run.json; echo "run $id: INCOMPLETE (api_errors)"
       else echo '{"incomplete": false}' > runs/$id/run.json; echo "run $id: complete"; fi;;
esac
exit 0
'''


def rerun(tmp_path, spent="78.566027", fail=()):
    """rerun-sonnet.sh against the stub in a throwaway HOME (its own copy of the tree's layout and of
    questions-v4.jsonl): no real CLI, no key, no network. Returns (driver output, the stub's command log)."""
    tree = tmp_path / "Documents" / "Dev" / "context-ledger"
    ext = "cf9bb5aff4ed6"
    for d in (tree / ".venv" / "bin", tree / "runs" / "corpus" / ext, tree / ".cache" / "corpus" / ext):
        d.mkdir(parents=True)
    (tree / "runs" / "corpus" / ext / "manifest.json").write_text("{}")
    (tree / ".cache" / "corpus" / ext / "address-v4.sqlite").write_text("")
    (tree / "runs" / "corpus" / ext / "questions-v4.jsonl").write_bytes((ROOT / "runs" / "corpus" / ext / "questions-v4.jsonl").read_bytes())
    for name in ("v4-grid-sonnet-allowed-r10-e1-4-c260", "v4-grid-haiku-forced-r10000-e6-10-c260"):  # originals for analyze-v4's first glob
        (tree / "runs" / name).mkdir()
    stub = tree / ".venv" / "bin" / "python"
    stub.write_text(STUB)
    stub.chmod(0o755)
    (tree / "stub-spend").write_text(spent + "\n")
    if fail:
        (tree / "stub-fail").write_text("".join(f + "\n" for f in fail))
    out = subprocess.run(["zsh", str(RERUN)], env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}, capture_output=True, text=True, timeout=120)
    return out.stdout + out.stderr, (tree / "stub-calls.log").read_text().splitlines()


SONNET_IDS = [f"v4-grid-sonnet-{c}-r{rho}-e{h}-c260-fix1" for h in ("1-4", "5-8") for rho in (10, 100, 1000, 10000) for c in ("allowed", "forced")]
HAIKU_IDS = [f"v4-grid-haiku-forced-r10000-e{e}-c260-fix1" for e in ("2-5", "1-5", "9-10", "6-10")]


def _runs(calls):
    return [c.split() for c in calls if c.startswith("run ")]


def _arg(call, flag):
    return call[call.index(flag) + 1]


def test_rerun_script_parses_and_every_live_run_is_on_the_research_key_at_260_calls():
    text = RERUN.read_text()
    assert subprocess.run(["zsh", "-n", str(RERUN)]).returncode == 0
    live = [line for line in text.splitlines() if "--live" in line]
    assert live and all("--key research" in line for line in live) and "--use-reserve" not in text
    assert "MAXCALLS=${MAXCALLS:-260}" in text and "--max-calls-episode $MAXCALLS" in text and "CAPLIMIT=150" in text
    assert "tee runs/v4/$id.log" in text and "rc=${pipestatus[1]}" in text and "grep -q INCOMPLETE" in text and "FAILED/INCOMPLETE" in text
    assert "runs/v4/rerun-driver.log" in text and "/^research spent/" in text and "/^research outstanding/" in text


def test_rerun_runs_only_l_and_l4_on_sonnet_then_the_lost_haiku_units_in_grid_order_then_analyzes(tmp_path):
    out, calls = rerun(tmp_path)
    runs = _runs(calls)
    assert [_arg(c, "--run-id") for c in runs] == SONNET_IDS + HAIKU_IDS
    assert {_arg(c, "--arms") for c in runs[:16]} == {"L,L4"} and [_arg(c, "--arms") for c in runs[16:]] == ["L", "L4,L4-nogate", "L", "L4,L4-nogate"]
    assert all("--live" in c and _arg(c, "--key") == "research" and _arg(c, "--max-calls-episode") == "260" for c in runs)
    assert {_arg(c, "--model") for c in runs[:16]} == {"claude-sonnet-5-5"} and {_arg(c, "--model") for c in runs[16:]} == {"claude-haiku-5-5"}
    assert [_arg(c, "--episodes") for c in runs[16:]] == ["2-5", "1-5", "9-10", "6-10"] and {(_arg(c, "--condition"), _arg(c, "--rho")) for c in runs[16:]} == {("forced", "10000")}
    analyze = [c for c in calls if c.startswith("analyze-v4")]
    assert len(analyze) == 1 and analyze[0].split()[-2:] == ["--out", "runs/v4/grid-analysis-c260-fix1"]
    shown = analyze[0].split()[1:-2]  # the originals first, then every -fix1 run
    assert shown[:2] == ["runs/v4-grid-haiku-forced-r10000-e6-10-c260", "runs/v4-grid-sonnet-allowed-r10-e1-4-c260"]
    assert len(shown[2:]) == 20 and all(s.endswith("-fix1") for s in shown[2:])
    plans = [n for n, c in enumerate(calls) if c.startswith("plan --v4") and "--measured" in c]
    assert plans == [0, 1, 2]  # priced before anything runs
    assert "ESTIMATE" in out and "RERUN-DONE" in out and "HALT" not in out


def test_rerun_stops_sonnet_before_150_holding_the_haiku_caps_back(tmp_path):
    out, calls = rerun(tmp_path, spent="141")
    ids = [_arg(c, "--run-id") for c in _runs(calls)]
    assert 0 < len(ids) - 4 < 16 and ids[:-4] == SONNET_IDS[: len(ids) - 4] and ids[-4:] == HAIKU_IDS  # the Haiku units still run
    assert "HALT stopping rule" in out and "SONNET PHASE ENDED" in out and any(c.startswith("analyze-v4") for c in calls)
    assert float((tmp_path / "Documents" / "Dev" / "context-ledger" / "stub-spend").read_text()) <= 150


def test_rerun_halts_everything_when_a_run_stops_on_api_errors(tmp_path):
    out, calls = rerun(tmp_path, fail=[SONNET_IDS[0]])
    assert [_arg(c, "--run-id") for c in _runs(calls)] == [SONNET_IDS[0]]
    assert "stopped on API errors" in out and not any(c.startswith("analyze-v4") for c in calls)


# PREREG-v4-AMENDMENT-2: scripts/v4/run-sonnet-nogate.sh, Sonnet's L4-nogate arm

NOGATE = ROOT / "scripts" / "v4" / "run-sonnet-nogate.sh"
NOGATE_STUB = r'''#!/bin/zsh
# a stand-in for ".venv/bin/python -m context_ledger.eval": logs each command; spend lives in ./stub-spend, the plan in ./stub-plan.txt
shift 2
echo "$*" >> stub-calls.log
case $1 in
  spend) echo "research spent \$$(cat stub-spend)"; echo "research outstanding \$0";;
  plan) cat stub-plan.txt;;
  run) i=${@[(i)--run-id]}; id=${@[i+1]}; j=${@[(i)--budget]}; b=${@[j+1]}
       echo "$(cat stub-spend) + $b / 2" | bc -l > stub-spend.new; mv stub-spend.new stub-spend; mkdir -p runs/$id
       if [ -f stub-crash ] && grep -qx $id stub-crash; then echo '{"incomplete": false}' > runs/$id/run.json; echo "leakscan: FAIL; results in runs/$id"; exit 1; fi
       if [ -f stub-fail ] && grep -qx $id stub-fail; then echo '{"incomplete": true}' > runs/$id/run.json; echo "run $id: INCOMPLETE (api_errors)"
       else echo '{"incomplete": false}' > runs/$id/run.json; echo "run $id: complete"; fi;;
esac
exit 0
'''
NOGATE_IDS = [f"v4-grid-sonnet-{c}-r{rho}-e{h}-c260-nogate1" for h in ("1-4", "5-8") for rho in (10, 100, 1000, 10000) for c in ("allowed", "forced")]
NOGATE_CAPS = {"allowed": "1.15", "forced": "1.60"}
GIT = ["git", "-c", "user.name=Mahimn Patel", "-c", "user.email=mahimn.patel.k@gmail.com"]


def stub_plan() -> str:
    """The plan's real rendering (render_plan_v4, run_plan's 2x caps) of Sonnet L4-nogate at the unit costs `plan --v4 --measured`
    gave at AMENDMENT-2 (Haiku's measured L4-nogate x the v3 Sonnet/Haiku ratio of L)."""
    from context_ledger.eval.plan_v4 import design_cost, render_plan_v4, run_plan
    from context_ledger.pricing import PICO_PER_USD

    design = {"model": "claude-sonnet-5-5", "episodes": 8, "rhos": (10, 100, 1000, 10000), "conditions": ("allowed", "forced"), "arms": ("L4-nogate",)}
    costs = {("claude-sonnet-5-5", c, "L4-nogate"): (PICO_PER_USD * v // 10**6, "pilot x v3 sonnet/haiku ratio") for c, v in (("allowed", 139723), ("forced", 197330))}
    return render_plan_v4({"source": "stub", "pilot_runs": [], "max_calls": 260, "costs": costs, "items": {"custom grid": design_cost(costs, design)},
                           "runs": {"custom grid": run_plan(costs, design)}, "ignored_pilot": [], "remaining_pusd": None}) + "\n"  # fmt: skip


def nogate_tree(tmp_path, spent="122.345727", plan=None):
    """A throwaway HOME holding the tree's layout: the real corpus manifest, address index and questions-v4.jsonl (linked, so
    their sha256 checks run on the real bytes), a git repo with PREREG-v4-AMENDMENT-2.md, the script and a src/ file
    committed, and the CLI stub."""
    tree = tmp_path / "Documents" / "Dev" / "context-ledger"
    ext = "cf9bb5aff4ed6"
    for d in (tree / ".venv" / "bin", tree / "runs" / "corpus" / ext, tree / ".cache" / "corpus" / ext):
        d.mkdir(parents=True)
    for rel in (f"runs/corpus/{ext}/manifest.json", f"runs/corpus/{ext}/questions-v4.jsonl", f".cache/corpus/{ext}/address-v4.sqlite"):
        (tree / rel).symlink_to(ROOT / rel)
    for name in ("v4-grid-sonnet-allowed-r10-e1-4-c260", "v4-grid-sonnet-allowed-r10-e1-4-c260-fix1", "v4-grid-haiku-forced-r10000-e6-10-c260-fix1"):
        (tree / "runs" / name).mkdir()  # an original and two -fix1 runs for analyze-v4's globs
    stub = tree / ".venv" / "bin" / "python"
    stub.write_text(NOGATE_STUB)
    stub.chmod(0o755)
    (tree / "stub-spend").write_text(spent + "\n")
    (tree / "stub-plan.txt").write_text(plan if plan is not None else stub_plan())
    (tree / "PREREG-v4-AMENDMENT-2.md").write_text("the amendment\n")
    (tree / "scripts" / "v4").mkdir(parents=True)
    (tree / "scripts" / "v4" / "run-sonnet-nogate.sh").write_bytes(NOGATE.read_bytes())
    (tree / "src").mkdir()
    (tree / "src" / "harness.py").write_text("# the harness\n")
    for cmd in (["init", "-q"], ["add", "PREREG-v4-AMENDMENT-2.md", "scripts", "src"], ["commit", "-q", "-m", "Amend PREREG-v4"]):
        subprocess.run([*GIT, *cmd], cwd=tree, check=True, capture_output=True, env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    return tree


def nogate(tmp_path, tree=None, fail=(), crash=(), **kw):
    """run-sonnet-nogate.sh against the stub: no real CLI, no key, no network. HOME is the tree's own throwaway home and the
    working directory a scratch one, so the script never reaches the real tree. Returns (exit code, driver output, the stub's
    command log)."""
    tree = tree or nogate_tree(tmp_path, **kw)
    for name, ids in (("stub-fail", fail), ("stub-crash", crash)):
        if ids:
            (tree / name).write_text("".join(i + "\n" for i in ids))
    scratch = tree.parents[2] / "scratch"
    scratch.mkdir(exist_ok=True)
    out = subprocess.run(["zsh", str(NOGATE)], env={"HOME": str(tree.parents[2]), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}, cwd=scratch, capture_output=True,
                         text=True, timeout=120)  # fmt: skip
    log = tree / "stub-calls.log"
    return out.returncode, out.stdout + out.stderr, log.read_text().splitlines() if log.exists() else []


def test_nogate_script_parses_and_every_live_run_is_sonnet_l4_nogate_on_the_research_key_at_260_calls():
    text = NOGATE.read_text()
    assert subprocess.run(["zsh", "-n", str(NOGATE)]).returncode == 0
    live = [line for line in text.splitlines() if "--live" in line]
    assert len(live) == 1 and "--key research" in live[0] and "--arms L4-nogate" in live[0] and "--model $MODEL" in live[0] and "--use-reserve" not in text
    assert "MODEL=claude-sonnet-5-5" in text and "MAXCALLS=260" in text and "--max-calls-episode $MAXCALLS" in text and "CAPLIMIT=150" in text
    assert "typeset -A CAP=(allowed 1.15 forced 1.60)" in text and "--questions v4 --window 3000 --corpus $EXT" in text
    assert "tee runs/v4/$id.log" in text and "rc=${pipestatus[1]}" in text and "grep -q INCOMPLETE" in text and "FAILED/INCOMPLETE" in text
    assert "runs/v4/nogate-driver.log" in text and "runs/v4/nogate-plan.log" in text and "/^research spent/" in text and "/^research outstanding/" in text
    assert "security" not in text and "_API_KEY" not in text and "environment credential" not in text  # the key is the CLI's business, at run time


def test_nogate_runs_sonnet_l4_nogate_in_all_16_cells_in_grid_order_then_analyzes(tmp_path):
    rc, out, calls = nogate(tmp_path)
    runs = _runs(calls)
    assert [_arg(c, "--run-id") for c in runs] == NOGATE_IDS
    assert all(_arg(c, "--arms") == "L4-nogate" and _arg(c, "--model") == "claude-sonnet-5-5" and _arg(c, "--key") == "research" for c in runs)
    assert all("--live" in c and _arg(c, "--max-calls-episode") == "260" and _arg(c, "--questions") == "v4" and _arg(c, "--window") == "3000" for c in runs)
    assert all(_arg(c, "--corpus") == "cf9bb5aff4ed6" and _arg(c, "--budget") == NOGATE_CAPS[_arg(c, "--condition")] for c in runs)
    assert [(_arg(c, "--condition"), _arg(c, "--rho"), _arg(c, "--episodes")) for c in runs[:2]] == [("allowed", "10", "1-4"), ("forced", "10", "1-4")]
    plan = [c for c in calls if c.startswith("plan")]
    assert len(plan) == 1 and calls[0] == plan[0] and "--v4" in plan[0] and "--measured" in plan[0] and "--arms L4-nogate" in plan[0] and "--episodes 8" in plan[0]
    analyze = [c for c in calls if c.startswith("analyze-v4")]
    assert len(analyze) == 1 and calls[-2:-1] == analyze and analyze[0].split()[-2:] == ["--out", "runs/v4/grid-analysis-c260-nogate1"]
    shown = analyze[0].split()[1:-2]  # the original, then the -fix1 runs, then every -nogate1 run
    assert shown[0].endswith("-c260") and [s.endswith("-fix1") for s in shown[1:3]] == [True, True] and shown[3:] == [f"runs/{i}" for i in sorted(NOGATE_IDS)]
    assert "ESTIMATE (runs/v4/nogate-plan.log): expected $10.785" in out and "x 1.2 $12.94" in out and "$27.654273 left under $150" in out
    assert "NOGATE ARM: 16 of 16 runs complete\n" in out and out.rstrip().endswith("NOGATE-DONE") and "HALT" not in out and "NOTE" not in out and rc == 0
    tree = tmp_path / "Documents" / "Dev" / "context-ledger"
    assert "AMENDMENT-2 commit" in (tree / "runs" / "v4" / "nogate-driver.log").read_text() and (tree / "runs" / "v4" / "nogate-plan.log").exists()
    assert float((tree / "stub-spend").read_text()) <= 150


def test_nogate_halts_before_150_in_balanced_order_and_reports_a_partial_arm(tmp_path):
    rc, out, calls = nogate(tmp_path, spent="140")
    ids = [_arg(c, "--run-id") for c in _runs(calls)]
    assert 0 < len(ids) < 16 and ids == NOGATE_IDS[: len(ids)]  # the balanced order's prefix
    assert "NOTE: the estimate x 1.2 ($12.942" in out and "the arm runs anyway" in out  # more than the $10 left, and still allowed
    spent = float((tmp_path / "Documents" / "Dev" / "context-ledger" / "stub-spend").read_text())
    nxt = NOGATE_IDS[len(ids)]
    assert spent <= 150 and spent + float(NOGATE_CAPS["allowed" if "-allowed-" in nxt else "forced"]) > 150  # it stopped only where the rule binds
    assert f"HALT stopping rule: research-key spend {spent:.6f} + the ${NOGATE_CAPS['allowed' if '-allowed-' in nxt else 'forced']} cap of {nxt} would pass $150" in out
    assert f"NOGATE ARM: {len(ids)} of 16 runs complete (halted: cap; a partial arm)" in out and any(c.startswith("analyze-v4") for c in calls) and rc == 0


def test_nogate_halts_everything_when_a_run_stops_on_api_errors_or_exits_non_zero(tmp_path):
    rc, out, calls = nogate(tmp_path / "a", fail=[NOGATE_IDS[0]])
    assert [_arg(c, "--run-id") for c in _runs(calls)] == NOGATE_IDS[:1] and "stopped on API errors" in out and rc == 1
    assert "NOGATE ARM: 0 of 16 runs complete (halted: api_errors" in out and any(c.startswith("analyze-v4") for c in calls) and "NOGATE-DONE" in out
    rc, out, calls = nogate(tmp_path / "b", crash=[NOGATE_IDS[2]])
    assert [_arg(c, "--run-id") for c in _runs(calls)] == NOGATE_IDS[:3] and f"HALT: {NOGATE_IDS[2]} exited 1" in out and rc == 1


def test_nogate_skips_complete_runs_and_resumes_an_incomplete_one(tmp_path):
    tree = nogate_tree(tmp_path)
    for name, incomplete in ((NOGATE_IDS[0], "false"), (NOGATE_IDS[1], "true")):
        (tree / "runs" / name).mkdir()
        (tree / "runs" / name / "run.json").write_text(f'{{"incomplete": {incomplete}}}')
    rc, out, calls = nogate(tmp_path, tree=tree)
    assert [_arg(c, "--run-id") for c in _runs(calls)] == NOGATE_IDS[1:] and f"SKIP {NOGATE_IDS[0]}" in out and f"RESUME {NOGATE_IDS[1]}" in out and rc == 0


def test_nogate_refuses_before_any_call_without_the_committed_amendment_or_on_another_corpus(tmp_path):
    def refused(tree, why):
        rc, out, calls = nogate(tmp_path, tree=tree)
        assert rc == 1 and calls == [] and f"STOP: {why}" in out and "NOGATE-DONE" in out  # not even the plan or spend status ran
    tree = nogate_tree(tmp_path / "edited")
    (tree / "PREREG-v4-AMENDMENT-2.md").write_text("edited after the commit\n")
    refused(tree, "PREREG-v4-AMENDMENT-2.md is not committed as it is")
    tree = nogate_tree(tmp_path / "untracked")
    subprocess.run([*GIT, "rm", "-q", "--cached", "PREREG-v4-AMENDMENT-2.md"], cwd=tree, check=True, env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    refused(tree, "PREREG-v4-AMENDMENT-2.md is not committed as it is")
    tree = nogate_tree(tmp_path / "script")
    (tree / "scripts" / "v4" / "run-sonnet-nogate.sh").write_text("# edited after the commit\n")
    refused(tree, "scripts/v4/run-sonnet-nogate.sh is not committed as it is")
    for name, edit in (("edited-src", "harness.py"), ("new-src", "added.py")):
        tree = nogate_tree(tmp_path / name)
        (tree / "src" / edit).write_text("# not committed\n")
        refused(tree, "src/ differs from the commit")
    tree = nogate_tree(tmp_path / "corpus")
    manifest = tree / "runs" / "corpus" / "cf9bb5aff4ed6" / "manifest.json"
    manifest.unlink()
    manifest.write_text("{}")
    refused(tree, "runs/corpus/cf9bb5aff4ed6/manifest.json is ")


def test_nogate_stops_without_its_tree_and_writes_nothing_where_it_was_started(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    out = subprocess.run(["zsh", str(NOGATE)], env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, cwd=scratch, capture_output=True, text=True, timeout=60)
    assert out.returncode == 1 and "STOP: no ~/Documents/Dev/context-ledger" in out.stdout and list(scratch.iterdir()) == []


def test_nogate_stops_when_the_estimate_cannot_be_read(tmp_path):
    rc, out, calls = nogate(tmp_path, plan="no estimate here\n")
    assert rc == 1 and _runs(calls) == [] and "STOP: no custom grid estimate" in out


def test_nogate_notes_a_plan_whose_caps_differ_from_the_amendments_and_keeps_the_amendments(tmp_path):
    plan = stub_plan().replace("--budget $1.60", "--budget $1.75")
    rc, out, calls = nogate(tmp_path, plan=plan)
    assert "NOTE: the plan now suggests --budget $1.75 for forced, not AMENDMENT-2's $1.60" in out and rc == 0
    assert {_arg(c, "--budget") for c in _runs(calls) if _arg(c, "--condition") == "forced"} == {"1.60"}


def test_amendment_2_states_the_scripts_design_caps_and_stop_rule():
    text = (ROOT / "PREREG-v4-AMENDMENT-2.md").read_text()
    script = NOGATE.read_text()
    for needle in ("-c260-nogate1", "$150", "$1.15", "$1.60", "SECONDARY", "EXPLORATORY", "scripts/v4/run-sonnet-nogate.sh", "runs/v4/grid-analysis-c260-nogate1"):
        assert needle in text, needle
    for sha in re.findall(r"^[QMI]SHA=([0-9a-f]{64})", script, re.M):
        assert sha in text, sha
