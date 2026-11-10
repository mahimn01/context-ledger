#!/bin/zsh
# PREREG-v4-AMENDMENT-1 rerun (BUILD-V4FIX). What it does: reruns, LIVE and billed on the research
# key (--key research), only the units the v4 grid lost to two harness bugs, at W = 3000, questions
# v4, the extension corpus and 260 model calls per episode, as scripts/v4/grid.sh ran them:
#   1. Sonnet (claude-sonnet-5-5) arms L and L4 in all 16 cells (allowed and forced x rho 10, 100,
#      1000, 10000 x episodes 1-4 and 5-8), new run ids v4-grid-sonnet-<cond>-r<rho>-e<a>-<b>-c260-fix1,
#      in grid.sh's balanced order (the first half over every rho and condition, then the second);
#   2. then the Haiku forced rho-10000 units both stopped runs lost (L stopped in episode 2 of e1-5
#      and episode 9 of e6-10; L4 and L4-nogate never started): L episodes 2-5 and 9-10, L4 and
#      L4-nogate episodes 1-5 and 6-10, run ids ...-e2-5/-e1-5/-e9-10/-e6-10-c260-fix1, same order;
#   3. then analyze-v4 over the original runs and the -fix1 runs (a unit complete in a -fix1 run
#      supersedes the original's; units a run stopped in are dropped) into
#      runs/v4/grid-analysis-c260-fix1. runs/v4/grid-analysis-c260 is kept as it is.
# Baselines (B3, B4, B5) are not rerun. Before anything, it prints the plan (read-only) the caps
# come from: `eval plan --v4 --measured` over the completed units of the Sonnet grid and pilot runs
# (and the Haiku grid and pilot for the Haiku units), runs/v4/rerun-plan.log.
# STOP RULES: before each run, halt if the RESEARCH key's spend (spent + outstanding, its own
# ledger) plus that run's --budget would pass $150 (PREREG-v4-AMENDMENT-1); while Sonnet runs
# remain, the Haiku runs' caps ($1.85) are held back too, so the primary (Haiku) family is always
# completed; a Sonnet run halted that way ends the Sonnet phase, not the script. A run that stops
# on API errors (the bugs this reruns around) halts everything: nothing more is billed until it is
# looked at. A complete -fix1 run is skipped on a re-run; an incomplete one resumes (same --run-id).
# What it changes: new runs/<id>/ directories, the research key's spend ledger, runs/v4/<id>.log for
# each run, runs/v4/rerun-plan.log, runs/v4/rerun-driver.log (this script's own output) and
# runs/v4/grid-analysis-c260-fix1/. Run from the main tree, outside Claude Code:
#   zsh scripts/v4/rerun-sonnet.sh
cd ~/Documents/Dev/context-ledger
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT ANTHROPIC_LOG ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS
export PYTHONPATH=src
mkdir -p runs/v4
exec > >(tee -a runs/v4/rerun-driver.log) 2>&1
E=".venv/bin/python -m context_ledger.eval"
EXT=cf9bb5aff4ed6
QSHA=ad98051173a38fa13ba053f00f02d251e6de7d80c8eb7adcf59b0ec050d4b4c9  # questions-v4.jsonl, seed 3, 13 episodes (PREREG-v4)
MAXCALLS=${MAXCALLS:-260}
CAPLIMIT=150  # PREREG-v4-AMENDMENT-1: total research-key spend stays at or under $150
SONNET_CAP_ALLOWED=4.35; SONNET_CAP_FORCED=6.30  # 2x the plan's expected run (4 episodes of L and L4, measured)
HAIKU_CAP_L4=0.70; HAIKU_CAP_L4_E=0.70; HAIKU_CAP_L=0.30; HAIKU_CAP_L_E=0.15  # 2x expected: L4+L4-nogate x 5 episodes; L x 4; L x 2
HAIKU_RESERVE=1.85  # the four Haiku caps
echo "RERUN START $(date)"

if [ ! -f runs/corpus/$EXT/manifest.json ] || [ ! -f .cache/corpus/$EXT/address-v4.sqlite ]; then
  echo "STOP: the extension corpus or its index is missing"; echo RERUN-DONE; exit 1
fi
got=$(shasum -a 256 runs/corpus/$EXT/questions-v4.jsonl | awk '{print $1}')
if [ "$got" != "$QSHA" ]; then echo "STOP: questions-v4.jsonl is $got, not PREREG-v4's $QSHA"; echo RERUN-DONE; exit 1; fi

# The estimate (read-only): completed units only (eval/plan_v4.complete_units)
{
  echo "== Sonnet L, L4: 8 episodes x rho 10/100/1000/10000 x both conditions"
  ${=E} plan --v4 --key research --max-calls-episode $MAXCALLS --measured runs/v4-grid-sonnet-*-c260(N) runs/v4-pilot-sonnet-*-c260(N) \
    --model claude-sonnet-5-5 --episodes 8 --rhos 10,100,1000,10000 --arms L,L4
  echo "== Haiku forced rho 10000: L4, L4-nogate x 10 episodes"
  ${=E} plan --v4 --key research --max-calls-episode $MAXCALLS --measured runs/v4-grid-haiku-*-c260(N) runs/v4-pilot-haiku-*-c260(N) \
    --model claude-haiku-5-5 --episodes 10 --rhos 10000 --arms L4,L4-nogate --conditions forced
  echo "== Haiku forced rho 10000: L x 6 episodes (2-5, 9-10)"
  ${=E} plan --v4 --key research --max-calls-episode $MAXCALLS --measured runs/v4-grid-haiku-*-c260(N) runs/v4-pilot-haiku-*-c260(N) \
    --model claude-haiku-5-5 --episodes 6 --rhos 10000 --arms L --conditions forced
} > runs/v4/rerun-plan.log 2>&1
echo "ESTIMATE (runs/v4/rerun-plan.log, expected then x 1.2):"
grep -E "^== |custom grid  " runs/v4/rerun-plan.log | grep -v -E "allowed|forced"

research() { ${=E} spend status 2>/dev/null | awk '/^research spent/{gsub(/\$/,"",$3); s=$3} /^research outstanding/{gsub(/\$/,"",$3); o=$3} END{if (s != "" && o != "") print s + o}'; }
HALTED=""
go() { # model cond rho eps arms cap reserve id
  local id=$8
  if [ -f runs/$id/run.json ] && grep -q '"incomplete": false' runs/$id/run.json; then echo "SKIP $id"; return 0; fi
  s=$(research)
  if [ -z "$s" ]; then echo "HALT: spend status gave no research-key reading before $id"; echo RERUN-DONE; exit 1; fi
  if [ "$(echo "$s + $6 + $7 > $CAPLIMIT" | bc -l)" != 0 ]; then
    echo "HALT stopping rule: research-key spend $s + the \$$6 cap of $id (+ \$$7 held for Haiku) would pass \$$CAPLIMIT"; return 1
  fi
  echo "START $id $(date) research=$s"
  ${=E} run --live --key research --model $1 --arms $5 --questions v4 --window 3000 --corpus $EXT --max-calls-episode $MAXCALLS \
    --condition $2 --rho $3 --episodes $4 --budget $6 --run-id $id 2>&1 | tee runs/v4/$id.log | grep -v '^progress'
  rc=${pipestatus[1]}  # the run's exit code, not grep's
  if [ "$rc" != 0 ] || grep -q INCOMPLETE runs/v4/$id.log; then
    echo "EXIT $id FAILED/INCOMPLETE rc=$rc $(date) research=$(research)"
    if grep -q "INCOMPLETE (api_errors)" runs/v4/$id.log; then
      echo "HALT: $id stopped on API errors; nothing more is billed until it is looked at"; echo RERUN-DONE; exit 1
    fi
  else
    echo "EXIT $id $rc $(date) research=$(research)"
  fi
  return 0
}

# 1. Sonnet L and L4, grid.sh's order
for eps in 1-4 5-8; do for rho in 10 100 1000 10000; do for cond in allowed forced; do
  [ -n "$HALTED" ] && continue
  [ $cond = allowed ] && cap=$SONNET_CAP_ALLOWED || cap=$SONNET_CAP_FORCED
  go claude-sonnet-5-5 $cond $rho $eps L,L4 $cap $HAIKU_RESERVE v4-grid-sonnet-$cond-r$rho-e$eps-c260-fix1 || HALTED=1
done; done; done
[ -n "$HALTED" ] && echo "SONNET PHASE ENDED by the stopping rule; the Haiku units run on the reserve"

# 2. the Haiku forced rho-10000 units, in the same order (first half, then second; L before L4 as within a grid run)
for eps arms cap in 2-5 L $HAIKU_CAP_L 1-5 L4,L4-nogate $HAIKU_CAP_L4 9-10 L $HAIKU_CAP_L_E 6-10 L4,L4-nogate $HAIKU_CAP_L4_E; do
  go claude-haiku-5-5 forced 10000 $eps $arms $cap 0 v4-grid-haiku-forced-r10000-e$eps-c260-fix1 || break
done

# 3. the analysis: the originals first, the -fix1 runs after (a unit complete in both comes from the -fix1 run)
${=E} analyze-v4 runs/v4-grid-*-c$MAXCALLS(N) runs/v4-grid-*-c$MAXCALLS-fix1(N) --out runs/v4/grid-analysis-c$MAXCALLS-fix1 > runs/v4/grid-analysis-c$MAXCALLS-fix1.log 2>&1
echo "EXIT analyze-v4 $? research=$(research)"
echo RERUN-DONE
