#!/bin/zsh
# BUILD-V4B pilot (decision 7): every v4 arm (B3, B4, B5, L, L4, L4-nogate) on Haiku 5.5, W = 3000,
# 260 model calls per episode for every arm, episodes 11-12 of questions v3 (no v3 grid episode),
# rho 10, 1000 and the extension level 10000, conditions allowed and forced. Run from the main
# tree after `v4` is merged (--ff-only), outside Claude Code, by Mahimn's agent.
#
# Step 1 is offline and free. `corpus snapshot` copies every source file the extension uses into
# data/corpus-src/objects/ and checks it against the committed MANIFEST-cf9bb5aff4ed6.tsv (it reads
# nothing if the store is already complete; if this Mac's site-packages changed it refuses: then
# copy data/corpus-src/objects/ from the v4 worktree). `corpus extend` reads ONLY that snapshot and
# must build cf9bb5aff4ed6; then the address index.
# Step 2 is LIVE and billed on the research key (--key research): two runs, one per condition, each
# with all six arms and the three rho parts, capped at $2.50 each ($5.00 in all, under the $15
# pilot limit; about $1.66 expected, `eval plan --v4`). STOP RULE: before each run, halt if the
# COMBINED spend over every key's ledger (spent + outstanding) plus that run's --budget would pass
# $180, the research key's usable cap ($200 less the $20 reserve). Each run's output is kept in
# runs/v4/<id>.log; its EXIT line says FAILED/INCOMPLETE when it exited non-zero or its summary says
# INCOMPLETE. A complete run is skipped on a re-run; an incomplete one resumes (same --run-id).
cd ~/Documents/Dev/context-ledger
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT ANTHROPIC_LOG ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS
mkdir -p runs/v4
E=".venv/bin/python -m context_ledger.eval"
EXT=cf9bb5aff4ed6
MAXCALLS=${MAXCALLS:-260}
USABLE=180
ARMS=B3,B4,B5,L,L4,L4-nogate
CAP=2.50
if [ "$(echo "2 * $CAP > 15" | bc -l)" != 0 ]; then echo "STOP: the pilot's run caps would sum past \$15"; exit 1; fi

# 1. The source snapshot, the extension corpus from it, and the address index (offline).
${=E} corpus snapshot --base ca8e01d67009c --rho 10000 --seed 1 2>&1 | tee runs/v4/corpus-snapshot.log
rc=${pipestatus[1]}
if [ "$rc" != 0 ]; then echo "STOP corpus snapshot failed rc=$rc; see runs/v4/corpus-snapshot.log"; exit 1; fi
if [ ! -f runs/corpus/$EXT/manifest.json ]; then
  ${=E} corpus extend --base ca8e01d67009c --rho 10000 --seed 1 2>&1 | tee runs/v4/corpus-extend.log
  rc=${pipestatus[1]}
  if [ "$rc" != 0 ] || [ ! -f runs/corpus/$EXT/manifest.json ]; then echo "STOP corpus extend failed or built another id (rc=$rc); see runs/v4/corpus-extend.log"; exit 1; fi
fi
if [ ! -f .cache/corpus/$EXT/address-v4.sqlite ]; then
  ${=E} index build --corpus $EXT 2>&1 | tee runs/v4/index-build.log
  rc=${pipestatus[1]}
  if [ "$rc" != 0 ]; then echo "STOP index build failed rc=$rc"; exit 1; fi
fi
${=E} plan --v4 --max-calls-episode $MAXCALLS --key research | tee runs/v4/plan-v4.log

# 2. The live runs.
C="--live --key research --model claude-haiku-5-5 --questions v3 --window 3000 --corpus $EXT --episodes 11-12 --max-calls-episode $MAXCALLS"
combined() { ${=E} spend status 2>/dev/null | awk '/^combined spent/{gsub(/\$/,"",$3); s=$3} /^combined outstanding/{gsub(/\$/,"",$3); o=$3} END{if (s != "" && o != "") print s + o}'; }
complete() { # id parts: every rho part's run.json says incomplete false (one rho: runs/<id>/run.json; several: rho-*/run.json)
  n=$(grep -l '"incomplete": false' runs/$1/run.json runs/$1/rho-*/run.json(N) 2>/dev/null | wc -l | tr -d ' ')  # (N): no rho-* parts is no match, not an error
  [ "$n" = "$2" ]
}
go() { # id arms rho cond cap
  if complete $1 $(echo $3 | tr ',' '\n' | wc -l | tr -d ' '); then echo "SKIP $1"; return; fi
  s=$(combined)
  if [ -z "$s" ]; then echo "HALT: spend status gave no combined reading; check the ledgers before $1"; echo PILOT-DONE; exit 1; fi
  if [ "$(echo "$s + $5 > $USABLE" | bc -l)" != 0 ]; then echo "HALT stopping rule: combined spend $s + the \$$5 cap of $1 would pass \$$USABLE"; echo PILOT-DONE; exit 0; fi
  echo "START $1 $(date) combined=$s"
  ${=E} run ${=C} --arms $2 --rho $3 --condition $4 --budget $5 --run-id $1 2>&1 | tee runs/v4/$1.log | grep -v '^progress'
  rc=${pipestatus[1]}  # the run's exit code, not grep's
  if [ "$rc" != 0 ] || grep -q INCOMPLETE runs/v4/$1.log; then
    echo "EXIT $1 FAILED/INCOMPLETE rc=$rc $(date) combined=$(combined)"
  else
    echo "EXIT $1 $rc $(date) combined=$(combined)"
  fi
  ${=E} quick runs/$1 > runs/v4/$1.quick.txt 2>&1
}
for cond in allowed forced; do
  go v4-pilot-haiku-$cond-c$MAXCALLS $ARMS 10,1000,10000 $cond $CAP
done
${=E} analyze-v4 runs/v4-pilot-haiku-*-c$MAXCALLS --out runs/v4/pilot-analysis-c$MAXCALLS > runs/v4/pilot-analysis-c$MAXCALLS.log 2>&1
echo "EXIT analyze-v4 $?"
${=E} plan --v4 --max-calls-episode $MAXCALLS --key research --measured runs/v4-pilot-haiku-allowed-c$MAXCALLS runs/v4-pilot-haiku-forced-c$MAXCALLS > runs/v4/plan-v4-measured.log 2>&1
echo "EXIT plan --v4 --measured $? (runs/v4/plan-v4-measured.log: the grid priced from the pilot; fill PREREG-v4's {{PILOT_COSTS}} from it)"
echo PILOT-DONE
