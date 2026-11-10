#!/bin/zsh
# BUILD-V4B grid (decision 7; the design PREREG-v4 freezes): one model's grid per invocation, W =
# 3000, questions v4 (fresh: no item of questions v3 episodes 1-12 or 19) on the extension corpus,
# both conditions, 260 model calls per episode for every arm. Run from the main tree after the
# pilot, outside Claude Code, by Mahimn's agent; Haiku first, then Sonnet:
#   scripts/v4/grid.sh                                               # Haiku: 6 arms, episodes 1-10
#   MODEL=claude-sonnet-5-5 EPISODES=1-8 ARMS=B3,B4,B5,L,L4 scripts/v4/grid.sh
# Parameters (environment): MODEL, EPISODES (a-b), RHOS (comma list), ARMS (comma list), MAXCALLS,
# and the per-run caps CAP_ALLOWED and CAP_FORCED (defaults per model from `eval plan --v4`: 2x the
# expected cost of a run; price other choices with `plan --v4 --model --episodes --rhos --arms`).
# Run order, as v3's: the episodes in two halves (a run per condition, rho and half, all arms in
# it), the first half over every rho and condition before the second.
# LIVE and billed on the research key (--key research). STOP RULE: before each run, halt if the
# RESEARCH key's spend (spent + outstanding, its own ledger) plus that run's --budget would pass
# $180, the research key's usable cap ($200 less the $20 reserve). The Aperture key is frozen at
# $26.27 and not counted (equivalently, a combined cap of $206.27). Each run's output is kept in
# runs/v4/<id>.log; its EXIT line says FAILED/INCOMPLETE when it exited non-zero or its summary
# says INCOMPLETE. A complete run is skipped on a re-run; an incomplete one resumes (same --run-id).
cd ~/Documents/Dev/context-ledger
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT ANTHROPIC_LOG ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS
mkdir -p runs/v4
E=".venv/bin/python -m context_ledger.eval"
EXT=cf9bb5aff4ed6
QSHA=ad98051173a38fa13ba053f00f02d251e6de7d80c8eb7adcf59b0ec050d4b4c9  # questions-v4.jsonl, seed 3, 13 episodes (PREREG-v4)
MODEL=${MODEL:-claude-haiku-5-5}
EPISODES=${EPISODES:-1-10}
RHOS=${RHOS:-10,100,1000,10000}
ARMS=${ARMS:-B3,B4,B5,L,L4,L4-nogate}
MAXCALLS=${MAXCALLS:-260}
USABLE=180
case $MODEL in
  claude-haiku-5-5) CAP_ALLOWED=${CAP_ALLOWED:-1.25}; CAP_FORCED=${CAP_FORCED:-1.55};;
  claude-sonnet-5-5) CAP_ALLOWED=${CAP_ALLOWED:-16.30}; CAP_FORCED=${CAP_FORCED:-19.05};;
  *) echo "STOP: unknown MODEL $MODEL"; exit 1;;
esac
SHORT=${${MODEL#claude-}%%-*}
A=${EPISODES%-*}; B=${EPISODES#*-}
M=$(( A + (B - A + 1 + 1) / 2 - 1 ))  # the first half: A..M (the larger half if odd)
HALVES=("$A-$M" "$((M + 1))-$B")

if [ ! -f runs/corpus/$EXT/manifest.json ] || [ ! -f .cache/corpus/$EXT/address-v4.sqlite ]; then
  echo "STOP: build the extension corpus and its index first (scripts/v4/pilot.sh step 1)"; exit 1
fi
if [ ! -f runs/corpus/$EXT/questions-v4.jsonl ]; then
  ${=E} questions build --corpus $EXT --version v4 --seed 3 --episodes 13 2>&1 | tee runs/v4/questions-v4.log
  rc=${pipestatus[1]}
  if [ "$rc" != 0 ]; then echo "STOP questions build failed rc=$rc"; exit 1; fi
fi
got=$(shasum -a 256 runs/corpus/$EXT/questions-v4.jsonl | awk '{print $1}')
if [ "$got" != "$QSHA" ]; then echo "STOP: questions-v4.jsonl is $got, not PREREG-v4's $QSHA"; exit 1; fi
${=E} questions overlap --corpus $EXT --version v4 > runs/v4/questions-v4-overlap.log 2>&1
if [ "$?" != 0 ]; then echo "STOP: questions v4 overlap questions v3's asked episodes; see runs/v4/questions-v4-overlap.log"; exit 1; fi

C="--live --key research --model $MODEL --arms $ARMS --questions v4 --window 3000 --corpus $EXT --max-calls-episode $MAXCALLS"
combined() { ${=E} spend status 2>/dev/null | awk '/^combined spent/{gsub(/\$/,"",$3); s=$3} /^combined outstanding/{gsub(/\$/,"",$3); o=$3} END{if (s != "" && o != "") print s + o}'; }
research() { ${=E} spend status 2>/dev/null | awk '/^research spent/{gsub(/\$/,"",$3); s=$3} /^research outstanding/{gsub(/\$/,"",$3); o=$3} END{if (s != "" && o != "") print s + o}'; }
go() { # cond rho eps cap
  id="v4-grid-$SHORT-$1-r$2-e$3-c$MAXCALLS"
  if [ -f runs/$id/run.json ] && grep -q '"incomplete": false' runs/$id/run.json; then echo "SKIP $id"; return; fi
  s=$(research)
  if [ -z "$s" ]; then echo "HALT: spend status gave no research-key reading before $id"; echo GRID-DONE; exit 1; fi
  if [ "$(echo "$s + $4 > $USABLE" | bc -l)" != 0 ]; then echo "HALT stopping rule: research-key spend $s + the \$$4 cap of $id would pass \$$USABLE"; echo GRID-DONE; exit 0; fi
  echo "START $id $(date) research=$s combined=$(combined)"
  ${=E} run ${=C} --condition $1 --rho $2 --episodes $3 --budget $4 --run-id $id 2>&1 | tee runs/v4/$id.log | grep -v '^progress'
  rc=${pipestatus[1]}  # the run's exit code, not grep's
  if [ "$rc" != 0 ] || grep -q INCOMPLETE runs/v4/$id.log; then
    echo "EXIT $id FAILED/INCOMPLETE rc=$rc $(date) research=$(research) combined=$(combined)"
  else
    echo "EXIT $id $rc $(date) research=$(research) combined=$(combined)"
  fi
}
for eps in $HALVES; do for rho in ${(s:,:)RHOS}; do for cond in allowed forced; do
  [ $cond = allowed ] && cap=$CAP_ALLOWED || cap=$CAP_FORCED
  go $cond $rho $eps $cap; done; done; done
${=E} analyze-v4 runs/v4-grid-*-c$MAXCALLS --out runs/v4/grid-analysis-c$MAXCALLS > runs/v4/grid-analysis-c$MAXCALLS.log 2>&1
echo "EXIT analyze-v4 $?"
echo GRID-DONE
