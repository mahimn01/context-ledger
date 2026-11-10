#!/bin/zsh
# PREREG-v4-AMENDMENT-2 (BRIEF-NOGATE): the Sonnet L4-nogate arm, so the gate ablation (L4 vs L4-nogate)
# exists on both models (Haiku ran L4-nogate in its grid). What it does: runs, LIVE and billed on the
# research key (--key research --use-reserve), arm L4-nogate alone on Sonnet (claude-sonnet-5-5) in all 16 cells of
# the Sonnet grid (allowed and forced x rho 10, 100, 1000, 10000 x episodes 1-4 and 5-8), at W = 3000,
# questions v4, the extension corpus and 260 model calls per episode, as scripts/v4/grid.sh ran the
# grid: run ids v4-grid-sonnet-<cond>-r<rho>-e<a>-<b>-c260-nogate1, in grid.sh's balanced order (the
# first half of the episodes over every rho and condition, then the second). Then analyze-v4 over the
# original grid runs, the -fix1 runs and the -nogate1 runs into runs/v4/grid-analysis-c260-nogate1
# (runs/v4/grid-analysis-c260 and -c260-fix1 are kept as they are).
# Before anything is billed it checks that PREREG-v4-AMENDMENT-2.md, this script and src/ are committed
# as they are, the corpus (manifest and address index) and questions-v4.jsonl against their sha256s, and writes the
# read-only estimate the caps come from (`eval plan --v4 --measured` over the units that ran to the
# end, runs/v4/nogate-plan.log); it prints it, expected and x 1.2, against what is left under $150.
# STOP RULES: before each run, halt if the RESEARCH key's spend (spent + outstanding, its own ledger)
# plus that run's --budget would pass $150 (PREREG-v4-AMENDMENT-1's cap, kept by AMENDMENT-2). A halt
# ends the arm where it is, so a partial arm is the balanced order's prefix; the analysis then reports
# it. A run that stops on API errors, or exits non-zero (a crash or a failed leak scan), halts
# everything: nothing more is billed until it is looked at (the read-only analysis still runs over
# what completed). A complete -nogate1 run is skipped on a re-run; an incomplete one resumes (same
# --run-id). The API key is read only at run time, by the CLI's --key research --use-reserve; this script never
# reads, prints or logs it.
# What it changes: new runs/<id>/ directories, the research key's spend ledger, runs/v4/<id>.log for
# each run, runs/v4/nogate-plan.log, runs/v4/nogate-driver.log (this script's own output) and
# runs/v4/grid-analysis-c260-nogate1/ with its .log. Run from the main tree, outside Claude Code:
#   zsh scripts/v4/run-sonnet-nogate.sh
cd ~/Documents/Dev/context-ledger || { echo "STOP: no ~/Documents/Dev/context-ledger"; echo NOGATE-DONE; exit 1; }
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT ANTHROPIC_LOG ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS
export PYTHONPATH=src
mkdir -p runs/v4
exec > >(tee -a runs/v4/nogate-driver.log) 2>&1
E=".venv/bin/python -m context_ledger.eval"
MODEL=claude-sonnet-5-5
EXT=cf9bb5aff4ed6
QSHA=ad98051173a38fa13ba053f00f02d251e6de7d80c8eb7adcf59b0ec050d4b4c9  # questions-v4.jsonl, seed 3, 13 episodes (PREREG-v4)
MSHA=cdbc6aa6cf2a997336641c4095945dc694192549126a071de5f9f61d2ae91c12  # runs/corpus/cf9bb5aff4ed6/manifest.json (AMENDMENT-2)
ISHA=e27499598413619c531bf428a12a41930bdb16e58b30a1a874b8c40738c8206a  # .cache/corpus/cf9bb5aff4ed6/address-v4.sqlite, the index every v4 grid run used
MAXCALLS=260
CAPLIMIT=200  # PREREG-v4-AMENDMENT-3: research-key hard max; reserve released for this arm
typeset -A CAP=(allowed 1.15 forced 1.60)  # 2x the plan's expected run (4 episodes of L4-nogate), AMENDMENT-2
AMENDMENT=PREREG-v4-AMENDMENT-2.md
echo "NOGATE START $(date)"
stop() { echo "STOP: $1"; echo NOGATE-DONE; exit 1; }

# 1. Nothing is billed before the amendment, this script and the harness (src/) are committed as they are, nor on another
# corpus or questions file
for f in $AMENDMENT scripts/v4/run-sonnet-nogate.sh; do
  { git ls-files --error-unmatch $f && git diff --quiet HEAD -- $f; } > /dev/null 2>&1 || stop "$f is not committed as it is"
done
[ -z "$(git status --porcelain -- src 2>&1)" ] || stop "src/ differs from the commit; only the committed harness runs"
echo "AMENDMENT-2 commit $(git log -1 --format='%h %cI' -- $AMENDMENT); HEAD $(git rev-parse --short HEAD)"
check() { [ -f $1 ] || stop "$1 is missing"; local got=$(shasum -a 256 $1 | awk '{print $1}'); [ "$got" = "$2" ] || stop "$1 is $got, not $2"; }
check runs/corpus/$EXT/manifest.json $MSHA
check .cache/corpus/$EXT/address-v4.sqlite $ISHA
check runs/corpus/$EXT/questions-v4.jsonl $QSHA

research() { ${=E} spend status 2>/dev/null | awk '/^research spent/{gsub(/\$/,"",$3); s=$3} /^research outstanding/{gsub(/\$/,"",$3); o=$3} END{if (s != "" && o != "") printf "%.6f\n", s + o}'; }  # to the micro-dollar, as the ledger

# 2. The estimate (read-only): Sonnet L4-nogate has no measured unit, so the plan prices it from Haiku's
# measured L4-nogate times the v3 Sonnet/Haiku ratio of L; only units that ran to the end count
MEASURED=(runs/v4-grid-sonnet-*-c260(N) runs/v4-grid-sonnet-*-c260-fix1(N) runs/v4-pilot-sonnet-*-c260(N)
          runs/v4-grid-haiku-*-c260(N) runs/v4-grid-haiku-*-c260-fix1(N) runs/v4-pilot-haiku-*-c260(N))
echo "== Sonnet L4-nogate: 8 episodes x rho 10/100/1000/10000 x both conditions" > runs/v4/nogate-plan.log
${=E} plan --v4 --key research --use-reserve --max-calls-episode $MAXCALLS --measured $MEASURED \
  --model $MODEL --episodes 8 --rhos 10,100,1000,10000 --arms L4-nogate >> runs/v4/nogate-plan.log 2>&1 || stop "the estimate failed; see runs/v4/nogate-plan.log"
est=($(awk '/^  custom grid  / && !/episodes/ {gsub(/\$/, ""); print $(NF-1), $NF; exit}' runs/v4/nogate-plan.log))
[ ${#est} = 2 ] || stop "no custom grid estimate in runs/v4/nogate-plan.log"
echo "ESTIMATE (runs/v4/nogate-plan.log): expected \$${est[1]}, x 1.2 \$${est[2]}"
grep -E "claude-sonnet-5-5 +(allowed|forced) +L4-nogate|custom grid  " runs/v4/nogate-plan.log
for cond in allowed forced; do  # the plan's own 2x caps should be AMENDMENT-2's
  got=$(awk -v c=$cond '/^  custom grid  / && $3 == c {gsub(/\$/, ""); print $NF; exit}' runs/v4/nogate-plan.log)
  [ "$got" = "$CAP[$cond]" ] || echo "NOTE: the plan now suggests --budget \$$got for $cond, not AMENDMENT-2's \$$CAP[$cond]; AMENDMENT-2's caps are used (the stop rule bounds the spend either way)"
done
s=$(research)
[ -n "$s" ] || stop "spend status gave no research-key reading"
left=$(echo "$CAPLIMIT - $s" | bc -l)
echo "research-key spend \$$s; \$$left left under \$$CAPLIMIT"
if [ "$(echo "${est[2]} > $left" | bc -l)" != 0 ]; then
  echo "NOTE: the estimate x 1.2 (\$${est[2]}) is more than the \$$left left under \$$CAPLIMIT; the arm runs anyway (AMENDMENT-2), and the stop rule halts it before the first run that could pass \$$CAPLIMIT, leaving a partial arm in the balanced order"
fi

# 3. The arm, in grid.sh's order; a halt ends it where it is
HALT=""
go() { # cond rho eps cap id
  local id=$5 s rc
  if [ -f runs/$id/run.json ] && grep -q '"incomplete": false' runs/$id/run.json; then echo "SKIP $id (complete)"; return 0; fi
  s=$(research)
  if [ -z "$s" ]; then echo "HALT: spend status gave no research-key reading before $id"; HALT=spend; return 1; fi
  if [ "$(echo "$s + $4 > $CAPLIMIT" | bc -l)" != 0 ]; then
    echo "HALT stopping rule: research-key spend $s + the \$$4 cap of $id would pass \$$CAPLIMIT"; HALT=cap; return 1
  fi
  [ -d runs/$id ] && echo "RESUME $id (incomplete)"
  echo "START $id $(date) research=$s"
  ${=E} run --live --key research --use-reserve --model $MODEL --arms L4-nogate --questions v4 --window 3000 --corpus $EXT --max-calls-episode $MAXCALLS \
    --condition $1 --rho $2 --episodes $3 --budget $4 --run-id $id 2>&1 | tee runs/v4/$id.log | grep -v '^progress'
  rc=${pipestatus[1]}  # the run's exit code, not grep's
  if [ "$rc" = 0 ] && ! grep -q INCOMPLETE runs/v4/$id.log; then echo "EXIT $id 0 $(date) research=$(research)"; return 0; fi
  echo "EXIT $id FAILED/INCOMPLETE rc=$rc $(date) research=$(research)"
  if grep -q "INCOMPLETE (api_errors)" runs/v4/$id.log; then
    echo "HALT: $id stopped on API errors; nothing more is billed until it is looked at"; HALT=api_errors; return 1
  fi
  if [ "$rc" != 0 ]; then
    echo "HALT: $id exited $rc (a crash or a failed leak scan); nothing more is billed until it is looked at"; HALT=failed; return 1
  fi
  return 0
}
for eps in 1-4 5-8; do for rho in 10 100 1000 10000; do for cond in allowed forced; do
  [ -n "$HALT" ] || go $cond $rho $eps $CAP[$cond] v4-grid-sonnet-$cond-r$rho-e$eps-c260-nogate1
done; done; done
n=0
for f in runs/v4-grid-sonnet-*-c260-nogate1/run.json(N); do grep -q '"incomplete": false' $f && n=$((n + 1)); done
echo "NOGATE ARM: $n of 16 runs complete${HALT:+ (halted: $HALT; a partial arm)}"

# 4. The analysis (read-only on runs/): the originals, then the -fix1 runs (a unit complete in both comes
# from the -fix1 run), then the -nogate1 runs (Sonnet L4-nogate units, which no earlier run has)
nogate=(runs/v4-grid-*-c260-nogate1(N))
if [ ${#nogate} = 0 ]; then
  echo "no -nogate1 run exists: analysis skipped"
else
  ${=E} analyze-v4 runs/v4-grid-*-c260(N) runs/v4-grid-*-c260-fix1(N) $nogate --out runs/v4/grid-analysis-c260-nogate1 > runs/v4/grid-analysis-c260-nogate1.log 2>&1
  echo "EXIT analyze-v4 $? research=$(research)"
fi
echo NOGATE-DONE
[ -z "$HALT" ] || [ "$HALT" = cap ]
