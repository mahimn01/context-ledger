#!/bin/zsh
# Tiny Sonnet cost pilot (Nov 2): episode 11 of questions v3 (not a grid episode), arms B5+L4, rho 10000,
# both conditions, $3 cap each; research key; same stop rule as grid.sh (research spend + cap <= $180).
cd ~/Documents/Dev/context-ledger
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT ANTHROPIC_LOG ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS
E=".venv/bin/python -m context_ledger.eval"
C="--live --key research --model claude-sonnet-5-5 --questions v3 --window 3000 --corpus cf9bb5aff4ed6 --episodes 11-11 --max-calls-episode 260 --arms B5,L4 --rho 10000"
research() { ${=E} spend status 2>/dev/null | awk '/^research spent/{gsub(/\$/,"",$3); s=$3} /^research outstanding/{gsub(/\$/,"",$3); o=$3} END{if (s != "" && o != "") print s + o}'; }
for cond in allowed forced; do
  id=v4-pilot-sonnet-$cond-c260
  if [ -f runs/$id/run.json ] && grep -q '"incomplete": false' runs/$id/run.json; then echo "SKIP $id"; continue; fi
  s=$(research); [ -z "$s" ] && { echo "HALT no research reading"; break; }
  if [ "$(echo "$s + 3 > 180" | bc -l)" != 0 ]; then echo "HALT stop rule $s"; break; fi
  echo "START $id $(date) research=$s"
  ${=E} run ${=C} --condition $cond --budget 3.00 --run-id $id 2>&1 | tee runs/v4/$id.log | grep -v '^progress'
  echo "EXIT $id rc=${pipestatus[1]} $(date) research=$(research)"
  ${=E} quick runs/$id > runs/v4/$id.quick.txt 2>&1
done
echo SONNET-PILOT-DONE
