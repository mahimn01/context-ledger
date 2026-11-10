#!/bin/zsh
# PREREG-v4 grid: the Haiku grid (6 arms, episodes 1-10), then the Sonnet core grid (5 arms, episodes 1-$SONNET_N), one live run at a time. Sonnet L4-nogate is not run here: it needs its own run ids and is decided after the core Sonnet grid from the research spend left (PREREG-v4 section 9).
cd ~/Documents/Dev/context-ledger
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT ANTHROPIC_LOG ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS
SONNET_N=${SONNET_N:-8}  # PREREG-v4 section 9 sizing: n = 8
echo "CHAIN START $(date)"
zsh scripts/v4/grid.sh > runs/v4/grid-haiku-driver.log 2>&1
echo "CHAIN haiku rc=$? $(date)"
MODEL=claude-sonnet-5-5 EPISODES=1-$SONNET_N ARMS=B3,B4,B5,L,L4 zsh scripts/v4/grid.sh > runs/v4/grid-sonnet-driver.log 2>&1
echo "CHAIN sonnet rc=$? $(date)"
echo CHAIN-DONE
