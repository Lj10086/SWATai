#!/bin/bash
# Run every LTM experiment at once: Exp 1 (rotation), Exp 2 (nitrogen) and Exp 3 (irrigation)
# at efficiency 0.80, 0.55 and 0.90. 4 ES seeds x 3 strategies, 100k HRU-episode steps per seed
# (2026-10-30: about 6 h on a 10-core Apple Silicon Mac; Exp 3 dominates).
#
#   perl -e 'use POSIX; setsid(); exec @ARGV' -- bash scripts/run_ltm.sh   # detached
#   SMOKE=1 bash scripts/run_ltm.sh                                          # tiny budgets, minutes
#
# Logs: runs/ltm_<tag>.log (each ends "=== ... exit <code>"); runs/ltm_done.log when all finish.
# Results: runs/ltm_exp<n>[_eff55|_eff90].json; finished stages are reused on a rerun.
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-4}
BUDGET=${BUDGET:-100000}
[ "$SMOKE" = 1 ] && SEEDS=1 && BUDGET=3000
run() {
  tag=$1; shift
  {
    echo "=== $* start $(date)"
    uv run python -m swat_gym.experiments.ltm "$@" --seeds "$SEEDS" --budget "$BUDGET" 2>&1 \
      | grep --line-buffered -v "evals  best"
    echo "=== $* exit ${PIPESTATUS[0]} $(date)"
  } > "runs/ltm_$tag.log"
}
mkdir -p runs
rm -f runs/ltm_done.log
run exp1 --lever R &
run exp2 --lever N &
run exp3 --lever I &
run exp3_eff55 --lever I --irr-eff 0.55 &
run exp3_eff90 --lever I --irr-eff 0.90 &
wait
echo "=== ALL DONE $(date)" > runs/ltm_done.log
