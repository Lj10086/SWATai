#!/usr/bin/env bash
# Full pipeline in paper order: tests -> ceiling gate -> controller -> grower rule -> Exp 1 -> Exp 2.
#
#   bash scripts/run_all.sh                           # lambda_n = 0
#   NO3_PRICE=16 OUT_TAG=n16 bash scripts/run_all.sh  # one frontier point
#   SMOKE=1 bash scripts/run_all.sh                   # tiny budgets, end-to-end check (~minutes)
#
# Detach long runs (harness reaps at ~30 min):
#   perl -e 'use POSIX; setsid(); exec @ARGV' -- bash scripts/run_all.sh
# Finished CMA/PPO stages are reused if their config matches, so reruns resume.
set -euo pipefail
cd "$(dirname "$0")/.."

NO3_PRICE="${NO3_PRICE:-0.0}"
# All cores: the engine is CPU-bound and scales to ~5x at 10 workers on an M5.
WORKERS="${WORKERS:-$(getconf _NPROCESSORS_ONLN)}"
# Budgets are engine runs, except CEIL_PW (CMA-ES evals per oracle and for the shared fit)
# and GROWER_EVALS (a behaviour fit, not a profit search). The controller gets the same
# engine runs as CMA-ES and each PPO seed (rule 2).
if [ "${SMOKE:-0}" = "1" ]; then
    OUT_TAG="${OUT_TAG:-smoke}"; BUDGET=200; PPO_SEEDS=1
    CEIL_PW=12; CTRL_BUDGET=2000; GROWER_EVALS=8
else
    BUDGET="${BUDGET:-300000}"; PPO_SEEDS="${PPO_SEEDS:-3}"
    CEIL_PW="${CEIL_PW:-2000}"
    CTRL_BUDGET="${CTRL_BUDGET:-$BUDGET}"; GROWER_EVALS="${GROWER_EVALS:-220}"
fi
suffix="${OUT_TAG:+_$OUT_TAG}"
LOG="runs/run_all${suffix}.log"
mkdir -p runs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run() { log "$*"; uv run python -m "$@" >> "$LOG" 2>&1; }

log "git=$(git rev-parse --short HEAD) no3_price=$NO3_PRICE budget=$BUDGET seeds=$PPO_SEEDS"
uv run pytest -q >> "$LOG" 2>&1 || { log "TESTS FAILED"; exit 1; }

# The gate is lambda_n-independent: one file serves every frontier point (smoke gets its own).
CEIL="runs/exp1_ceiling.json"
[ "${SMOKE:-0}" = "1" ] && CEIL="runs/exp1_ceiling_smoke.json"
# Reused when its config (budget, prices, model, rendering) matches; recomputed otherwise.
run swat_gym.experiments.exp1_ceiling \
    --per-window-budget "$CEIL_PW" --workers "$WORKERS" --out "$CEIL"
gate=$(uv run python -c "import json;print(json.load(open('$CEIL'))['result']['gate_pass'])")
if [ "$gate" != "True" ] && [ "${FORCE_PPO:-0}" != "1" ] && [ "${SMOKE:-0}" != "1" ]; then
    log "FORESIGHT GATE FAILED -- stopping before PPO (FORCE_PPO=1 overrides)"; exit 2
fi

run swat_gym.experiments.exp1_controller --budget "$CTRL_BUDGET" --no3-price "$NO3_PRICE" \
    --workers "$WORKERS" --out "runs/exp1_controller${suffix}.json"
# The grower fit never sees a price: tagged runs reuse the untagged fit when it exists.
if [ -n "$suffix" ] && [ "${SMOKE:-0}" != "1" ] && [ -f runs/exp1_grower_rule.json ]; then
    run swat_gym.experiments.exp1_grower_rule --from-fit runs/exp1_grower_rule.json \
        --no3-price "$NO3_PRICE" --out "runs/exp1_grower_rule${suffix}.json"
else
    # Exit 3 = fitted rule failed the rule 5 gate; keep going.
    run swat_gym.experiments.exp1_grower_rule --evals "$GROWER_EVALS" --no3-price "$NO3_PRICE" \
        --workers "$WORKERS" --out "runs/exp1_grower_rule${suffix}.json" || log "grower exit=$?"
fi
for exp in exp1_irrigation exp2_nitrogen; do
    run "swat_gym.experiments.$exp" --budget "$BUDGET" --ppo-seeds "$PPO_SEEDS" \
        --no3-price "$NO3_PRICE" --workers "$WORKERS" --out "runs/${exp}${suffix}.json"
done
log "done"
