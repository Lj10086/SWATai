# `swat-gym` — experiment runbook

Agent entrypoint: [`CLAUDE.md`](CLAUDE.md). Code: `src/swat_gym/`. Rotation is fixed.

## Run

```bash
SMOKE=1 bash scripts/run_all.sh          # tiny budgets, ~4 min: checks everything runs
perl -e 'use POSIX; setsid(); exec @ARGV' -- bash scripts/run_all.sh     # production
NO3_PRICE=16 OUT_TAG=n16 bash scripts/run_all.sh                         # a frontier point
```

`run_all.sh` order: pytest → `exp1_ceiling` → `exp1_controller` → `exp1_grower_rule` →
`exp1_irrigation` → `exp2_nitrogen`. It stops before PPO if the ceiling gate fails
(`FORCE_PPO=1` overrides). The ceiling is λ_n-independent and reused while its config matches.
Env overrides: `BUDGET` (engine runs per method; the controller's `CTRL_BUDGET` defaults to it),
`PPO_SEEDS` (trained in parallel), `WORKERS` (default: all cores), `CEIL_PW` (CMA-ES evals per
oracle and for the shared schedule), `GROWER_EVALS`.

Outputs in `runs/`: `exp1_{ceiling,controller,grower_rule,irrigation}.json`,
`exp2_nitrogen.json`, plus reusable stages `<exp>_fixed.json`, `<exp>_ppo_s<seed>.json` and PPO
checkpoints. Delete a stage file to force it to rerun; it also reruns if its config changes.

## Levers

| Lever | Vector (unit box) | Env steps | Held at measured practice |
|---|---|---|---|
| `I` | 42 monthly depths × 200 mm | 40 (barley Septembers are post-harvest) | manure, N |
| `N` | 4 annual-crop years × (6 monthly mineral N × 150 kg + April manure × 90 Mg) | 20 (months whose 15th precedes harvest) | irrigation |

PPO acts in residuals around `default_x` (measured practice), so it starts where CMA-ES does.

## Rows

| Row | Meaning |
|---|---|
| `measured` | shipped `management.sch` (2013–19 log) replayed on each window |
| `default` | measured practice through the plan generator (quote lever gains vs this) |
| `fixed` | CMA-ES open-loop plan, trained on train windows |
| `policy` | PPO (≥ 3 seeds; representative = median seed) |
| `frozen` | the policy's best train-window plan, replayed open-loop on test |
| `ceiling` (Exp 1) | per-window oracle minus shared plan, both fitted on test with equal CMA-ES evals; gate at 250 $/ha |
| `controller` (Exp 1) | 3-param feedback rule fitted by CMA-ES on profit, same engine-run budget |
| `grower` (Exp 1) | same rule fitted to logged monthly depths (reference row, rule 5 gated) |

`exp*_*.json` records per-seed `advantage_over_fixed` and `adaptivity_value` plus
`sign_consistent`; if signs differ, report "unresolvable at this seed count".

## Windows (`src/swat_gym/windows.py`)

Train starts 1995–2004 (10), test starts 2013–2017 (5), zero shared calendar years, ESS 1.25.

## Crop models (`scripts/bench_rl.py`)

```bash
uv run python scripts/bench_rl.py            # 5 crops × levers (9 cases), 3 seeds, 20k each
uv run python scripts/bench_rl.py --smoke    # ~2.5 min
```

Per case: `default` (the model's own decision tables), `default_x` (their train-mean monthly
amounts, open loop), `fixed` (CMA-ES), `policy` (PPO, per seed), `frozen`. Windows are a
spin-up year + a scored season: train seasons 1996–2014, test 2016–2025, ESS 5.26. Budgets are
candidate-seasons. Stages live in `runs/bench_rl/` and are reused while their config matches.
Annual crops run K candidates as K HRUs of one engine run (exact; `tests/test_batch.py`);
alfalfa runs unbatched.

## Cluster (`instgpu-01`…`05`)

CPU-bound; Linux ELF engine. One `run_all.sh` per node per frontier point (`OUT_TAG`);
`WORKERS` defaults to all cores.

## Checklist before full runs

- [ ] `scripts/check_param_state.py` passes (rule 11b); `uv run pytest` passes
- [ ] `SMOKE=1 bash scripts/run_all.sh` completes
- [ ] Ceiling `gate_pass` true (else write the bounded null)
- [ ] Intervals quoted as `se_ess`
- [ ] Full λ_n frontier produced; leaching column non-degenerate, or say it isn't
- [ ] Sustainability claims state: leaching unvalidated on site; N₂O is Tier 1, unpriced

## Removed code (recover from git history)

- 2026-09-30 simplification: annual `SwatEnv`/`_focused.py`/`constrainers.py` stack (Exp 2 now
  uses the monthly env), N cap (`--max-n`), `obsnorm.py`, `frontier.py`, `swat_kimberly`
  runner, and finished diagnostics (`monoculture`, `et_gap_check`, `et_nostress`,
  `event_size_check`, `verify_model_state`, `calibrate/screen`, `check_source_data`,
  `exp1_strategy_table`, `rerun_exp1.sh`).
- 2026-09-09 scope cut (correct; safe to restore): `exp3_rotation.py`, `exp4_joint.py`
  (restore rule 8 with them).
- 2026-08-07 leaky split (re-point to `swat_gym.windows` first): `exp2_rl.py`,
  `exp2b_controls.py`.
- `runs/exp1b_price_ratio.json`, `runs/exp1c_cadence_e500.json`: tracked results of retired
  sweeps; keep.
