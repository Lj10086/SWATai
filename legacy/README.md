# legacy/: retired code, models and data

Nothing here is used by the LTM environment, and nothing here runs from this tree. Everything
was moved on 2026-10-30, unchanged. **To run it, check out commit `c37710e`**, the last commit
before the move, where every path below sits at its original location (drop the `legacy/`
prefix).

## The GRACEnet environment (retired 2026-10-25)

This is the repo's first environment. It is a single-HRU corn → barley → alfalfa model of the
Kimberly GRACEnet field, 2013–2019, with two levers (monthly irrigation, nitrogen), AgriMet and
later gridMET weather, PPO against CMA-ES, and profit-based rewards. The LTM trial replaced it
(user decision 2026-10-25).

- `model/rotation/`: the model. `model/archive/` holds its earlier snapshots.
- `model/TxtInOut/`: the user's copy of the Ai-SWAT-Plus base template.
- `model/aiswat_fit/`, `model/lib/`, `model/swatplus_rev60.5.7`: older fitted values and the
  rev 60 engine.
- `src/swat_gym/`:
  - `plan.py`, `schedule.py`: lever vector → management schedule.
  - `env.py`: the monthly Gymnasium env.
  - `rewarders.py`: profit. Its price code now lives in `swat_gym/ltm/econ.py`.
  - `params.py`
  - `windows.py`
  - `experiments/exp1_*.py`, `experiments/exp2_nitrogen.py`
  - `experiments/common.py`: the full version, with PPO.
- `scripts/`: model ports (`port_*.py`), calibration (`calibrate/`, `calibrate_petco.py`),
  checks and the pipeline `run_all.sh`.
- `data/`: GRACEnet measurements and AgriMet/gridMET series at the GRACEnet point.
- `runs/`: three tracked records cited by that code (`calibration.json`,
  `exp1b_price_ratio.json`, `exp1c_cadence_e500.json`).
- `PROVENANCE.md`: where every input of that model came from.
- `EXPERIMENTS.md`: its runbook.

## The crop-model RL benchmark

This benchmark chose ES over PPO and a GA for the LTM experiments (`swat_gym/ltm/es.py` cites
it). It ran 5 Ai-SWAT-Plus single-crop models × 2 levers.
- Code: `src/swat_gym/{spec,crop,crop_env}.py`, `experiments/{crops,crops_ga}.py`,
  `scripts/bench_{rl,ga}.py`, and the tests `test_batch`, `test_crop`, `test_ga`.
- Its results (`runs/bench_*_xl.json`) are gitignored outputs.
- The crop models themselves are still in use, so they stay at `model/crops/`.

## What the LTM code took from here

These were copied into the LTM code, so it no longer imports anything from this directory:
- the nitrogen dead-zone mapping → `ltm/practice.py`
- prices → `ltm/econ.py`
- IPCC N₂O → `ltm/score.py`
- window statistics and `paired` → `swat_gym/stats.py`
- CMA-ES and stage reuse → `experiments/common.py`
- the ES weight initialisation → `ltm/es.py`
- the alfalfa and corn-silage monthly irrigation, computed once by `crop_env.default_x` →
  `model/ltm/exp1_irr_monthly.json`

LTM outputs were byte-identical before and after the move.
