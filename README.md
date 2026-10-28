# swat-kimberly

A SWAT+ field model of the USDA-ARS **long-term manure (LTM) trial** at Kimberly, ID, wrapped as
a Gymnasium-style environment (`swat_gym.ltm`) for farm-management optimisation. The question:
can a learned policy maximise farm profit while limiting nitrate leaching, water, P loss/buildup
and N₂O?

- **Three levers, one experiment each:** crop rotation (Exp 1), nitrogen as manure vs synthetic
  (Exp 2), and irrigation (Exp 3, run at 55 / 80 / 90 % application efficiency).
- **Three fixed strategies:** `profit` (w = 1), `balanced` (0.5) and `sustainability` (0), each
  scored against measured trial practice. The sustainability strategy has a profit floor at
  default practice.
- **Methods:** a learned closed-loop policy (OpenAI-ES, 8 seeds) against the best open-loop plan
  (CMA-ES at a matched engine-run budget; Exp 1 also scores every rule-abiding rotation built
  from a repeating cycle of up to 4 years).
- **Contributions:** the environment (the first Gymnasium environment on official SWAT+), and an
  evaluation protocol that separates optimizer search from state-dependent adaptation.
- **Bounds on claims:**
  - Leaching is unvalidated on site.
  - N₂O is IPCC Tier 1 accounting, never simulated.
  - Crop parameters are Ai-SWAT-Plus's, copied verbatim, so each crop's yield bias matters to
    rotation results.

**Docs:**
- Runbook: `EXPERIMENTS.md`
- Inputs and their sources: `PROVENANCE.md`
- Open items and decisions: `OPEN_ITEMS.md`
- Manuscript template: `paper/`

## Model (`model/ltm/`)

- **Field:** the trial's south field, 2012–2020: 2012 pinto beans as spin-up (auto-irrigated,
  100 kg N + 50 kg P/ha at planting), then barley → sugarbeet → spring wheat → potato → barley →
  sugarbeet → spring wheat → sugarbeet.
- **HRUs:** three, one per treatment: 52A (52 Mg/ha DM manure every fall), fertiliser-only and
  control.
- **Inputs:** trial workbooks and paper tables, aggregated by `scripts/port_ltm.py` into
  `data/ltm/*.csv`. Every row carries its source. Trial measurements override SSURGO.
- **Weather:** gridMET at the LTM field's cell, 1995–2025 (`data/gridmet_ltm_1995_2025.csv`). An
  episode is an 8-year window plus spin-up; train and test windows share no calendar year.
- **Crop parameters:** Ai-SWAT-Plus's fitted `plants.plt` rows and harvest ops
  (`model/ltm/crop_params/`), never edited here. The soil-N active fraction is the one fitted
  value: it is fitted to the control's measured mineralisation (`scripts/ltm_calibrate_n.py`).
- **Engine:** SWAT+ rev **62.0.0**, a macOS Mach-O and a Linux ELF binary in `model/ltm/engine/`
  and `model/ltm/TxtInOut/`.

## Layout

- `src/swat_gym/ltm/`:
  - `practice.py`: trial seasons, crop templates, rotation rules, irrigation efficiency.
  - `render.py`: `EpisodePlan` → 9-year schedule.
  - `runner.py`: `LtmRunner`, K plans as K HRUs of one engine run, with an optional output cache.
  - `score.py`: profit, the four sustainability terms, `Reference` and the strategy score.
  - `econ.py`: prices and crop operating costs.
  - `env.py`: closed loop and observations.
  - `es.py`: ES.
  - `windows.py`: train/test windows.
- `src/swat_gym/experiments/ltm.py`: the experiment driver (`--lever R|N|I`, `--irr-eff`).
- `src/swat_gym/tests/`: `test_ltm.py` covers the LTM environment. Among its checks:
  - closed-loop episode equals open loop;
  - zero policy equals measured practice;
  - batched runs equal single runs;
  - truncated steps and the cache leave outputs unchanged;
  - rotation rules, the profit floor, and tied ES fitness.
- `scripts/` (LTM):
  - Build, in order: `port_ltm.py` → `ltm_crop_params.py` → `port_gridmet.py --site ltm` →
    `build_ltm.py` → `ltm_calibrate_n.py --apply`.
  - Checks: `ltm_validate.py`, `ltm_fit_auto_irr.py`.
  - Results page: `ltm_results.py`, `ltm_figures.py`.
- `data/`: ported measurements and weather.
- `runs/`: outputs, gitignored apart from a few tracked records.

## Setup and run

```bash
uv sync --all-extras
uv run pytest                                                     # full suite, ~1 min
SMOKE=1 uv run python -m swat_gym.experiments.ltm --lever N       # one experiment, tiny budget
uv run python -m swat_gym.experiments.ltm --lever I --budget 200000 --seeds 8 --irr-eff 0.55
uv run python scripts/ltm_results.py                              # runs/ltm_results.html
```

Full commands for all five runs are in `EXPERIMENTS.md`. Engine runs are CPU-bound. Each run
uses every core, and the five can run at once (about a day at full budgets on an Apple Silicon
laptop).

## Engine quirks (rev 62)

- A `plant` condition in `calibration.cal` only matches a community's **first** plant. The
  per-crop `phu_mat` lines therefore use the condition var `phu_plant`.
- Decision-table plant conditions resolve to plant 1 only.
- A standalone `kill` op is ignored; end crops with `hvkl`.
- `op_data3` on `plnt` is not heat units and crashes the engine; PHU comes from `days_mat`.
- `days_mat`/`yrs_mat` must be written as integers or the `plants.plt` row mis-parses.
- Irrigation applies `amt × eff × (1 − surq)`, and `hru_wb irr` reports the applied (net)
  amount. Water is scored gross (net / eff).

## Retired: GRACEnet environment

The repo began as a single-HRU corn–barley–alfalfa model of the Kimberly GRACEnet site
(irrigation and nitrogen levers, AgriMet weather). It was replaced by the LTM trial on
2026-10-25; its code remains for reproducibility:
- `model/rotation/`
- `src/swat_gym/{env,plan,schedule,rewarders}.py`
- `experiments/exp*.py`
- `scripts/run_all.sh`
- `scripts/port_*.py`

The Ai-SWAT-Plus monoculture models and the RL benchmark are in `model/crops/`, `crop*.py` and
`scripts/bench_*.py`.
