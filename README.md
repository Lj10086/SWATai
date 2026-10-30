# swat-kimberly

A SWAT+ model of one field of the USDA-ARS **long-term manure (LTM) trial** at Kimberly, Idaho,
wrapped as an optimisation environment (`swat_gym.ltm`).

**The question.** Can a learned management policy maximise **crop yield** while limiting four
sustainability costs?
- nitrate leaching
- irrigation water
- phosphorus loss and buildup
- N₂O

The policy is compared with the trial's own practice and with the best fixed plan.

## In one minute

- **SWAT+** is a process-based agro-hydrology model: daily weather, soil, crops and management
  in; yields, water and nutrient flows out. This repo runs the official rev 62.0.0 engine as a
  black box: it writes input files, runs the engine, and reads the outputs.
- **The field.** The trial's south field, 2012–2020. 2012 pinto beans are spin-up, then barley →
  sugarbeet → spring wheat → potato → barley → sugarbeet → spring wheat → sugarbeet. There are
  three treatments, one HRU (simulated unit) each:
  - 52A: 52 Mg/ha DM dairy manure every fall
  - fertiliser-only
  - control
- **An episode** replays that 8-season management on a different 9-year stretch of weather
  (gridMET, 1995–2025). Train and test episodes never share a calendar year.
- **Three experiments, one lever each:**
  - Exp 1: crop rotation
  - Exp 2: nitrogen, manure vs synthetic
  - Exp 3: monthly irrigation, at 55 / 80 / 90 % application efficiency
- **Three strategies**, each a fixed weighting: `yield` (w = 1), `balanced` (0.5) and
  `sustainability` (0).
  - score = w · relative yield − (1 − w) · mean sustainability cost.
  - Both parts are normalised by measured practice.
  - Profit is reported but not scored.
- **Methods compared per strategy:**
  - a closed-loop policy trained with OpenAI-ES, which reacts to the simulated soil state;
  - the best open-loop plan, from CMA-ES at the same simulation budget (Exp 1 also scores every
    rotation exhaustively).

  The gap between the two measures whether reacting to the soil state helps.

**What results can claim.** These bounds hold for every result; state them alongside any result:
- **Crop parameters** come from the sister project Ai-SWAT-Plus, copied verbatim and never
  fitted here. Potato under-yields, and spring wheat uses the winter-wheat parameters. Each
  crop's yield bias therefore matters to rotation results.
- **Leaching** is not validated on site.
- **N₂O** is IPCC Tier 1 accounting. SWAT+ does not simulate it.
- **Calibration.** One soil-N parameter is fitted, to the control's measured mineralisation.
  Nothing else is calibrated.

## Repository map

| Path | What |
|---|---|
| `src/swat_gym/ltm/` | the environment: practice templates, schedule rendering, the batched engine runner, the score, prices, the closed-loop env, ES, train/test windows |
| `src/swat_gym/experiments/ltm.py` | the experiment driver (`--lever R\|N\|I`, `--irr-eff`) |
| `src/swat_gym/{fastrunner,engine,manifest,printprt}.py` | running the SWAT+ engine repeatedly in a temp copy of a model |
| `src/swat_gym/stats.py` | paired comparisons over overlapping weather windows (ESS, block bootstrap) |
| `src/swat_gym/tests/test_ltm.py` | the environment's tests (below) |
| `model/ltm/TxtInOut/` | the built SWAT+ model and engine binaries (macOS, Linux) |
| `model/ltm/crop_params/` | each crop's Ai-SWAT-Plus parameters (`scripts/ltm_crop_params.py`) |
| `model/crops/` | Ai-SWAT-Plus single-crop models: the barley one is the LTM build's base |
| `data/ltm/` | the trial's measurements, one source per row (`scripts/port_ltm.py`) |
| `data/gridmet_*.csv` | cached gridMET weather (`scripts/port_gridmet.py`) |
| `scripts/` | model build, validation, reports and the run script (below) |
| `runs/` | outputs (gitignored) |
| `paper/` | manuscript template (Elsevier CAS) |
| `legacy/` | the retired GRACEnet environment and crop-model benchmark (see `legacy/README.md`) |

Docs:
- `EXPERIMENTS.md`: how to run, rows and budgets.
- `PROVENANCE.md`: where every model input came from.
- `OPEN_ITEMS.md`: decisions, with dates and reasons, and what is still open.

## Setup and run

Requires [uv](https://docs.astral.sh/uv/). Engine runs are CPU-bound, and every job uses all
cores.

```bash
uv sync --all-extras
uv run pytest                                                         # ~30 s
SMOKE=1 bash scripts/run_ltm.sh                                       # every experiment, tiny budgets
perl -e 'use POSIX; setsid(); exec @ARGV' -- bash scripts/run_ltm.sh  # full run, ~6 h, detached
uv run python scripts/ltm_nutrient_summary.py                         # water/nutrient report
```

### Rebuilding the model

Each step is idempotent, and the outputs are committed.

```bash
uv run python scripts/port_ltm.py           # trial workbooks -> data/ltm/*.csv
uv run --project ~/Desktop/projects/Ai-SWAT-Plus/calibration python scripts/ltm_crop_params.py
uv run python scripts/port_gridmet.py       # gridMET at the field (cached download)
uv run python scripts/build_ltm.py          # -> model/ltm/TxtInOut
uv run python scripts/ltm_calibrate_n.py --apply && uv run python scripts/build_ltm.py
uv run python scripts/ltm_validate.py       # vs measurements -> runs/ltm_validation.json
uv run python scripts/ltm_fit_auto_irr.py   # auto-irrigation trigger (management, not calibration)
```

`port_ltm.py` reads the collaborator's workbooks, and `ltm_crop_params.py` reads the
Ai-SWAT-Plus checkout. Neither source is in this repo.

## What the tests pin

`src/swat_gym/tests/test_ltm.py` checks:
- a closed-loop episode equals the open-loop run of its final plan, for every score term;
- the zero policy replays measured practice;
- K plans batched as K HRUs equal K single runs, alfalfa included;
- truncated runs and the output cache change nothing;
- an engine crash during ES is survived;
- water is counted gross of efficiency;
- auto-irrigation stays inside its windows;
- the rotation rules and their action mask agree;
- the yield floor binds only the sustainability strategy;
- tied ES fitness gives no gradient.

## Engine quirks (rev 62)

- A `plant` condition in `calibration.cal` only matches a community's first plant. The per-crop
  `phu_mat` lines therefore use the condition variable `phu_plant`.
- Decision-table plant conditions also resolve to plant 1 only.
- A standalone `kill` op is ignored. End crops with `hvkl`.
- `op_data3` on `plnt` is not heat units, and it crashes the engine. PHU comes from `days_mat`.
- `days_mat`/`yrs_mat` must be written as integers, or the `plants.plt` row mis-parses.
- Irrigation puts `amt × eff × (1 − surq)` into the soil, and `hru_wb irr` reports that net
  amount. Water is scored gross (net ÷ eff).
- SWAT+ reports soil NO₃ and solution P only as whole-profile totals. Daily values come from
  0-kg fertiliser "probe" operations, whose management-output rows carry the soil state.
