# swat-kimberly

This repository provides a **reinforcement-learning environment built on a calibrated SWAT+
model of an irrigated field at Kimberly, Idaho**. The model represents the south field of the
USDA-ARS long-term manure (LTM) trial, 2012–2020, and is run through the official SWAT+ engine
(rev 62.0.0). The environment (`swat_gym.ltm`) exposes crop rotation, nitrogen management and
irrigation as decision levers, and scores each management plan on crop yield and on four
sustainability costs: nitrate leaching, irrigation water, phosphorus loss and buildup, and N₂O.

The research question is whether a learned, state-dependent management policy can outperform
both the trial's own practice and the best fixed (open-loop) plan found at the same simulation
budget. The evaluation protocol separates the two effects: the gain from search alone, and the
additional gain from adapting to the simulated soil state.

## Installation and usage

Requires [uv](https://docs.astral.sh/uv/). Engine runs are CPU-bound, and each job uses all
available cores.

```bash
uv sync --all-extras
uv run pytest                                                         # ~30 s
SMOKE=1 bash scripts/run_ltm.sh                                       # all experiments, small budgets
perl -e 'use POSIX; setsid(); exec @ARGV' -- bash scripts/run_ltm.sh  # full run, ~6 h, detached
uv run python scripts/ltm_nutrient_summary.py                         # water and nutrient report
```

### Rebuilding the model

Each step is idempotent, and its outputs are committed.

```bash
uv run python scripts/port_ltm.py           # trial workbooks -> data/ltm/*.csv
uv run --project ~/Desktop/projects/Ai-SWAT-Plus/calibration python scripts/ltm_crop_params.py
uv run python scripts/port_gridmet.py       # gridMET at the field (cached download)
uv run python scripts/build_ltm.py          # -> model/ltm/TxtInOut
uv run python scripts/ltm_calibrate_n.py --apply && uv run python scripts/build_ltm.py
uv run python scripts/ltm_validate.py       # comparison with measurements -> runs/ltm_validation.json
uv run python scripts/ltm_fit_auto_irr.py   # auto-irrigation trigger (management, not calibration)
```

`port_ltm.py` reads the collaborator's workbooks and `ltm_crop_params.py` reads the
Ai-SWAT-Plus repository; neither source is included here.

## Model

- **Engine.** SWAT+ rev 62.0.0, run as a black box: the code writes input files, runs the
  engine and reads its outputs (`swat_gym.fastrunner`).
- **Site and period.** South field of the LTM trial, Kimberly, Idaho. The 2012 pinto-bean crop
  is spin-up, followed by eight seasons: barley, sugarbeet, spring wheat, potato, barley,
  sugarbeet, spring wheat, sugarbeet.
- **Treatments.** One hydrologic response unit (HRU) per treatment:
  - 52A: 52 Mg/ha dry-matter dairy manure applied each fall;
  - fertiliser only;
  - unamended control.
- **Inputs.** Trial measurements (`data/ltm/`, each row with its source) override SSURGO
  defaults. Weather is gridMET at the field, 1995–2025.
- **Calibration.**
  - Crop parameters are those fitted in the companion project Ai-SWAT-Plus, copied without
    modification.
  - One soil-nitrogen parameter (the active-humus fraction) is fitted to the control
    treatment's measured mineralisation.
  - No other parameter is calibrated. `scripts/ltm_validate.py` compares the model with the
    trial's measurements.

## Environment and experiments

An episode replays the trial's eight-season management on a different nine-year stretch of
weather. Training and test episodes share no calendar year (training windows start
1995–2001, test windows 2011–2017).

Each experiment varies one lever and holds the others at trial practice:

| Experiment | Lever | Decisions |
|---|---|---|
| 1 | Crop rotation | crop per season from barley, sugarbeet, spring wheat, potato, corn silage, alfalfa |
| 2 | Nitrogen | fall manure rate and monthly synthetic N |
| 3 | Irrigation | monthly depth, at 55, 80 and 90 % application efficiency |

Plans are scored under three fixed strategies with weight w ∈ {1, 0.5, 0} (`yield`,
`balanced`, `sustainability`):

    score = w · ry / ry_meas − (1 − w) · mean_k(term_k / term_k,meas)

where ry is relative harvested yield and term_k are the four sustainability costs, each
normalised by measured practice. Profit is reported but not scored.

Each strategy compares two methods:
- a closed-loop policy trained with OpenAI evolution strategies (ES), which observes the
  simulated soil state;
- the best open-loop plan found by CMA-ES at an equal engine budget. Experiment 1 also
  evaluates every admissible rotation exhaustively.

Run settings, result rows and reporting rules are documented in `EXPERIMENTS.md`.

## Limitations

The following limits apply to every result and should be stated alongside it:
- **Crop parameters** are not fitted to this site. Potato is under-predicted, and spring wheat
  uses winter-wheat parameters. Crop-specific yield bias therefore affects rotation results.
- **Nitrate leaching** has not been validated against on-site measurements.
- **N₂O** is estimated with IPCC Tier 1 emission factors. SWAT+ does not simulate it.

## Repository structure

| Path | Contents |
|---|---|
| `src/swat_gym/ltm/` | environment: practice templates, schedule rendering, batched engine runner, score, prices, closed-loop environment, ES, train/test windows |
| `src/swat_gym/experiments/ltm.py` | experiment driver (`--lever R\|N\|I`, `--irr-eff`) |
| `src/swat_gym/{fastrunner,engine,manifest,printprt}.py` | repeated SWAT+ execution in a temporary copy of the model |
| `src/swat_gym/stats.py` | paired comparisons over overlapping weather windows (effective sample size, block bootstrap) |
| `src/swat_gym/tests/test_ltm.py` | environment tests |
| `model/ltm/TxtInOut/` | built SWAT+ model and engine binaries (macOS, Linux) |
| `model/ltm/crop_params/` | per-crop Ai-SWAT-Plus parameters |
| `model/crops/` | Ai-SWAT-Plus single-crop models; the barley model is the LTM build's base |
| `data/ltm/` | trial measurements, one source per row |
| `data/gridmet_*.csv` | cached gridMET weather |
| `scripts/` | model build, validation, reports and the run script |
| `runs/` | outputs (not tracked) |
| `paper/` | manuscript template (Elsevier CAS) |
| `legacy/` | retired GRACEnet environment and crop-model benchmark (`legacy/README.md`) |

Documentation:
- `EXPERIMENTS.md`: running the experiments, budgets and result rows.
- `PROVENANCE.md`: sources of every model input.
- `OPEN_ITEMS.md`: dated design decisions and open issues.
- `ARTIFACTS.md`: index of published report pages.

## Tests

`src/swat_gym/tests/test_ltm.py` verifies that:
- a closed-loop episode equals the open-loop run of its final plan, for every score term;
- the zero policy reproduces measured practice;
- K plans batched as K HRUs equal K single runs, including alfalfa;
- truncated runs and the output cache do not change results;
- the ES loop survives an engine crash;
- irrigation water is counted gross of application efficiency;
- auto-irrigation stays within its windows;
- the rotation rules agree with the action mask;
- the yield floor applies only to the sustainability strategy;
- tied ES fitness produces no gradient.

## SWAT+ rev 62 behaviour relied on

- A `plant` condition in `calibration.cal` matches only a community's first plant, so the
  per-crop `phu_mat` lines use the condition variable `phu_plant`.
- Decision-table plant conditions also resolve to the first plant only.
- A standalone `kill` operation is ignored; crops are ended with `hvkl`.
- `op_data3` on `plnt` is not heat units and crashes the engine; PHU is set from `days_mat`.
- `days_mat` and `yrs_mat` must be written as integers, or the `plants.plt` row is misparsed.
- Irrigation delivers `amt × eff × (1 − surq)` to the soil, and `hru_wb irr` reports this net
  amount. Water is scored gross (net ÷ eff).
- SWAT+ reports soil NO₃ and solution P only as whole-profile totals. Daily values are taken
  from 0-kg fertiliser "probe" operations, whose management-output rows record the soil state.
