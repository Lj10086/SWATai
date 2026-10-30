# Experiment runbook (LTM environment)

How to run the three experiments, what each row of a result means, and how long it takes.
Background and claim bounds: `README.md`. Decisions and their dates: `OPEN_ITEMS.md`.

## Run

```bash
uv sync --all-extras
uv run pytest                                                         # ~30 s
SMOKE=1 bash scripts/run_ltm.sh                                       # all five, tiny budgets
perl -e 'use POSIX; setsid(); exec @ARGV' -- bash scripts/run_ltm.sh  # full run, detached, ~6 h
```

`scripts/run_ltm.sh` starts all five jobs at once, and each one uses every core:

| Job | Command it runs | Output |
|---|---|---|
| Exp 1, rotation | `--lever R` | `runs/ltm_exp1.json` |
| Exp 2, nitrogen (manure vs synthetic) | `--lever N` | `runs/ltm_exp2.json` |
| Exp 3, irrigation at 80 % efficiency (base) | `--lever I` | `runs/ltm_exp3.json` |
| Exp 3 at 55 % (furrow) | `--lever I --irr-eff 0.55` | `runs/ltm_exp3_eff55.json` |
| Exp 3 at 90 % (drip) | `--lever I --irr-eff 0.90` | `runs/ltm_exp3_eff90.json` |

The commands are `uv run python -m swat_gym.experiments.ltm ... --seeds 4 --budget 100000`.
Logs go to `runs/ltm_<job>.log`, which ends `=== ... exit <code>`. `runs/ltm_done.log` appears
when all five have finished. `SEEDS` and `BUDGET` override the defaults.

**Stage reuse.** Every intermediate result (reference runs, each ES seed, each CMA-ES search,
Exp 1's rotation table) is a stage file in `runs/ltm_exp<n>[...]/`. A stage is reused only while
its config matches exactly: model hash (every input file plus the engine), prices,
`LTM_VERSION`, ES settings, practice settings and the score's reference. Change any of these and
it recomputes. Delete a stage file to force a rerun. Per-crop reference yields are one shared
stage, `runs/ltm_crop_ref.json`.

## Budget and wall time

The budget unit is the **HRU-episode step**: one 9-year simulation of one plan
(`LtmRunner.n_hru_runs`). Each ES seed gets `--budget` steps, and CMA-ES gets the same per
strategy (CLAUDE.md rule 2).

As of 2026-10-30 the settings are 4 seeds × 3 strategies at 100k steps for every experiment
(OPEN_ITEMS L20). On a 10-core Apple Silicon Mac with all five running together:
- Exp 2 takes about 4.5 h.
- Each Exp 3 case takes about 6 h.
- Exp 1 takes a few hours from scratch, less than its budget suggests: its rotations
  repeat, so most steps are cache hits.

The v4 batch at these settings took 6.2 h.

## Strategies and the score

Three fixed strategies, w ∈ {1, 0.5, 0}: `yield`, `balanced`, `sustainability`.

    score = w · ry / ry_meas − (1 − w) · mean_k(term_k / term_k,meas)

- **ry** is relative harvested yield: each season's dry yield ÷ that crop's reference yield,
  averaged over the 8 seasons.
- **term_k** are the four sustainability terms: NO₃-N leached, gross irrigation, P loss plus
  labile-P buildup, and N₂O (IPCC Tier 1).
- **Denominators** are train-window means under measured 52A practice.
- **Yield floor.** At w = 0, an episode below default practice's train-mean ry loses 10 ×
  shortfall / ry_meas.
- **Profit** is computed and reported, never scored.

Code: `src/swat_gym/ltm/score.py`.

## Decisions per lever

| Lever | Decisions | Held at |
|---|---|---|
| R | 8: one crop per season from barl, sgbt, wwht, pota, corn (silage), alfa (a 3-year stand). No crop two years running outside a stand; potato at most once in any 3 years. | each crop's trial practice (alfa/corn: Ai-SWAT-Plus dates); 52A manure each fall except into standing alfalfa; auto-irrigation at 80 % |
| N | 53: fall manure 0–52 Mg DM/ha in each of 8 falls, plus 45 monthly synthetic N amounts (0–150 kg/ha; a dead zone below 5 % applies nothing) | measured rotation and irrigation |
| I | 40: monthly irrigation depth in every season (cap 1.5 × the trial's wettest month) | measured rotation, manure and N |

## Rows of a result file

Each `runs/ltm_exp*.json` holds, per strategy:

| Row | What |
|---|---|
| `measured` | the trial's 52A practice replayed. Its train means normalise the score. |
| `default` | measured practice passed through the lever's plan generator (`measured_x`). For R: the trial rotation under auto-irrigation. **Quote Exp 1 against this row**, since `measured` also changes the irrigation method. |
| `auto` | (I only) trial management with auto-irrigation at the case efficiency |
| `fixed` | the best open-loop plan on train. N, I: CMA-ES. R: every rule-abiding repeating cycle of up to 4 years, scored exhaustively (298 rotations). |
| `es` | the closed-loop ES policy, per seed |
| `frozen` | each ES policy's train decisions averaged (R: per-season majority), replayed open-loop |

Every row is scored on the **test** windows. Comparisons are paired per-window differences with
moving-block-bootstrap and ESS intervals (`swat_gym.stats.paired`). Quote `se_ess` or
`ci95_ess`, never `sd/√n`. Where seeds disagree in sign, report "unresolvable at this seed
count".

## Windows

An episode is a spin-up year plus 8 seasons, on gridMET weather at the field:
- Train windows start 1995–2001 and test windows start 2011–2017, 7 of each.
- Train and test share no calendar year.
- Effective sample size is 1.34 for each set (`src/swat_gym/ltm/windows.py`).

## Reports

- `scripts/ltm_nutrient_summary.py` writes `runs/ltm_nutrient_summary.html` and CSVs. They
  cover the three treatments under measured irrigation and auto-irrigation: water, applied N
  and P, leaching, biomass, uptake, and daily soil NO₃ and solution P.
- `scripts/ltm_validate.py` compares the build to the trial's measurements and writes
  `runs/ltm_validation.json`.
- `scripts/ltm_results.py` builds the results page from `runs/ltm_exp*.json`. **It still
  expects the retired "profit" strategy** and must be rewritten for the relative-yield results
  (OPEN_ITEMS L19).

## Checklist before quoting a result

- [ ] `uv run pytest` passes.
- [ ] Every job's log ends `exit 0`.
- [ ] Exp 1 is quoted against `default`, with each crop's yield bias stated next to it.
- [ ] Intervals are ESS-based. Sign-inconsistent seeds are reported as unresolvable.
- [ ] Sustainability claims state that leaching is unvalidated on site and N₂O is IPCC Tier 1,
      never simulated.
