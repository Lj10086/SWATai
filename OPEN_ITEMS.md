# Open items before submission

Working checklist for PAPER.md. Numbers are cited elsewhere; don't renumber.

1. Replace Section 3.5 with the completed nitrogen-arm result.
2. Find the journal version of the Richards et al. Magic Valley SWAT study (§1.1 cites a summary).
3. Replace §1.1 population and snowpack claims with primary citations (Census; peer-reviewed
   streamflow timing).
4. *Closed 2026-09-14.* Prices sourced in `rewarders.py`: water $0.433/mm/ha (WD01 rental +
   Schedule 24 pumping; pump head/efficiency/load factor assumed; range
   `WATER_PRICE_RANGE`), manure $7.94/Mg, application pass $37.07/ha (n = 1; report breakeven
   where pass count matters). Exp 2 artefacts before 2026-09-14 used different prices.
5. *Closed.* GRACEnet N uptake is per cutting (alfalfa) / grain only (barley); not a target.
6. Confirm whether the reference model's barley yield is grain or whole-plant.
7. *Superseded by #11.*
8. Confirm co-authors and affiliations.
9. Production cluster runs for Exp 1–2 (Exp 3/4 cut). Pre-refit gate figures (+721.5, +836,
   +2,042, +713) must not be cited; current gate is in #18.
10. *Closed 2026-07-31.* Inputs reconciled to REF HRU 000140001; `pet_co` = 0.964 fitted to ETos.
11. **Leaching column.** Percolation is threshold-behaved in applied water. Current
    (`runs/percolation_check.json`, 2026-08-28):

    | applied mm | perc mm/yr | NO₃ kg/ha/yr | mg/L |
    |---|---|---|---|
    | 2,363 (×0.6) | 0.00 | 0.00 | — |
    | 3,151 (×0.8) | 0.00 | 0.00 | — |
    | 3,939 (×1.0) | 0.00 | 0.00 | — |
    | 4,727 (×1.2) | 16.08 | 1.22 | 7.6 |
    | 5,366 (×1.4) | 76.82 | 8.28 | 10.8 |
    | 5,778 (×1.6) | 117.11 | 27.49 | 23.5 |

    Zero at measured practice and default; λ_n only separates arms that over-irrigate ≥ 20 %.
    Test windows at measured practice: 16.01 / 0.10 / 0.33 / 0.00 / 5.71 mm perc. Usable for
    within-model ranking only; not validated on site. Cause of drift from the 2026-08-06 table
    is unestablished (likely `alfa.bm_e` 12.25 → 17.0). Alfalfa soil-NO3 rise is unreachable
    (rule 13), so leaching can't be validated here; the paper must say so. The manuscript also
    needs the ET/PET engine-difference finding (rev 62 vs SWAT2012).
12. **`rsd_decomp` inert; manure organic N never enters fresh residue** (`rsd_nitorg_n` = 0).
    `fresh_manure` tagging ruled out. Likely rev 62 `fert` routes organic N straight to humus.
    Confirm against rev 62 source before attributing the mineralisation shortfall to the engine.
13. **Methodology principles** (tests in `test_stats.py`, `test_plan_env.py`): `paired()`
    reports `se_ess`, no `se`; open-loop and policy share `plan.evaluate`; `run_all.sh` reuses
    one gate and exits 2 on failure. Still binding: re-scoring a winner across λ is not
    re-selecting it.
14. *Resolved 2026-08-28.* Under workbook params corn N stress fell to 5.2 d; the test now
    asserts the annual crops collectively (barley 35.7 d) vs alfalfa 0
    (`test_annuals_are_n_stressed_and_alfalfa_is_not`).
15. **`pet_co` stays 0.964 — decided, don't reopen** without on-site crop-ET data. ET protocol
    run vs regional OpenET (`et_nostress.py`/`et_gap_check.py`, removed 2026-09-30; in git
    history; outputs `runs/et_*.json`): nitrogen
    moves ET < 0.4 %; canopy moves Kc ≤ 0.02 (LAI already saturated); ~25–40 % of the gap is
    water supply (irrigation management); no single `pet_co` fits all crops. Report in-crop ET
    11–31 % below the benchmark (21 other fields, 2020–22) as a limitation.
16. **`adaptivity_value` is unresolvable at 3 seeds.** Pre-refit per-seed: −96.4 / −344.1 /
    +890.1 (mean +149.8 from one seed). `advantage_over_fixed` −328.7 / −632.2 / −193.8 (mean
    −384.9) is sign-consistent: PPO loses to matched-budget CMA. Don't average the adaptivity
    decomposition; estimate the seeds needed from this spread before more cluster time.
    Re-check both on the refit model (#18). **Confounded until rerun (2026-10-23):** those PPO
    runs started from zero water (mean action 0.055, 48 % of actions clipped to 0) while CMA-ES
    started at measured practice, and the renderer let a later month's water land in an
    earlier, already-observed month. Both are fixed (residual actions; no carry-back); don't
    cite these numbers as evidence about PPO.
17. *Closed 2026-09-28.* PAPER.md has no Exp 3/4 sections; Exp 1 numbers there are pre-refit
    (#18).
18. **All experiments must be (re-)run: `scripts/run_all.sh`.** Model refit 2026-09-10
    (`PARAMS` = N cycle only; PROVENANCE §6); prices sourced 2026-09-14; code simplified
    2026-09-30 (Exp 2 now monthly, no N cap). The 2026-09-11 ceiling (+360.8 $/ha, ESS CI
    [220, 502], `gate_pass: true`), controller and Exp 1 CMA run used pre-09-14 prices and are in
    `runs/archive/pre_price_20260914/`; pre-refit copies in `runs/archive/pre_refit_20260910/`.
    Paired differences (signs) should survive; $/ha levels will not.
19. *Closed 2026-09-28.* `runs/calibration.json` is tracked; `optimize.py` refuses to
    overwrite `--out` without `--overwrite`.
20. **Grower rule** (`exp1_grower_rule`): controller fitted to the logs; a human bar that
    responds to weather. In `run_all.sh`. To do: run (#18), fill Table 3, disclose in §6.2
    (test-period weather, not fitted to profit, reference row), report fit R² and per-year totals.
21. *Closed 2026-10-23.* PAPER.md §3–4 match the code: no N cap or repair, channel 10, the
    nitrogen arm's action, 40/20 steps, prices ($7.94, $37.07, final 2024 NASS), ESS 1.25 and
    the moving-block bootstrap (now implemented), restored A2 tests, the equal-search ceiling and
    the controller's engine-run budget. Results sections (§6) still quote pre-fix numbers until
    the rerun (#18).
22. **Crop models (`model/crops`, `swat_gym.crop`, `scripts/bench_rl.py`).** Ai-SWAT-Plus
    monoculture builds, copied; their yields are not validated against Kimberly, so quote
    paired differences only. Sugarbeet price assumes 23 % root dry matter (unsourced). Report
    any batching use with `tests/test_batch.py` passing: SWAT+ rev 62 keeps N/P demand per
    plant, not per HRU (`pl_nut_demand.f90`; it also resets `uapd(ipl)` with a stale loop
    index), so alfalfa breaking dormancy reads the previous HRU's demand. The same stale value
    exists in single-HRU runs (same HRU, previous day): small, but a real engine bug worth
    reporting upstream and fixing in the Ai-SWAT-Plus engine build.

## Moved to supplementary material (must exist before submission)

- Input-reconciliation table (10 rows: `orgn_min`, soil profile, initial NO3/P, barley HI,
  solar, wgn station, bulk density, manure incorporation, object areas, sim window).
- Morris screen: 17 params, bounds, four dropped as inert (`n_uptake`, `n_perc`, corn
  `tmp_opt`, `esco`).
- Before/after reconciliation table (PET, ET, per-crop yield, NO₃ leached).
- Engine quirks: standalone `kill` ignored (use `hvkl`); `op_data3` on `plnt` is not PHU and
  crashes the engine.
- Runner correctness properties and regression tests.
- Object-area fix made every path ~3× faster (engine was routing a 1,432 ha network).
