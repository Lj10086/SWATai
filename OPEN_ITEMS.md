# Open items before submission

Working checklist for PAPER.md. Numbers are cited elsewhere; don't renumber.

## LTM environment (2026-10-25; the paper's environment from now on)

L1. *Closed 2026-10-25.* Potato tuber DM 0.214 = 1 − water 78.6 g/100 g, USDA-ARS FoodData
    Central SR Legacy FDC 170027 "Potatoes, russet, flesh and skin, raw" (Foundation FDC 2346401
    agrees). Potato $1,070/Mg DM. It also gives a check: measured 2016 potato 69.5 Mg/ha fresh ×
    0.214 = 14.9 t DM vs 10.6 simulated (−29 %, the Ai-SWAT-Plus canopy problem, L5).
L2. *Closed 2026-10-26.* Fertilizer P from Leytem et al. 2024 Table S1 (supplement): 52A and
    control none; fertilizer plots 39/151/19/98/16/38/20/59 kg P/ha (2013–2020). S1's 2014 (151)
    disagrees with Leytem et al. 2025 Table 3 (66); S1 is this field's record.
L3. *Closed 2026-10-25.* Crop operating costs escalated to mean 2022–2024 dollars with USDA
    NASS's index of prices paid by farmers (all farms, 2011=100; annual means 2017 106.93, 2019
    110.26, 2022–24 139.32): ×1.264 (2019 budgets), ×1.303 (potato, 2017). `ltm/econ.py`.
L4. **Soil N cycle: one parameter fitted** (active-humus fraction f = fr_hum_act =
    n_act_frac) to the control's measured mineralisation (72 kg N/ha, non-residue years). The
    rate `orgn_min` is kept at 0.0008: at its 0.006 ceiling the default pool gives 33 of 72.
    Residue/manure mineralisation is never reported by rev 62 (`nut_nminrl.f90`; cf. #12), so
    manured plots have no comparator; simulated is whole-profile vs 0–60 cm bags.
L5. **Spring wheat runs on Ai-SWAT-Plus `wwht` parameters** (user decision); its calendar is
    already spring (emerge 3/1). Potato runs on Ai-SWAT-Plus `pota` as-is (Ai-SWAT-Plus flagged
    a failed canopy; user decision: copy and flag).
L6. **Soil hydraulics are SSURGO** (mukey 80295: AWC 0.20/0.18, Ksat), texture/BD/OM/chemistry
    measured. The only measured water retention is from four manure-treated plots in 2016 (AWC
    ~0.12, 0–30 cm) and does not represent the starting soil.
L7. *Closed 2026-10-25.* IPCC factors verified against the 2019 Refinement Vol. 4 Ch. 11
    Tables 11.1/11.3 (aggregated Tier 1: EF1 0.010, EF4 0.010, EF5 0.011, FracGASF 0.11,
    FracGASM 0.21; `rewarders.py`). Report as sensitivity: Kimberly is "dry" (P/PET < 1), where
    the disaggregated EF1 and EF4 are 0.005 (direct N2O roughly halves).
L8. **8-year episodes give ESS 1.34** on 7 test windows (user decision). Quote ESS intervals.
L9. *Decided 2026-10-27 (user), replacing 2026-10-25's "no rules":* with no rules Exp 1 grew
    potato 8 years running (`runs/ltm_exp1.json`, LTM_VERSION 2), which SWAT+ cannot penalise (no
    soilborne disease/nematode module). Rules now: no crop two years running except inside one
    3-yr alfalfa stand (a stand may not follow a stand); potato at most once in any 3 years
    ("potatoes followed by two years of grain", the most common Idaho rotation, 38 % of growers:
    Patterson & Bohl 1995, UI A.E. Extension Series 95-4; "not more than one crop of potatoes in
    succession": McDole & Dallimore 1973, UI Current Information Series 200). Sugarbeet gets the
    no-repeat rule only: the trial grows it in 2018 and 2020. Exp 1 searches 298 cycles (≤ 4 yr).
    Still a limitation: within the rules, yield does not respond to the rotation's disease load.
L10. **52A over-mineralises after the soil-N fit** (humus-only mineralisation > measured total
    in 5 of 5 non-residue years; `runs/ltm_validation.json`): manure-plot leaching is an upper
    bound. The manure split `rtof` is not identifiable from reported fluxes.
L11. *Closed 2026-10-25; refitted 2026-10-27.* Auto-irrigation = one 25.4 mm (gross) pass
    whenever profile plant-available water < sw_frac × its FC value, window = first measured
    event .. 7 days before harvest; sw_frac root-found at efficiency 1.0 so the rule applies the
    trial's 4,542 mm (2013–2020) under its rotation (`scripts/ltm_fit_auto_irr.py` →
    `model/ltm/auto_irr.json`): 0.7236, 0.7523 after the spin-up beans (L14). At 80 % efficiency
    the same rule pumps more. Found on the way: Ai-SWAT-Plus's `build_irrigation_dtl` writes
    lim_op "-" (target FC − 0.95 mm, i.e. irrigate whenever below FC), not "0.95 × FC" as its
    docstring says; its crop fits ran under that. Raise it in Ai-SWAT-Plus.
L12. *Decided 2026-10-25 (user).* Manure on the collaborator's target-rate basis, which is 1.5–25 %/yr
    below Leytem et al. 2024 Table 2's applied N (and P vs the 2025 paper). State it with Exp 2.
L13. *Decided 2026-10-27 (user).* **Irrigation efficiency.** SWAT+ puts amount × efficiency in
    the soil; water is costed and scored gross. Cases from Irmak et al. 2011 (UNL Extension EC732)
    Table 2 midpoints: 0.55 furrow, 0.80 center pivot / solid set, 0.90 surface drip. The
    measured replay stays at 1.0: at 0.8 the trial's own water never drains the profile in
    2013–2020 (percolation 0, leaching 0, the score's NO₃ reference would be 0) and yields fall
    11–46 %. Auto-irrigation and every generated schedule run at the case efficiency: 0.80 in
    Exp 1 and the Exp 3 base, 0.55 and 0.90 as Exp 3 cases (`ltm_exp3_eff55`, `_eff90`).
L14. *Decided 2026-10-27 (collaborator setup, G. Richards email).* Spin-up year = pinto beans,
    auto-irrigated, 100 kg/ha N + 50 kg/ha P at planting (date: user decision, Jul 1), in the
    validation build (2012) and in every episode's spin-up year. Raises 2013–2020 leaching
    (52A 1,190 → 1,389, control 226 → 394 kg N/ha); validation otherwise ~unchanged
    (`runs/ltm_validation.json`).
L15. **2020 sugarbeet P uptake.** The collaborator corrected his SWAT run's 56.5 to 68.5 kg P/ha;
    it is a simulated value (no 2020 sugarbeet plant data from the trial), so it is not used as a
    target (rule: SWAT-output columns in the workbooks are not ported).
L16. *Decided 2026-10-27 (user).* Budgets halved: Exp 1 50k, Exp 2/3 100k HRU-episode steps per
    ES seed (every policy had settled by then in the 200k runs: Exp 1 by its first checkpoint).
    CMA-ES matched. *Reversed 2026-10-28 (user):* full budgets restored (Exp 1 100k, Exp 2/3
    200k) with 8 seeds, since the v3 batch finished in ~6 h (`runs/archive/ltm_v3_20261028/`).

L17. *Decided 2026-10-28 (user).* Sustainability profit floor. In v3, every Exp 3 efficiency
    case's w = 0 optimum was no irrigation (crops fail, −1,921 $/ha/yr) because w = 0 values
    profit at nothing. The floor is default practice's train-mean profit per lever and
    efficiency case. Each episode below it loses `FLOOR_PENALTY` (10) × shortfall / reference
    profit, graded so CMA-ES can steer back. It applies on every path, test scoring included
    (`Reference.score`; `test_profit_floor_only_binds_sustainability`).

L18. *Fixed 2026-10-28.* ES v1 defects, found in the v3 results:
    (a) `crops_ga.centered_ranks` ranks ties by list order. A population tied on a plateau
        then takes full Adam steps along a fake gradient. In Exp 2 every policy reached "no
        manure, no N" by generation ~16, sat there with median = max fitness, and all 4 seeds
        ended identical, $78/ha/yr below the fixed plan. LTM ES now averages tied ranks
        (`test_tied_fitness_gives_no_gradient`). The crop benchmark keeps the old function so
        its cited results stay reproducible.
    (b) ES scored 3 random train windows per generation while CMA-ES scores all 7, so ES
        optimised a noisy subset of the same objective. Its Exp 3 curves peaked mid-run and
        drifted down, and the policy trailed the fixed plan by $50–75/ha/yr on every seed. ES
        now uses all 7 (about 44 generations at 200k for Exp 3).

