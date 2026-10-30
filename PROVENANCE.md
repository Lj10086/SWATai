# Where every input of the LTM model came from

Every value in `model/ltm/TxtInOut` traced to the source it was ported from and the script that
did it. Each row of `data/ltm/*.csv` also names its own source. Bracketed tags ([W1]–[W4], [P1],
[P2]) are defined under *Sources* below. The paper (`PAPER.md`) cites values from here.

Build order (each step idempotent): `scripts/port_ltm.py` → `scripts/ltm_crop_params.py` →
`scripts/port_gridmet.py` → `scripts/build_ltm.py` → `scripts/ltm_calibrate_n.py --apply`
(then rebuild) → `scripts/ltm_validate.py`.

## The LTM scenario (built 2026-10-25)

`model/ltm/TxtInOut`, built by `scripts/build_ltm.py` from `data/ltm/` (`scripts/port_ltm.py`).
The retired GRACEnet rotation's record is `legacy/PROVENANCE.md`; it does not bind this model.

**Sources** (all in `~/Documents/Kimberly, Idaho/USDA Long-Term Manure/`): the collaborator's
workbooks *LT Manure Data Summary.xlsx* [W1], *LT Manure Properties & P adjustment.xlsx* [W2],
*LT Manure Soil Properties and P uptake data.xlsx* [W3]; Leytem et al. 2024, SSSAJ 88(1):71–88,
doi:10.1002/saj2.20599 [P1] (this field's rotation and N); Leytem et al. 2025, Crop Science
(sugarbeet years of both adjacent fields) [P2]. Every row of `data/ltm/*.csv` names its source.

**Precedence.** Manure nutrients are on the collaborator's **target-rate** basis ([W4]
*Long Term Manure Project Input Data.xlsx*, received 2026-10-25: 5,200 kg/ha × fractions, i.e.
52 Mg DM × that fall's concentration; user decision). [P1] Table 2's *applied* N is 2–33 %
higher every year (cumulative 6,305 vs 5,502 kg N/ha), and the applied-basis P reproduces [P2]
Table 2's independently reported P within 5.4 % (6 of 8 years within 2.1 %) while the target
basis is 5–25 % below it, so the target basis understates both; kept as
`n_applied_p1` for reference. Workbook columns that are outputs of the collaborator's earlier
SWAT model ([W1] NO3, SOL_P, DNIT, N leaching, PERC, NUP/PUP/BIOM, F_MN/A_MN) are not
measurements and are not ported. Irrigation 2013–2020 is [W4]'s (true calendar dates); [W1]
*Irrigation* has every 2016 and 2020 event exactly one day early (leap years: the series was
evidently built on day of year), so its 2012 events (history year, not in [W4]) are shifted one
day. Season totals reproduce [P1] §3.1 to ≤ 1.5 mm. The workbook label `30_Annual` is 52A.
Dates from [W4]: harvest = the bulk (field) harvest (barley 8/5 and 8/8, wheat 8/13 and 8/16,
potato 10/15, sugarbeet 10/9 and 10/12); 2014 beets killed unharvested 10/28; manure 10/20.
[W4] dates synthetic N on 3/20; [P1] §2.6's actual dates are kept. [W4]'s harvest index 1 /
efficiency 0.4 for sugarbeet and potato are not used: under rev 62 they cut sugarbeet root
yield from 17.2 to 6.9 t/ha (measured 17.2; tested 2026-10-25); the Ai-SWAT-Plus harvest ops
stay (rule 11b). [W4]'s soil row (BD 1.21, carbon 1.8/2.3 %) matches post-manure 2016 52A
samples and total (incl. carbonate) C, not the pre-study soil, and is not used.

**Fertiliser P.** [P1] Table S1 (`saj220599-sup-0001-tabless1-s2.docx`, the article's
open-access supplement, received 2026-10-26): 52A and control received no fertiliser P; the
fertiliser-only plots 39, 151, 19, 98, 16, 38, 20, 59 kg P/ha in 2013–2020, all as MAP, whose N
(0.485 kg per kg P) fits inside [P1] Table 2's N every year. [P2] Table 3 gives 66 for 2014; its
2014 N (107) does not match this field's 75 either, so S1 is used.

**Crop parameters** — Ai-SWAT-Plus, verbatim (CLAUDE.md rule 11b): `scripts/ltm_crop_params.py`
runs Ai-SWAT-Plus's builder (`copy_crop_models.copy_crop`) per crop in a temp dir and keeps its
`plants.plt` row, `calibration.cal` heat units and its tagged `harv.ops` row. barl/sgbt/wwht/
alfa/corn rows are byte-identical to `model/crops/*`. Shared hydrology from the same builds:
esco 0.48, epco 1.0, pet_co 1.0. Engine: the patched rev-62 build the fits ran on
(`model/ltm/engine/`, sha in `model/ltm/crop_params/_shared.json`).

**Engine quirk: `plant` conditions in `calibration.cal`.** rev 62 `cal_conditions.f90` tests a
`plant` condition against `pcom%pl(1)` only (it `exit`s the plant loop after one pass), so with
7 plants in one community only the first-listed crop's `phu_mat` applied. The LTM lines use the
condition variable `phu_plant`, which no filter case handles (so the condition passes) while the
`phu_mat` branch matches `cond%targc` to each plant by name. Drop test: removing one crop's line
changes that crop's first season (barl 2013, sgbt 2014, wwht 2015, pota 2016) and nothing before.
The GRACEnet rotation had no `calibration.cal`, so it was never affected.

**Weather.** gridMET at the trial's coordinates (42°33′4.5″ N, 114°21′14.7″ W, [P2] §2.2):
42.5513 N, 114.3541 W, the 4-km cell west of the GRACEnet point. `scripts/port_gridmet.py
--site ltm` (THREDDS NCSS; when the service stalls, `--from-chunks` assembles per-year
responses fetched by a resumable loop; same unpacking and checks).

**Soil.** SSURGO mukey 80295, Portneuf silt loam 0–2 % (USDA Soil Data Access, point query at the
field, 2026-10-25: Ap 0–28 cm awc 0.20 ksat 9 µm/s om 1.5 %; Bk 28–152 cm awc 0.18 ksat 2.82 om
0.75 % caco3 23 % ph 8.4) for hydraulics. Measured overrides (rule 11): texture per layer ([W3],
4 plots), bulk density 1.28 / 1.37 g cm⁻³ at 0–30 / 30–60 cm ([P1] §2.8), and the pre-study
0–30 cm OM 1.2 %, pH 8.0, CaCO3 5.9 %, EC 1.02, Olsen P 11.1 mg/kg → `lab_p` ([P2] §2.2). The
measured water retention ([W3], four 52A plots, 2016) is manure-affected and not used.
Initial nitrate keeps the Ai-SWAT-Plus value (no pre-study NO3 is reported).

**Management choices not in the record.** 2016 potato N split: equal thirds over the three
reported dates. 2012 pinto bean on Ai-SWAT-Plus's sheet dates (history only). The validation
build kills 2014's beets unharvested on 10/28 ([W4]); the environment's windows harvest them
on the sugarbeet template date (10/9).

**Auto-irrigation** (user decision 2026-10-25: every Exp 1 season, and an Exp 3 baseline;
"realistic, moderate, standard practice"): one decision table per HRU, one alternative per
season = profile plant-available water < 0.7236 × its field-capacity value (lim_op "*") AND
calendar year AND day of year in the season's window (trial crops: first measured event .. 7
days before harvest of the (template) year; alfalfa, silage: Ai-SWAT-Plus spec windows); one
25.4 mm pass at efficiency 1.0. 0.7236 is root-found to the trial's 2013–2020 total
(`model/ltm/auto_irr.json`). Conditioned on year and day, never on the plant (dtl plant
conditions resolve to plant 1). Ai-SWAT-Plus's table writes lim_op "-" (target = FC − 0.95 mm)
and is not used.

**Verifications (2026-10-25).** 0-kg `elem_n` probe ops are inert (hru nb/ls/wb/pw yearly
tables byte-identical with and without). Three treatments in one run equal each run alone
(max |diff| 0.0); mixed annual rotations likewise (`tests/test_ltm.py`). Irrigation gate:
simulated = measured 2013–2020 (`runs/ltm_validation.json`).

**Soil N fit** (`scripts/ltm_calibrate_n.py`; user decision: soil parameters only). One
parameter, the active-humus fraction f (`fr_hum_act` and `n_act_frac`, so the pool starts at
equilibrium), root-found so the control HRU's humus mineralisation over the buried-bag
windows matches [P1] §3.5's 72 kg N/ha (control/fertilizer, non-residue years). `orgn_min`
stays 0.0008: at its 0.006 ceiling the default pool gives under half the target. rev 62 adds
residue/manure mineralisation (`nut_nminrl.f90`, 0.8 × decomposed residue N) to no output
counter, so manured plots have no reported comparator (cf. OPEN_ITEMS #12). The fitted f and
its before/after effect are in `model/ltm/n_cycle.json` and `runs/ltm_validation*.json`.

**Economics** (`swat_gym/ltm/econ.py`): Idaho NASS other-spring-wheat and potato prices
(Crop Values 2024 Summary, Feb 2025, for 2022–23; 2025 Summary, Feb 2026, for 2024 final);
University of Idaho Southcentral Idaho costs-and-returns estimates (2019; potato 2017) for
crop operating costs net of the levers' inputs, escalated to 2022–2024 dollars by NASS's
prices-paid index (×1.264 from 2019, ×1.303 from 2017). Potato tuber DM 0.214 (USDA-ARS FoodData
Central SR Legacy FDC 170027, russet, raw: water 78.6 g/100 g).

### Changes of 2026-10-27 (collaborator email chain; LTM_VERSION 3)

Source: email chain G. Richards / J. Li → E. Du (received 2026-10-27) with the [W4] workbook
(re-downloaded; `scripts/port_ltm.py` reproduces `data/ltm/` byte for byte).

- **Spin-up beans** as the collaborator set them up: 2012 pinto beans auto-irrigated with
  100 kg/ha N + 50 kg/ha P (elemental, `ltm_bnn`/`ltm_bnp`, at planting Jul 1: user decision),
  in the build and in every episode's spin-up year (`practice.SPINUP_*`). The 2012 measured
  events (W1) are no longer replayed. Effect (`runs/ltm_validation.json` vs the 2026-10-25
  record): 52A sugarbeet root DM +3.6 % unchanged, control +21.2 → +20.1 %; 2015 wheat yield
  −11.4 → −2.4 %; 2013–2020 leaching 52A 1,190 → 1,389, control 226 → 394 kg N/ha.
- **No additional calibration** (collaborator). The one soil-N fit (L4, f = 0.21556) is kept and
  not refitted after these changes: before it the control's sugarbeet came out −81.5 %
  (`runs/ltm_validation_before_nfit.json`), the "really off" case the collaborator allowed for.
- **Irrigation efficiency** (OPEN_ITEMS L13). Rev 62 `mgt_sched.f90`/`actions.f90`: applied =
  amt_mm × eff × (1 − surq), and `hru_wb irr` reports the applied amount; the water term and cost
  use gross = applied / eff (`score.episode`, `render.season_irr_eff`; pinned by
  `test_water_is_gross`). Values: Irmak, Odhiambo, Kranz & Eisenhauer 2011, *Irrigation
  Efficiency and Uniformity, and Crop Water Use Efficiency*, UNL Extension EC732, Table 2
  ("potential" application efficiencies): furrow (conventional) 45–65 %, center pivot 75–85 %,
  solid set 70–85 %, surface drip 85–95 % → midpoints 0.55 / 0.80 / 0.90. Measured replay at
  0.8 was tried and rejected: no percolation in 2013–2020, leaching 0, wheat yield −31/−46 %.
- **Rotation rules** (OPEN_ITEMS L9): Patterson & Bohl 1995, *Potato Production Practices in
  Southeastern Idaho*, UI A.E. Extension Series 95-4 ("The most common, potatoes followed by two
  years of grain, was used by 38 percent of the growers"); McDole & Dallimore 1973, *Potato
  Cropping Rotations on Coarse-Textured Soils in Southeastern Idaho*, UI Current Information
  Series 200 ("not more than one crop of potatoes in succession").
- **Not used:** the corrected 2020 sugarbeet P uptake (68.5 kg/ha) is the collaborator's SWAT
  output, not a measurement (OPEN_ITEMS L15). [W4]'s harvest index/efficiency columns stay
  unused (Ai-SWAT-Plus harvest ops kept, rule 11b).
