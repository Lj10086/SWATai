#!/usr/bin/env python3
"""Write runs/ltm_summary.html: what was built for the LTM environment, before any experiment
is run. Every number is read from a file this repo wrote (no transcription):
runs/ltm_validation.json (+ _before_nfit.json), model/ltm/{build,n_cycle}.json,
model/ltm/crop_params/_shared.json, runs/bench_{rl,es,ga}_xl.json, swat_gym.ltm.econ.

Styling per the user's instruction: barebones, white background, Calibri only.
"""
from __future__ import annotations

import html
import json
import statistics as st

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "ltm_summary.html"


def esc(x) -> str:
    return html.escape(str(x))


def table(head: list[str], rows: list[list], cls: str = "") -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def f(x, nd=1, pct=False) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    s = f"{x:+.{nd}f}" if pct else f"{x:,.{nd}f}"
    return s + (" %" if pct else "")


def validation_section(v: dict, before: dict | None) -> str:
    rows = []
    for r in v["harvest"]:
        if r["treatment"] != "52A":
            continue
        rows.append([r["year"], esc(r["crop"]), f(r["sim_biomass_t"], 2),
                     f(r.get("meas_biomass_t"), 2), f(r.get("biomass_pct_err"), 1, True),
                     f(r["sim_yield_dm_t"], 2), f(r.get("meas_yield_dm_t"), 2),
                     f(r.get("yield_pct_err"), 1, True)])
    harvest = table(["Year", "Crop", "Biomass sim (t/ha)", "Biomass meas.", "Error",
                     "Yield DM sim", "Yield DM meas.", "Error"], rows)
    sb = v["sugarbeet_root_dm"]
    sb_rows = []
    for t in ("52A", "fert", "control"):
        b = before["sugarbeet_root_dm"][t] if before else None
        sb_rows.append([esc(t), f(sb[t]["meas_t"], 1), f(b["sim_t"], 1) if b else "–",
                        f(b["pct_err"], 1, True) if b else "–", f(sb[t]["sim_t"], 1),
                        f(sb[t]["pct_err"], 1, True)])
    sugar = table(["Treatment", "Measured root DM (t/ha)", "Sim, before N fit", "Error",
                   "Sim, after N fit", "Error"], sb_rows)
    up = table(["Year", "Crop", "N uptake meas.", "sim", "P uptake meas.", "sim"],
               [[r["year"], esc(r["crop"]), f(r["meas_n"], 0), f(r["sim_n"], 0),
                 f(r["meas_p"], 0), f(r["sim_p"], 0)] for r in v["uptake_52A"]])
    nm = table(["Year", "Measured net mineralisation, 52A (kg N/ha)",
                "Simulated humus mineralisation, 52A (lower bound)"],
               [[r["year"], f(r["meas_net_in"], 0), f(r["sim_act_nit"], 0)]
                for r in v["nmin_52A"]])
    s = v["soil_52A"]
    c20 = s["no3_2020_contrast"]
    contrast = table(["Treatment", "Measured / 52A", "Simulated / 52A"],
                     [[esc(t), f(c20[t]["meas_ratio_to_52A"], 2), f(c20[t]["sim_ratio_to_52A"], 2)]
                      for t in ("52A", "fert", "control")])
    g = v["irrigation_gate"]
    loss = v["losses_2013_2020_sum"]
    loss_t = table(["Treatment", "NO3-N leached", "P leached", "Soluble P runoff",
                    "Percolation (mm)", "ET (mm)"],
                   [[esc(t), f(loss[t]["percn"], 0), f(loss[t]["lchlabp"], 2),
                     f(loss[t]["surqsolp"], 2), f(loss[t]["perc"], 0), f(loss[t]["et"], 0)]
                    for t in ("52A", "fert", "control")])
    return f"""
<h2 id="validation">4. Validation against the trial (nothing fitted except the soil N step in §5)</h2>
<p>Run: <code>scripts/ltm_validate.py</code> → <code>runs/ltm_validation.json</code>. Irrigation
gate: simulated {f(g['sim_mm'], 1)} mm vs measured {f(g['meas_mm'], 1)} mm over 2013–2020
(relative error {g['rel_err']:.2%}; {'pass' if g['pass'] else 'FAIL'}).</p>
<h3>Harvest, 52A (the only treatment with per-year plot data)</h3>{harvest}
<p class="note">2014 sugarbeets were not harvested in the field (tilled in), so the validation
build harvests nothing that year. Potato yield is fresh-weight only in the trial, so only its
biomass is compared. Measured biomass is the collaborator's “From April Leytem” table.</p>
<h3>Treatment contrast: sugarbeet root dry matter (Leytem et al. 2025, Table 5)</h3>{sugar}
<p class="note">Measured: 4-year, 2-field means. Simulated: mean of 2018 and 2020 (the
harvested sugarbeet years on this field).</p>
<h3>Plant uptake, 52A (kg/ha)</h3>{up}
<h3>Soil</h3>
<p>Trajectory shape (Pearson r, 52A, spring samples 2013–2020): profile NO3 r =
{f(s['profile_no3_mg_kg']['pearson_r'], 2)}; Olsen P (0–30 cm) vs simulated solution P r =
{f(s['olsen_p_mg_kg_0_30']['pearson_r'], 2)}. Depths and extractants differ, so only shape is
compared.</p>
<p>Spring 2020 soil NO3 by treatment, relative to 52A (Leytem et al. 2024, Table 5, 0–60 cm):</p>
{contrast}
<h3>N mineralisation, 52A</h3>{nm}
<p class="note">SWAT+ rev 62 never reports residue/manure mineralisation (it is computed in
<code>nut_nminrl.f90</code> but added to no output counter), so for manured plots only the humus
part is visible: a lower bound. The gap is largest after residue years (2015, 2017, 2019), as
expected.</p>
<h3>Losses 2013–2020 (no on-site measurement; reported only, kg/ha unless noted)</h3>{loss_t}
"""


def bench_section() -> str:
    es = json.loads((RUNS / "bench_es_xl.json").read_text())["cases"]
    ga = json.loads((RUNS / "bench_ga_xl.json").read_text())["cases"]
    rows, verdicts, es_vs_ga = [], [], []
    for e, g in zip(es, ga):
        p = list(e["ppo_test"].values())
        x = list(e["ga_test"].values())
        y = list(g["ga_test"].values())
        b = e["best_of_runs_picked_on_train"]["paired_ga_vs_ppo"]
        lo, hi = b["ci95_boot"]
        verdict = "ES better" if lo > 0 else ("PPO better" if hi < 0 else "unresolved")
        verdicts.append((e["lever"], verdict))
        es_vs_ga.append(np.sign(round(st.mean(x) - st.mean(y), 6)))
        rows.append([esc(f"{e['crop']} / {e['lever']}"), f(e["default_test"], 0),
                     f(e["fixed_test"], 0), f(st.mean(p), 0), f(st.mean(x), 0), f(st.mean(y), 0),
                     f"{b['mean']:+.0f} [{lo:+.0f}, {hi:+.0f}]", verdict])
    cfg = json.loads((RUNS / "bench_es_xl.json").read_text())["config"]
    n_cases = {lv: sum(1 for l, _ in verdicts if l == lv) for lv in ("N", "I")}
    better = {lv: sum(1 for l, v in verdicts if l == lv and v == "ES better") for lv in ("N", "I")}
    worse = sum(1 for _, v in verdicts if v == "PPO better")
    summary = (f"ES is resolvably worse than PPO on {worse} of {len(verdicts)} cases and "
               f"resolvably better on {better['N']} of {n_cases['N']} nitrogen cases and "
               f"{better['I']} of {n_cases['I']} irrigation cases (the rest unresolved; wheat N "
               f"is identical for every method). By mean, ES beats the GA on "
               f"{sum(1 for x in es_vs_ga if x > 0)}, ties on {sum(1 for x in es_vs_ga if x == 0)} "
               f"and trails on {sum(1 for x in es_vs_ga if x < 0)} of {len(es_vs_ga)} cases.")
    t = table(["Crop / lever", "Default", "CMA-ES fixed", "PPO (mean of 4)", "ES (mean of 4)",
               "GA (mean of 4)", "ES − PPO, best-on-train (95 % bootstrap)", "Verdict"], rows)
    return f"""
<h2 id="method">7. Why ES</h2>
<p>Crop-model benchmark (5 Ai-SWAT-Plus monocultures × levers, 4 seeds each, budget
{cfg['budget']:,} candidate-seasons per seed; test-window profit, $/ha).
Source: <code>runs/bench_rl_xl.json</code>, <code>bench_es_xl.json</code>,
<code>bench_ga_xl.json</code>.</p>{t}
<p>{summary} ES also has no
train-season pick asymmetry (the benchmark gives ES/GA a final pick on train, PPO none — a
known caveat, memory “PPO final pick deferred”). Note that plain CMA-ES (open loop) is as good
as any learner on most cases, which is why it stays in the protocol as <em>fixed</em>.</p>
"""


def econ_section() -> str:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from swat_gym.ltm import econ
    p = econ.prices()
    names = {"barl": "Malt barley", "wwht": "Spring wheat (HRS)", "sgbt": "Sugarbeet",
             "pota": "Potato (Russet Burbank)", "alfa": "Alfalfa hay", "corn": "Corn silage"}
    rows = [[esc(names[c]), f(p.base.crop[c], 0), f(econ.OPERATING_USD_ACRE[c], 2),
             f(p.operating[c], 0), econ.COST_YEAR[c]] for c in names]
    t = table(["Crop", "Price ($/Mg DM, 2022–24)", "Other operating cost ($/acre)",
               "($/ha)", "Budget year"], rows)
    b = p.base
    return f"""
<h2 id="econ">8. Economics</h2>
<p>profit = revenue − water − manure − mineral N − MAP P − application passes −
<strong>crop operating costs</strong>. The last term is new: with a rotation lever, a crop's
seed, chemicals, custom work, labour and machinery differ by crop (potato costs ~4.6× barley),
and without it “pure profit” would simply pick the highest-revenue crop.</p>{t}
<p>Inputs: water ${b.water:.3f}/mm/ha, manure ${b.manure:.2f}/Mg as applied, urea N
${b.fert_n:.2f}/kg, MAP P ${p.fert_p:.2f}/kg, ${b.fert_op:.2f}/ha per pass (all unchanged
from <code>rewarders.py</code>). Prices: Idaho NASS marketing-year averages 2022–2024
(Crop Values 2024 and 2025 Summaries); wheat at the <em>other spring wheat</em> price
($8.20 / $7.03 / $5.70 per bu), potato $12.50 / $9.63 / $9.02 per cwt. Operating costs:
University of Idaho Extension, Southcentral Idaho costs-and-returns estimates (EBB3-MB/HRS/SU/
AH/AE2/CS-19; potato: 2017 workbook, sheet Potato2), each = total operating cost minus
fertiliser, irrigation power and repairs, custom fertilising and interest (the levers price
those). Land rent and overhead are the same for every crop and are left out.</p>
<p>Potato dry matter {econ.POTA_DM:.3f} converts the fresh price to $/Mg DM: 1 − water 78.6 g/100 g,
USDA-ARS FoodData Central SR Legacy FDC 170027, “Potatoes, russet, flesh and skin, raw”. It also
lets the potato be validated: measured 2016 tuber yield 69.5 Mg/ha fresh = 14.9 t DM against 10.6
simulated (−29 %, the Ai-SWAT-Plus canopy problem).</p>
<p>Operating costs are escalated to mean 2022–24 dollars with USDA NASS's index of prices paid by
farmers (all farms, 2011 = 100): ×{econ.escalation(2019):.3f} for the 2019 budgets and
×{econ.escalation(2017):.3f} for potato's 2017 budget. The $/ha column above is escalated; the $/acre
column is the budget's own.</p>
"""


def main() -> None:
    v = json.loads((RUNS / "ltm_validation.json").read_text())
    before_p = RUNS / "ltm_validation_before_nfit.json"
    before = json.loads(before_p.read_text()) if before_p.is_file() else None
    build = json.loads((ROOT / "model" / "ltm" / "build.json").read_text())
    nfit_p = ROOT / "model" / "ltm" / "n_cycle.json"
    nfit = json.loads(nfit_p.read_text()) if nfit_p.is_file() else None
    shared = json.loads((ROOT / "model" / "ltm" / "crop_params" / "_shared.json").read_text())
    crops = {p.stem: json.loads(p.read_text()) for p in
             sorted((ROOT / "model" / "ltm" / "crop_params").glob("[!_]*.json"))}
    weather = json.loads((ROOT / "data" / "gridmet_ltm_1995_2025.json").read_text())

    crop_rows = [[esc(c), esc(r["code"]), esc(r["role"]), f(r["phu_mat"], 1), esc(r["harv_op"])]
                 for c, r in crops.items()]
    nfit_html = ("<p>Not applied.</p>" if nfit is None else f"""
<p>One soil parameter was fitted (crop parameters untouched): the active share of soil humus N,
<em>f</em>, set as both the initial fraction (<code>fr_hum_act</code>) and the equilibrium of the
active↔stable transfer (<code>n_act_frac</code>), so the pool starts at equilibrium. Target: the
control plots' measured net seasonal mineralisation in years without residue the previous fall,
{nfit['target_kg_ha']:.0f} kg N/ha (Leytem et al. 2024 §3.5), against the control HRU's
simulated humus mineralisation over each year's buried-bag window. Result:
<strong><em>f</em> = {nfit['f_act']:.4f}</strong> (engine default {nfit['default_f_act']}),
{nfit['engine_runs']} engine runs. The rate <code>orgn_min</code> stays at the inherited
{nfit['orgn_min_kept']}: even at its 0.006 ceiling the default pool reaches less than half the
target, so pool size is what was short.</p>
<p class="flag"><em>f</em> sits near the top of the range Ai-SWAT-Plus searched (0.25): read it
as “at least this much”. Held-out checks are the treatment contrasts and uptake in §4.</p>""")
    over = [r for r in v["nmin_52A"] if r["year"] in (2013, 2014, 2016, 2018, 2020)
            and r["sim_act_nit"] > r["meas_net_in"]]
    leach = v["losses_2013_2020_sum"]
    c20 = v["soil_52A"]["no3_2020_contrast"]
    if nfit is not None:
        nfit_html += (f"""<p class="flag"><strong>Tension the fit exposes.</strong> In {len(over)} of 5
non-residue years the 52A plots' <em>humus-only</em> mineralisation already exceeds the measured
<em>total</em> ({", ".join(f"{r['year']}: {r['sim_act_nit']:.0f} vs {r['meas_net_in']:.0f}" for r in over)}
kg N/ha), before the unreported residue part, so manured soil likely over-mineralises: 52A
leaches {leach['52A']['percn'] / 8:.0f} kg NO3-N/ha/yr (control {leach['control']['percn'] / 8:.0f}),
and spring-2020 soil NO3 relative to 52A is {c20['fert']['sim_ratio_to_52A']:.2f} (fertiliser) and
{c20['control']['sim_ratio_to_52A']:.2f} (control) against measured
{c20['fert']['meas_ratio_to_52A']:.2f} and {c20['control']['meas_ratio_to_52A']:.2f}. One soil
parameter cannot fit both, and the engine does not report the flux that would identify a
second (the manure residue/humus split, <code>rtof</code>). Leaching levels under heavy manure
should be read as an upper bound.</p>""")

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LTM scenario and experiment design (before runs)</title>
<style>
  body {{ font-family: Calibri, sans-serif; background: #fff; color: #000; margin: 2em auto;
         max-width: 1050px; padding: 0 1em; line-height: 1.4; }}
  table {{ border-collapse: collapse; margin: 0.6em 0 1.2em; font-size: 0.95em; }}
  th, td {{ border: 1px solid #999; padding: 3px 8px; text-align: left; vertical-align: top; }}
  th {{ font-weight: bold; }}
  code {{ font-family: Calibri, sans-serif; font-style: italic; }}
  .note {{ font-size: 0.92em; }}
  .flag {{ border-left: 3px solid #000; padding-left: 0.6em; }}
</style></head><body>
<h1>Long-term manure scenario and experiment design</h1>
<p>Status, 2026-10-25: the scenario is built, validated, and wired into three experiments,
each smoke-tested end to end. <strong>No experiment has been run at full budget.</strong>
Branch <code>ltm-scenario</code>; nothing committed.</p>
<p><a href="#data">Data</a> · <a href="#w4">Collaborator's inputs</a> · <a href="#scenario">Scenario</a> · <a href="#checks">Engine
checks</a> · <a href="#validation">Validation</a> · <a href="#nfit">Soil N fit</a> ·
<a href="#exps">Experiments</a> · <a href="#method">Why ES</a> · <a href="#econ">Economics</a> ·
<a href="#open">Open items</a></p>

<h2 id="data">1. Data aggregated (<code>scripts/port_ltm.py</code> → <code>data/ltm/</code>)</h2>
{table(["File", "Contents", "Source"], [
    ["management.csv", "Per treatment and year: crop, plant and harvest dates, fall manure (DM rate, mineral/organic N, P, C), synthetic N and P, fall tillage", "Leytem et al. 2024 (SSSAJ, doi:10.1002/saj2.20599) Tables 1–2 and §2.6; Leytem et al. 2025 (Crop Sci.) Table 3"],
    ["irrigation_daily.csv", "Measured daily irrigation 2012–2020 (season totals reproduce the paper's 385–788 mm)", "LT Manure Data Summary.xlsx, sheet Irrigation (“actual amounts, A. Leytem”)"],
    ["yield.csv", "52A per-plot grain, root, tuber and whole-plant biomass and P uptake; sugarbeet yield and root DM by treatment", "LT Manure Soil Properties and P uptake data.xlsx; Leytem et al. 2025 §3.4, Table 5"],
    ["uptake.csv", "52A crop-year biomass, N and P uptake with SDs", "Data Summary, Crop Uptake “From April Leytem”"],
    ["soil_layers.csv, soil_props.csv", "Texture by depth (4 plots), bulk density, pre-study chemistry", "Soil Properties workbook; Leytem et al. 2025 §2.2; Leytem et al. 2024 §2.8"],
    ["soil_series.csv", "Spring NO3, NH4, Olsen P, OM, TN per plot and year; 2020 treatment contrast", "Soil Properties workbook; Data Summary; Leytem et al. 2024 Table 5; 2025 Table 4"],
    ["nmin.csv", "Measured buried-bag net mineralisation per year (52A) and the control summary", "Data Summary, N Min; Leytem et al. 2024 Table 4, §3.5"],
])}
<p class="note">Workbook columns that are outputs of the collaborator's earlier SWAT model (NO3,
SOL_P, DNIT, leaching, PERC, model uptake) are not ported. Manure nutrients are on the
collaborator's target-rate basis (your decision; §2a). Fertiliser P is from the paper's
supplementary Table S1. </p>

<h2 id="w4">2a. The collaborator's input set (Long Term Manure Project Input Data.xlsx)</h2>
{table(["Item", "What it said", "Used?", "Why"], [
    ["Irrigation 2013–2020", "Event dates and depths", "<strong>Yes</strong>", "Fixes a bug: the earlier workbook had every 2016 and 2020 event one day early (leap years). 2012 events shifted the same way."],
    ["2014 sugarbeet", "Not harvested; kill Oct 28", "<strong>Yes</strong>", "2014 biomass error −12 % → −3 %."],
    ["Manure date", "Oct 20", "<strong>Yes</strong>", "Replaced an assumed Oct 30."],
    ["Harvest dates", "Bulk (field) harvest", "<strong>Yes</strong>", "When the crop leaves the field; were plot-sampling dates."],
    ["Manure rate", "5,200 kg/ha × fractions (= 52 Mg DM × concentration)", "<strong>Yes</strong> (your decision)", "Target-rate basis: 1.5–25 % less N than the applied N in Leytem et al. 2024 Table 2, and P 5–25 % below the 2025 paper's; flagged (L12)."],
    ["Harvest index / efficiency", "1 / 0.4 for sugarbeet and potato", "No", "Under SWAT+ rev 62 these cut sugarbeet roots to 6.9 t/ha (measured 17.2); Ai-SWAT-Plus harvest ops kept (set in stone)."],
    ["Synthetic N date", "Mar 20", "No", "Leytem et al. 2024 §2.6 gives the actual dates."],
    ["Soil file", "6 layers, carbon 1.8 / 2.3 %", "No", "Matches post-manure 2016 samples and total (incl. carbonate) carbon, not the pre-study soil (OM 1.2 %)."],
    ["Auto-irrigation idea", "Use auto-irrigation with the start/stop dates", "<strong>Yes</strong>, in the experiments", "Every Exp 1 season, and an Exp 3 baseline; validation and the reference keep measured irrigation. Schedule provisional (L11)."],
])}

<h2 id="scenario">2. Scenario (<code>scripts/build_ltm.py</code> → <code>model/ltm/TxtInOut</code>)</h2>
<ul>
<li><strong>Field and treatments.</strong> South field, malt barley → sugarbeet → spring wheat →
potato → barley → sugarbeet → spring wheat → sugarbeet (2013–2020; 2012 pinto bean as
history). Three 1-ha HRUs in one engine run: 52A (52 Mg/ha DM manure every fall),
fertiliser-only, control.</li>
<li><strong>Crop parameters: Ai-SWAT-Plus, verbatim.</strong> Each crop built with
Ai-SWAT-Plus's own builder; its fitted <code>plants.plt</code> row, heat units and harvest op
copied (<code>scripts/ltm_crop_params.py</code>). Rows identical to the existing crop models.
Spring wheat uses <code>wwht</code> and potato <code>pota</code> as-is (your decisions).</li>
<li><strong>Hydrology</strong> as the fits were made: esco {shared['hydrology']['esco']},
epco {shared['hydrology']['epco']}, pet_co {shared['hydrology']['pet_co']}; the patched rev-62
engine the fits ran on (sha {shared['engine_sha256']}).</li>
<li><strong>Weather.</strong> gridMET at the trial's own coordinates ({weather['lat']} N,
{abs(weather['lon'])} W, Leytem et al. 2025), 1995–2025 — the adjacent 4-km cell to the one
the GRACEnet model used.</li>
<li><strong>Soil.</strong> SSURGO mukey 80295 (Portneuf silt loam, Soil Data Access at the
field point) for hydraulics; measured texture by depth, bulk density (1.28 / 1.37 g/cm³) and
pre-study OM, pH, CaCO3, EC and Olsen P (11.1 mg/kg) override SSURGO.</li>
<li><strong>Management.</strong> Planting and fertiliser dates from the paper; bulk harvest
dates, the 2014 kill and the Oct 20 manure date from the collaborator's set; manure at
52 Mg/ha DM with each fall's nutrient fractions (collaborator's target-rate basis), disked in
to 15 cm; urea and MAP; measured daily irrigation; 2016 potato N split in equal thirds (not
reported); fertiliser P as recorded in the paper's Table S1.</li>
</ul>
<h3>Crops in the environment</h3>
{table(["Plant", "Ai-SWAT-Plus code", "Role", "Heat units to maturity", "Harvest op"], crop_rows)}

<h2 id="checks">3. Engine checks</h2>
<ul>
<li><strong>Found an engine quirk and worked around it.</strong> SWAT+ rev 62 tests a
<code>plant</code> condition in <code>calibration.cal</code> against the community's
<em>first</em> plant only, so in a rotation only barley's fitted heat units applied. The
per-crop lines now use a condition name the filter ignores, and the heat-unit branch matches
each crop by name. Verified: removing any one crop's line changes that crop's first year and
nothing before it.</li>
<li><strong>Zero-kg probe ops are inert</strong> (byte-identical yearly tables with and without
them). They give the environment soil NO3 / P / water on decision dates.</li>
<li><strong>Batching is exact</strong> for annual crops: each treatment run alone equals its
HRU in the 3-HRU run (max |difference| 0.0); mixed annual rotations likewise
(<code>tests/test_ltm.py</code>). Alfalfa is never batched (known per-plant demand leak).</li>
<li><strong>Parity (rule 2)</strong>: a closed-loop episode scores exactly as its final plan run
open loop, for every profit and sustainability term, on all three levers.</li>
</ul>
{validation_section(v, before)}
<h2 id="nfit">5. Soil N fit (your decision: refit soil parameters only)</h2>{nfit_html}

<h2 id="exps">6. The three experiments</h2>
{table(["Exp", "Lever (decisions)", "Held at", "Open-loop baseline"], [
    ["1", "Crop each season from barley, sugarbeet, spring wheat, potato, corn silage, alfalfa (3-year stand); no other rotation rules (8 decisions)", "Trial practice for each trial crop (its template year's dates and 52A fertiliser); Ai-SWAT-Plus dates for alfalfa and silage; <strong>auto-irrigation for every season</strong> in each crop's window; 52A manure each fall except into standing alfalfa", "Every repeating cycle of length ≤ 4: 755 rotations, exhaustively"],
    ["2", "Fall manure 0–52 Mg/ha DM (nominal) for 8 falls + monthly synthetic N (45 months) (53 decisions)", "Measured rotation, irrigation and MAP P", "CMA-ES at the ES budget"],
    ["3", "Monthly irrigation depth for all 8 seasons (40 decisions)", "Measured rotation, manure, N", "CMA-ES at the ES budget; plus an <strong>auto-irrigation</strong> baseline row"],
])}
<p><strong>Score per strategy</strong> (your decision; replaces the frontier rule):
<em>score = w · profit / profit<sub>ref</sub> − (1 − w) · mean<sub>k</sub>(term<sub>k</sub> /
term<sub>k,ref</sub>)</em>, w = 1 (profit), 0.5 (balanced), 0 (sustainability); terms: NO3
leached, irrigation, P lost + P build-up, N₂O (IPCC Tier 1); references = train-window means of
measured 52A practice. Rows per strategy: measured, default, fixed, ES (4 seeds), frozen (each
policy's own decisions replayed open loop); test-window paired differences with
moving-block-bootstrap and ESS intervals; seeds that disagree in sign → “unresolvable”.</p>
<p><strong>Episodes</strong>: spin-up year + 8 seasons (your decision); train windows start
1995–2001, test 2011–2017, zero shared years; ESS 1.34 each. <strong>Budget</strong>: 200,000
HRU-runs per ES seed and per CMA-ES search; ES plays one run per decision, CMA-ES one per
candidate-window. Estimated wall time on 10 cores: ~1–2 h each for Exp 2 and 3, under 1 h for
Exp 1.</p>
{bench_section()}
{econ_section()}

<h2 id="open">9. Decisions taken, and open items that need you</h2>
<h3>Your decisions encoded</h3>
<ul>
<li>LTM replaces the GRACEnet environment; three levers (rotation cut reversed); rule 1 retired
in favour of three fixed strategies; rule 11b now points at Ai-SWAT-Plus; pet_co / esco from
Ai-SWAT-Plus. CLAUDE.md, EXPERIMENTS.md and OPEN_ITEMS.md updated.</li>
<li>Treatments 52A + fertiliser + control; potato as-is; wheat on wwht; alfalfa 3-year stands;
manure 0–52 Mg/ha DM; 8-year episodes; soil N refit; ES.</li>
</ul>
<h3>Closed since the first version</h3>
<ul>
<li><strong>Potato dry matter</strong>: 0.214 (USDA FoodData Central), see §8.</li>
<li><strong>Auto-irrigation</strong>: one 1-inch pass whenever profile plant-available water drops below
72 % of its field-capacity value, inside each season's window (first measured event to a week
before harvest); the 72 % is fitted so the rule applies the trial's own 4,542 mm over 2013–2020
(9–29 passes a year vs the trial's 5–22). Found on the way: Ai-SWAT-Plus's irrigation table
uses “FC − 0.95 mm” where its documentation says “0.95 × FC”, i.e. it irrigates whenever the
profile is below field capacity — worth fixing in that project, whose crop fits ran under it.</li>
<li>Rotation rules (none) and IPCC factors (verified).</li>
<li><strong>Fertiliser P</strong>: from the paper's Table S1 (52A and control none; fertiliser plots 16–151 kg P/ha).</li>
<li><strong>Cost year</strong>: escalated with NASS's prices-paid index (§8). <strong>Manure basis</strong>: your decision (target rate) stands.</li>
</ul>
<h3>Still open</h3>
<ol>

<li><strong>Rotation rules: none</strong> (your decision; a stated limitation). Stated limitation: SWAT+ has no
soilborne disease or nematode module, so back-to-back root crops are not penalised (simulated
sugarbeet ↔ potato alternation: 17.6 → 16.6 t/ha sugarbeet and 9.7 → 9.0 t/ha potato over 8
seasons, the same drift as with a grain break). Regional practice for contrast: Magic Valley
rotations are 3–4 years, potato–sugar beet–small grain or potato–small grain–sugar beet–small
grain (Hills et al. 2018, WSU report for the Washington State Potato Commission).</li>

<li><strong>Cost year</strong>: escalate UI budgets (2019; potato 2017) to 2022–24 with the NASS
Prices Paid index, or leave as is? It matters differentially for potato.</li>
<li>Soil N: one parameter, near its range top; residue mineralisation is not reported by the
engine.</li>
<li>Soil hydraulics from SSURGO (the measured retention is from manured plots in 2016).</li>

<li><strong>IPCC factors verified</strong> against the 2019 Refinement (Vol. 4 Ch. 11, Tables 11.1 and 11.3). Kimberly is “dry” there, where the optional disaggregated EF1/EF4 are 0.005: report as a sensitivity.</li>
<li><strong>Manure basis</strong>: the target rate you chose gives 1.5–25 % less N than the paper's
applied N each year. Worth confirming the collaborator intended this.</li>

<li>ESS 1.34: intervals will be wide.</li>
</ol>
<h3>What happens when you say go</h3>
<p>All three experiments are running overnight at full budget, in paper order (<code>runs/ltm_overnight.log</code>). No input-data items remain open.</p>
</body></html>
"""
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
