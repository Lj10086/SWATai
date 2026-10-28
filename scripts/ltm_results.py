#!/usr/bin/env python3
"""Write runs/ltm_results.html: results of the three LTM experiments (full budget, 4 ES seeds).
Every number is read from a file this repo wrote (no transcription): runs/ltm_exp{1,2,3}.json,
runs/ltm_exp{1,2,3}/es_*.json (per-seed plans), runs/ltm_validation.json, data/ltm/yield.csv.

Styling per the user's instruction: barebones, white background, Calibri only.

    uv run python scripts/ltm_results.py
"""
from __future__ import annotations

import html
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swat_gym.ltm import econ  # noqa: E402
from swat_gym.ltm.env import plan_from_x  # noqa: E402
from swat_gym.ltm.practice import TRIAL_ROTATION, measured_x  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import ltm_figures as fig  # noqa: E402

RUNS = ROOT / "runs"
OUT = RUNS / "ltm_results.html"
EXPS = {1: ("R", "Crop rotation"), 2: ("N", "Nitrogen: manure vs synthetic"),
        3: ("I", "Irrigation")}
STRATS = ("profit", "balanced", "sustainability")
W = {"profit": 1.0, "balanced": 0.5, "sustainability": 0.0}
CROP = {"barl": "barley", "sgbt": "sugarbeet", "wwht": "spring wheat", "pota": "potato",
        "alfa": "alfalfa", "corn": "corn silage", "barley": "barley", "sugarbeet": "sugarbeet",
        "wheat": "spring wheat", "potato": "potato"}
MANURE_MG = 52.0
# Exp 3 efficiency cases (OPEN_ITEMS L13): result file per case, one sequential hue
EFFS = {0.55: "ltm_exp3_eff55", 0.80: "ltm_exp3", 0.90: "ltm_exp3_eff90"}
EFF_COL = {0.55: "#9ec5f0", 0.80: "#2a78d6", 0.90: "#0d3f80"}


def esc(x) -> str:
    return html.escape(str(x))


def table(head: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def f(x, nd=1, sign=False) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    return f"{x:+,.{nd}f}" if sign else f"{x:,.{nd}f}"


def rot(crops) -> str:
    """Compact rotation: runs of one crop as 'potato ×8'."""
    out, prev, n = [], None, 0
    for c in list(crops) + [None]:
        if c == prev:
            n += 1
            continue
        if prev is not None:
            out.append(CROP.get(prev, prev) + (f" ×{n}" if n > 1 else ""))
        prev, n = c, 1
    return " → ".join(out)


def cmp_cell(c: dict, key: str = "score", nd: int = 2) -> str:
    m, (lo, hi) = c[key]["mean"], c[key]["ci95_ess"]
    return f"{f(m, nd, True)} <span class='ci'>[{f(lo, nd, True)}, {f(hi, nd, True)}]</span>"


def load(n: int) -> dict:
    return json.loads((RUNS / f"ltm_exp{n}.json").read_text())


def es_runs(n: int, s: str, seeds) -> list[dict]:
    return [json.loads((RUNS / f"ltm_exp{n}" / f"es_{s}_s{sd}.json").read_text())["result"]
            for sd in seeds]


def es_mean(v: dict) -> dict:
    seeds = list(v["es"])
    return {k: st.mean(v["es"][sd]["test_means"][k] for sd in seeds)
            for k in v["es"][seeds[0]]["test_means"]}


ARM_LABEL = {"measured": "Measured 52A practice (replayed)",
             "default": "<code>default</code>: trial practice through the lever",
             "auto": "Auto-irrigation (fitted rule)", "fixed": "Best fixed plan (open loop)",
             "es": "Learned policy (ES, mean of 4 seeds)"}


def arms_table(v: dict) -> str:
    tm = {**v["test_means"], "es": es_mean(v)}
    rows = []
    for arm in ("measured", "default", "auto", "fixed", "es"):
        if arm not in tm:
            continue
        a = tm[arm]
        rows.append([ARM_LABEL[arm], f(a["score"], 2), f(a["profit"], 0), f(a["no3"], 1),
                     f(a["water"], 0), f(a["p"], 1), f(a["n2o"], 1)])
    return table(["Arm (test years)", "Score", "Profit $/ha/yr", "NO₃-N leached kg/ha/yr",
                  "Irrigation mm/yr", "P loss + buildup kg/ha/yr", "N₂O-N kg/ha/yr"], rows)


def seeds_table(v: dict, has_auto: bool) -> str:
    cols = [("policy_vs_default", "Policy − default"), ("policy_vs_fixed", "Policy − fixed"),
            ("adaptivity_value", "Adaptivity (policy − its frozen plan)")]
    if has_auto:
        cols.insert(1, ("policy_vs_auto", "Policy − auto-irrigation"))
    rows = [[f"seed {sd}"] + [cmp_cell(v["es"][sd][k]) for k, _ in cols] for sd in v["es"]]
    res = v["resolvable"]
    rows.append(["<em>sign across seeds</em>"] +
                [esc(res.get(k, "–")) for k, _ in cols])
    return table(["Score difference, test"] + [c for _, c in cols], rows)


def profit_row(v: dict, key: str) -> str:
    c = [v["es"][sd][key]["profit"] for sd in v["es"]]
    return ", ".join(f(x["mean"], 0, True) for x in c)


# --- experiment-specific plan descriptions ------------------------------------------------------
def exp1_plans(d: dict) -> str:
    rows = []
    for s in STRATS:
        v = d["strategies"][s]
        es_rot = {rot(v["es"][sd]["frozen_crops"]) for sd in v["es"]}
        rows.append([s, esc(" / ".join(sorted(es_rot))), esc(rot(v["fixed"]["crops"])),
                     f(v["fixed"]["train_score"], 2)])
    top = d["strategies"]["profit"]["fixed"]["top10"]
    top_rows = [[i + 1, esc(rot(c)), f(sc, 2)] for i, (c, sc) in enumerate(top[:6])]
    n = d["strategies"]["profit"]["fixed"]["n_candidates"]
    return (table(["Strategy", "Learned policy's rotation (all 4 seeds)",
                   f"Best fixed rotation (of {n})", "Fixed, train score"], rows) +
            "<p>Next-best fixed rotations for the profit strategy (train score):</p>" +
            table(["Rank", "Rotation", "Train score"], top_rows))


def exp2_plans(d: dict) -> str:
    meas = plan_from_x("N", measured_x("N"))
    rows = [["Measured practice", ", ".join(f(m * MANURE_MG, 0) for m in meas.manure_x),
             ", ".join("–" if a is None else f(sum(a), 0) for a in meas.n_kg)]]
    for s in STRATS:
        p = plan_from_x("N", np.asarray(d["strategies"][s]["fixed"]["x"]))
        rows.append([f"Best fixed, {s}", ", ".join(f(m * MANURE_MG, 1) for m in p.manure_x),
                     ", ".join("–" if a is None else f(sum(a), 0) for a in p.n_kg)])
    es_rows = []
    for s in STRATS:
        res = es_runs(2, s, d["strategies"][s]["es"])
        man = [st.mean(e["manure_dm_mg"] for e in r["test"]) for r in res]
        nkg = [st.mean(e["mineral_n_kg"] for e in r["test"]) for r in res]
        es_rows.append([s, ", ".join(f(x, 1) for x in man), ", ".join(f(x, 0) for x in nkg)])
    return (table(["Plan", "Fall manure, Mg DM/ha, falls 1–8",
                   "Urea N, kg N/ha, seasons 1–8 (MAP as measured)"], rows) +
            "<p>Learned policy, test-window means per year (one value per seed):</p>" +
            table(["Strategy", "Manure Mg DM/ha/yr", "Synthetic N kg/ha/yr"], es_rows))


def exp3_plans(d: dict) -> str:
    meas = plan_from_x("I", measured_x("I"))
    rows = [["<code>default</code> (measured, as monthly depths)"] +
            [f(sum(a), 0) for a in meas.irr_mm] + [f(sum(sum(a) for a in meas.irr_mm), 0)]]
    for s in STRATS:
        p = plan_from_x("I", np.asarray(d["strategies"][s]["fixed"]["x"]))
        rows.append([f"Best fixed, {s}"] + [f(sum(a), 0) for a in p.irr_mm] +
                    [f(sum(sum(a) for a in p.irr_mm), 0)])
    head = ["Plan"] + [f"{i + 1}: {CROP[c]}" for i, c in enumerate(TRIAL_ROTATION)] + ["Total"]
    return table(head, rows)


# --- validation context (crop biases, rule: state them next to rotation results) ---------------
def bias_table() -> str:
    v = json.loads((RUNS / "ltm_validation.json").read_text())
    rows = []
    for h in v["harvest"]:
        if h["treatment"] != "52A":
            continue
        rows.append([h["year"], CROP.get(h["crop"], h["crop"]), f(h.get("biomass_pct_err"), 0, True),
                     f(h.get("yield_pct_err"), 0, True)])
    y = pd.read_csv(ROOT / "data" / "ltm" / "yield.csv")
    pot = y[(y.treatment == "52A") & (y.crop == "potato")].yield_fresh_mg_ha.mean()
    pot_dm = pot * econ.POTA_DM
    sim = next(h["sim_yield_dm_t"] for h in v["harvest"]
               if h["treatment"] == "52A" and h["crop"] == "potato")
    sb = v["sugarbeet_root_dm"]
    note = (f"Potato tuber: measured {pot:.1f} Mg/ha fresh × {econ.POTA_DM:.3f} DM = "
            f"{pot_dm:.1f} t DM/ha vs {sim:.1f} simulated ({(sim / pot_dm - 1) * 100:+.0f} %). "
            f"Sugarbeet root DM (2014/2018/2020 mean): 52A {sb['52A']['pct_err']:+.1f} %, "
            f"fertiliser-only {sb['fert']['pct_err']:+.1f} %, control "
            f"{sb['control']['pct_err']:+.1f} %.")
    return table(["Year", "Crop (52A)", "Biomass error", "Yield error"], rows) + \
        f"<p class='note'>{esc(note)}</p>"


def figures(D: dict, E: dict) -> dict:
    names = {n: name for n, (_, name) in EXPS.items()}
    v1 = D[1]["strategies"]
    rot_rows = [("Trial rotation (default)", list(TRIAL_ROTATION))]
    for s in STRATS:
        fx = list(v1[s]["fixed"]["crops"])
        same = [sd for sd in v1[s]["es"] if list(v1[s]["es"][sd]["frozen_crops"]) == fx]
        rot_rows.append((f"{s.capitalize()}: fixed (= policy, {len(same)}/{len(v1[s]['es'])} seeds)", fx))
        for sd in v1[s]["es"]:
            if sd not in same:
                rot_rows.append((f"{s.capitalize()}: policy seed {sd}",
                                 list(v1[s]["es"][sd]["frozen_crops"])))
    short = {k: CROP[k] for k in fig.CROP_COL}
    short["wwht"], short["corn"] = "sp. wheat", "corn sil."

    def n_means(x):
        p = plan_from_x("N", np.asarray(x))
        return (st.mean(m * MANURE_MG for m in p.manure_x),
                sum(sum(a) for a in p.n_kg if a is not None) / len(p.n_kg))
    v2 = D[2]["strategies"]
    rows = [("Trial practice", fig.TRIAL, n_means(measured_x("N")))]
    for s in STRATS:
        rows.append((f"Best fixed, {s}", fig.STRAT_COL[s], n_means(v2[s]["fixed"]["x"])))
    for s in STRATS:
        res = es_runs(2, s, v2[s]["es"])
        rows.append((f"Learned policy, {s}", fig.STRAT_COL[s],
                     (st.mean(e["manure_dm_mg"] for r in res for e in r["test"]),
                      st.mean(e["mineral_n_kg"] for r in res for e in r["test"]))))
    n_inputs = fig.hbars([("Fall manure", "Mg DM/ha/yr", [(a, c, v[0]) for a, c, v in rows]),
                          ("Synthetic N (urea)", "kg N/ha/yr", [(a, c, v[1]) for a, c, v in rows])])

    def irr(x):
        return [sum(a) for a in plan_from_x("I", np.asarray(x)).irr_mm]
    v3 = D[3]["strategies"]
    irrigation = fig.season_bars(
        [CROP[c] for c in TRIAL_ROTATION],
        [("default (trial water)", fig.DEFAULT, irr(measured_x("I"))),
         ("best fixed, profit", fig.STRAT_COL["profit"], irr(v3["profit"]["fixed"]["x"])),
         ("best fixed, balanced", fig.STRAT_COL["balanced"], irr(v3["balanced"]["fixed"]["x"]))],
        "mm per season")
    # efficiency cases: one sequential hue, light (furrow) -> dark (drip)
    eff_rows = []
    for e, d in E.items():
        tm, v = d["strategies"]["profit"]["test_means"], d["strategies"]["profit"]
        col = EFF_COL[e]
        eff_rows += [(f"{e:.0%}: default", col, tm["default"]),
                     (f"{e:.0%}: auto-irrigation", col, tm["auto"]),
                     (f"{e:.0%}: best fixed", col, tm["fixed"]),
                     (f"{e:.0%}: learned policy", col, es_mean(v))]
    meas = E[0.80]["strategies"]["profit"]["test_means"]["measured"]
    eff_rows = [("Trial practice (100 %)", fig.TRIAL, meas)] + eff_rows
    efficiency = fig.hbars([("Profit", "$/ha/yr", [(a, c, m["profit"]) for a, c, m in eff_rows]),
                            ("Irrigation, gross", "mm/yr",
                             [(a, c, m["water"]) for a, c, m in eff_rows])])
    eff_seasons = fig.season_bars(
        [CROP[c] for c in TRIAL_ROTATION],
        [("trial water", fig.DEFAULT, irr(measured_x("I")))] +
        [(f"best fixed, profit, {e:.0%}", EFF_COL[e], irr(d["strategies"]["profit"]["fixed"]["x"]))
         for e, d in E.items()], "mm per season (gross)")
    return {"tradeoff": fig.tradeoff(D, es_mean, names), "forest": fig.forest(D, names),
            "rotations": fig.rotations(rot_rows, short), "n_inputs": n_inputs,
            "irrigation": irrigation, "efficiency": efficiency, "eff_seasons": eff_seasons}


def eff_table(E: dict) -> str:
    """Exp 3 across irrigation efficiencies, profit and balanced strategies."""
    rows = []
    for s in ("profit", "balanced"):
        for e, d in E.items():
            v = d["strategies"][s]
            tm = v["test_means"]
            pol = [v["es"][sd]["policy_vs_fixed"]["profit"]["mean"] for sd in v["es"]]
            rows.append([s if e == min(E) else "", f"{e:.0%}", f(tm["default"]["profit"], 0),
                         f(tm["auto"]["profit"], 0), f(tm["fixed"]["profit"], 0),
                         cmp_cell(v["fixed_vs_measured"], "profit", 0),
                         cmp_cell(v["fixed_vs_measured"], "no3", 0),
                         f(v["fixed_vs_measured"]["water"]["mean"], 0, True),
                         f"{f(min(pol), 0, True)} to {f(max(pol), 0, True)}",
                         esc(v["resolvable"]["adaptivity_value"])])
    return table(["Strategy", "Efficiency", "default profit", "Auto profit", "Best fixed profit",
                  "Fixed − measured, profit $/ha/yr", "Fixed − measured, NO₃-N kg/ha/yr",
                  "Fixed − measured, water mm/yr", "Policy − fixed, profit (4 seeds)",
                  "Adaptivity, sign"], rows)


def main() -> None:
    D = {n: load(n) for n in EXPS}
    E = {e: json.loads((RUNS / f"{t}.json").read_text()) for e, t in EFFS.items()}
    c1 = D[1]["config"]
    ref = c1["reference"]
    wall = {n: D[n]["wall_s"] / 3600 for n in EXPS}
    head_rows = []
    for n, (lever, name) in EXPS.items():
        for s in STRATS:
            v = D[n]["strategies"][s]
            tm = v["test_means"]
            esm = es_mean(v)
            head_rows.append([f"{n}: {esc(name)}" if s == "profit" else "", s,
                              f(tm["default"]["score"], 2), f(tm["fixed"]["score"], 2),
                              f(esm["score"], 2),
                              f(esm["profit"] - tm["default"]["profit"], 0, True),
                              f(esm["no3"] - tm["default"]["no3"], 1, True),
                              esc(v["resolvable"]["policy_vs_fixed"]),
                              esc(v["resolvable"]["adaptivity_value"])])
    headline = table(["Experiment", "Strategy", "default score", "Best fixed score",
                      "Policy score", "Policy − default, profit $/ha/yr",
                      "Policy − default, NO₃-N kg/ha/yr", "Policy vs fixed, sign",
                      "Adaptivity, sign"], head_rows)

    figs = figures(D, E)
    sec = {}
    for n, (lever, name) in EXPS.items():
        d = D[n]
        parts = []
        for s in STRATS:
            v = d["strategies"][s]
            parts.append(f"<h4>{s.capitalize()} (w = {W[s]:g})</h4>" + arms_table(v) +
                         seeds_table(v, lever == "I") +
                         f"<p class='note'>Profit, policy − default, per seed ($/ha/yr): "
                         f"{profit_row(v, 'policy_vs_default')}; policy − fixed: "
                         f"{profit_row(v, 'policy_vs_fixed')}.</p>")
        sec[n] = "".join(parts)

    # values quoted in the prose (computed here, never typed)
    c2 = D[2]["config"]
    r1p, r1s = D[1]["strategies"]["profit"], D[1]["strategies"]["sustainability"]
    r2p, r3p = D[2]["strategies"]["profit"], D[3]["strategies"]["profit"]
    r3s = D[3]["strategies"]["sustainability"]
    lose = [d["strategies"][s]["es"][sd]["policy_vs_fixed"]["profit"]["mean"]
            for d in [D[2], *E.values()] for s in ("profit", "balanced")
            for sd in d["strategies"][s]["es"]]
    n1_same = sum(list(v["es"][sd]["frozen_crops"]) == list(v["fixed"]["crops"])
                  for v in D[1]["strategies"].values() for sd in v["es"])
    n_cyc = r1p["fixed"]["n_candidates"]
    p2 = plan_from_x("N", np.asarray(r2p["fixed"]["x"]))
    n2_txt = " and ".join(f"season {k + 1} ({CROP[TRIAL_ROTATION[k]]}, {sum(a):.0f} kg N/ha)"
                          for k, a in enumerate(p2.n_kg) if a is not None and sum(a) > 0)

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Long-term manure experiments: results</title>
<style>
  body {{ font-family: Calibri, sans-serif; background: #fff; color: #000; margin: 2em auto;
         max-width: 1100px; padding: 0 1em; line-height: 1.4; }}
  table {{ border-collapse: collapse; margin: 0.6em 0 1.2em; font-size: 0.93em; }}
  th, td {{ border: 1px solid #999; padding: 3px 8px; text-align: left; vertical-align: top; }}
  th {{ font-weight: bold; }}
  code {{ font-family: Calibri, sans-serif; font-style: italic; }}
  .note {{ font-size: 0.92em; }}
  .ci {{ font-size: 0.88em; }}
  .flag {{ border-left: 3px solid #000; padding-left: 0.6em; }}
  figure {{ margin: 1em 0 1.6em; overflow-x: auto; }}
  figcaption {{ font-size: 0.92em; margin-bottom: 0.4em; max-width: 900px; }}
</style></head><body>
<h1>Long-term manure experiments: results (v3)</h1>
<p>Run v3, 2026-10-27/28. It includes the collaborators' setup notes: 2012 pinto beans as
spin-up (auto-irrigated, 100 kg N + 50 kg P/ha at planting), the 2014 sugarbeet kill on
Oct 28, and manure at 5,200 kg/ha × each year's fractions on Oct 20. It adds rotation rules,
water scored gross at three irrigation efficiencies, and halved budgets. Budgets per ES seed are
{c1['budget']:,} HRU-episode steps for Exp 1 and {c2['budget']:,} for Exp 2/3, with
{c1['seeds']} seeds and three strategies. The run uses branch <code>ltm-scenario</code>; nothing
is committed. This page replaces the v2 results, which are archived in
<code>runs/archive/ltm_v2_20261026/</code>.</p>
<p><a href="#headline">Headline</a> · <a href="#read">How to read</a> · <a href="#e1">Exp 1</a> ·
<a href="#e2">Exp 2</a> · <a href="#e3">Exp 3</a> · <a href="#eff">Efficiency</a> ·
<a href="#caveats">Caveats</a> · <a href="#log">Run log</a> · <a href="#next">Next</a></p>

<h2 id="headline">1. Headline</h2>
<ul>
<li><strong>Each lever's best fixed plan beats the trial's practice on score</strong> on the
test years in every experiment × strategy cell except one, with intervals that exclude 0. The
exception is irrigation at 55 % efficiency under the profit strategy, which ties.</li>
<li><strong>Exp 1, rotation:</strong> with the rules in place, the profit and balanced
strategies pick potato → sugarbeet → barley, repeated. That earns {cmp_cell(r1p['fixed_vs_default'], 'profit', 0)}
$/ha/yr against the trial rotation. Sustainability picks two 3-year alfalfa stands, each
followed by barley.</li>
<li><strong>Exp 2, nitrogen:</strong> almost no manure and almost no synthetic N again, for
{cmp_cell(r2p['fixed_vs_measured'], 'profit', 0)} $/ha/yr and
{cmp_cell(r2p['fixed_vs_measured'], 'no3', 0)} kg NO₃-N/ha/yr against the trial.</li>
<li><strong>Exp 3, irrigation:</strong> the gain over the trial depends on efficiency. Under
the profit strategy it is {cmp_cell(E[0.90]['strategies']['profit']['fixed_vs_measured'], 'profit', 0)}
$/ha/yr at 90 %, {cmp_cell(E[0.80]['strategies']['profit']['fixed_vs_measured'], 'profit', 0)}
at 80 % and {cmp_cell(E[0.55]['strategies']['profit']['fixed_vs_measured'], 'profit', 0)} at
55 %. At 55 % the extra pumping takes the whole gain.</li>
<li><strong>The learned policy never resolvably beats the best fixed plan.</strong> In Exp 2 and Exp 3,
under the profit and balanced strategies, it is worse on every seed, by {f(-max(lose), 0)}–{f(-min(lose), 0)}
$/ha/yr. In Exp 1 it picks the fixed plan's rotation on {n1_same} of 12 seed × strategy runs.</li>
<li><strong>Adapting to the field's state is worth about zero</strong> in all 15
experiment × strategy cells: each policy scores the same as its own plan frozen and replayed
blind. The optimiser's gains come from search, not from closed-loop adaptation. This is the
distinction the evaluation protocol was built to make.</li>
</ul>
<figure><figcaption><strong>Figure 1.</strong> Where every arm lands on the test years:
profit against nitrate leaching. Exp 3 is the 80 % case. Filled dot = best fixed plan,
ring = learned policy (mean of 4 seeds); a ring around a dot of the same colour means the policy
equals the fixed plan. Each panel has its own profit scale. In Exp 1 the profit and balanced
strategies pick the same rotation, so the orange mark covers the blue one. Hover a mark for its
numbers.</figcaption>{figs['tradeoff']}</figure>
<figure><figcaption><strong>Figure 2.</strong> The protocol's two questions, per seed (4 lines per
row) with 95 % intervals (effective n {c1['test_ess']:.2f}). Left: does the learned policy beat the best
fixed plan? Right: does reacting to the field's state add anything over the policy's own plan
replayed blind?</figcaption>{figs['forest']}</figure>
{headline}
<p class="note">Scores are test-window means (7 windows, effective n {c1['test_ess']:.2f}).
Measured 52A practice on the train windows scores exactly +1 / 0 / −1. "Sign" =
<em>consistent</em> if all 4 seeds agree in sign. It says nothing about whether the difference
is distinguishable from zero; see each experiment's per-seed intervals for that.</p>

<h2 id="read">2. How to read the scores</h2>
<p><code>score = w·profit/{f(ref['profit'], 0)} − (1−w)·mean(NO₃/{f(ref['terms']['no3'], 1)},
water/{f(ref['terms']['water'], 0)}, P/{f(ref['terms']['p'], 1)}, N₂O/{f(ref['terms']['n2o'], 2)})</code>.
The denominators are measured 52A practice averaged over the train windows ($/ha/yr, kg/ha/yr,
mm/yr). A profit score of 1.00 means the trial's profit; a sustainability score of −0.5 means
half the trial's average footprint. w = 1 is profit, 0.5 balanced, 0 sustainability.
<strong>Water is gross</strong>: what is pumped, which is the delivered amount divided by
irrigation efficiency.</p>
<p>Arms, all on test windows starting 2011–2017 (train windows start 1995–2001; no calendar
year is shared):</p>
<ul>
<li><strong>measured</strong>: the trial's own management replayed day by day, with its
water at 100 % efficiency.</li>
<li><strong>default</strong>: the same practice expressed through the lever. In Exp 1 that is
the trial rotation with auto-irrigation; in Exp 3 it is the trial's depths as monthly amounts,
pumped at the case's efficiency. <strong>Quote gains against default.</strong> Measured and
default differ through irrigation event size alone: the trial's 17–58 mm events leach
{f(r1p['test_means']['measured']['no3'], 0)} kg NO₃-N/ha/yr, against
{f(r1p['test_means']['default']['no3'], 0)} under 1-inch auto passes (Exp 1) and
{f(r3p['test_means']['default']['no3'], 0)} as monthly depths (Exp 3).</li>
<li><strong>fixed</strong>: the best open-loop plan, picked on train. Exp 1 searches
exhaustively over every rule-abiding rotation that repeats a cycle of ≤ 4 years; Exp 2/3 use
CMA-ES at a matched budget.</li>
<li><strong>policy</strong>: the ES closed-loop policy, picked on train.
<strong>Adaptivity</strong> = the policy minus the plan it chose on train, frozen and replayed
on test.</li>
</ul>
<p class="note">Bracketed intervals are <code>ci95_ess</code> (bootstrap with effective n
{c1['test_ess']:.2f}), never sd/√n.</p>

<h2 id="e1">3. Exp 1: crop rotation</h2>
<p>A crop each year from barley, sugarbeet, spring wheat, potato, corn silage and alfalfa (a
3-year stand). The rules, decided 2026-10-27:</p>
<ul>
<li>no crop two years running, except within one alfalfa stand;</li>
<li>no stand straight after another stand;</li>
<li>potato at most once in any 3 years.</li>
</ul>
<p>{n_cyc} cycles of ≤ 4 years satisfy them. 52A manure goes on each fall except into standing
alfalfa, and auto-irrigation runs every season at 80 %.</p>
<figure><figcaption><strong>Figure 3.</strong> Rotations, year by year: each strategy's best
fixed rotation (and how many of the 4 policy seeds chose the same), then each seed whose policy
chose differently.</figcaption>{figs['rotations']}</figure>
{exp1_plans(D[1])}
<p class="flag"><strong>Potato as often as the rule allows wins on profit and balanced.</strong>
Potato comes back every third year, with sugarbeet and barley between. Against default that is
{cmp_cell(r1p['fixed_vs_default'], 'profit', 0)} $/ha/yr, with no resolvable change in leaching
({cmp_cell(r1p['fixed_vs_default'], 'no3', 1)} kg/ha/yr). It wins even though the model
under-yields potato (see the table in <a href="#caveats">caveats</a>). SWAT+ also does not
simulate the disease load that builds up within the rules, so report this as the model's
answer, not as advice.</p>
<p class="flag"><strong>Sustainability picks alfalfa stands with barley between.</strong> P
loss + buildup falls from {f(r1s['test_means']['default']['p'], 1)} to
{f(r1s['test_means']['fixed']['p'], 1)} kg/ha/yr and N₂O from
{f(r1s['test_means']['default']['n2o'], 1)} to {f(r1s['test_means']['fixed']['n2o'], 1)}. Against
default it costs {cmp_cell(r1s['fixed_vs_default'], 'profit', 0)} $/ha/yr and
{cmp_cell(r1s['fixed_vs_default'], 'water', 0)} more mm of water per year, because alfalfa is
irrigated all season. Where a seed's policy chose a different rotation (Figure 3), it is no
better than the fixed plan: profit/balanced seeds tie within the interval, and sustainability's
seed 0 (spring wheat between the stands) scores slightly lower, with an interval that excludes 0.</p>
{sec[1]}

<h2 id="e2">4. Exp 2: nitrogen, manure vs synthetic</h2>
<p>Fall manure of 0–52 Mg DM/ha each year plus monthly synthetic N, on the trial rotation and
irrigation.</p>
<figure><figcaption><strong>Figure 4.</strong> Nitrogen inputs, 8-year means. The trial applies
52 Mg DM/ha of manure every fall; every optimised plan applies almost none.</figcaption>{figs['n_inputs']}</figure>
{exp2_plans(D[2])}
<p class="flag"><strong>Near-zero N wins because soil N supply is generous in the
model.</strong> After the active-humus fit, the unfertilised control's sugarbeet is still
over-predicted by {json.loads((RUNS / 'ltm_validation.json').read_text())['sugarbeet_root_dm']['control']['pct_err']:.0f} %,
so cutting N costs less yield in the model than in the field. Treat the size of the cut as an
upper bound. P falls to about 0 because with almost no manure there is no labile-P buildup.
The profit strategy's fixed plan applies synthetic N in only {n2_txt}.</p>
<p class="flag"><strong>The learned policy applies exactly zero manure and zero N</strong> on
every seed and test window, under every strategy: it has collapsed to the lower bound. That is
why it trails the fixed plan by {f(-r2p['es'][next(iter(r2p['es']))]['policy_vs_fixed']['profit']['mean'], 0)}
$/ha/yr under the profit strategy, identically on all 4 seeds. This is an optimiser outcome,
not adaptation.</p>
{sec[2]}

<h2 id="e3">5. Exp 3: irrigation (80 % efficiency, base case)</h2>
<p>Monthly depths for all 8 seasons, with the trial's rotation, manure and N. The efficiency
cases are in <a href="#eff">section 6</a>. Seasonal totals (mm, gross):</p>
<figure><figcaption><strong>Figure 5.</strong> Irrigation per season: the trial's water (as
monthly depths) against the best fixed schedules. The sustainability schedule is zero in every
season and is not drawn.</figcaption>{figs['irrigation']}</figure>
{exp3_plans(D[3])}
<p class="flag"><strong>The sustainability optimum is no irrigation</strong>: zero water and
almost zero leaching, at a profit of {f(r3s['test_means']['fixed']['profit'], 0)} $/ha/yr because
the crops fail. This is the degenerate end of the strategy scale (w = 0 puts no value on
profit), not a recommendation. A profit floor would remove it; that decision is still open (see
<a href="#next">Next</a>).</p>
{sec[3]}

<h2 id="eff">6. Exp 3 across irrigation efficiencies</h2>
<p>The same experiment at three efficiencies, from UNL EC732 Table 2 midpoints: 55 % (furrow),
80 % (centre pivot / sprinkler; the base case) and 90 % (drip). The efficiency applies to
default, auto-irrigation and every optimised schedule. The trial's measured water stays at
100 %: at 80 % the measured profile never drains in 2013–2020, so the score could not be
normalised (OPEN_ITEMS L13). The question of whether the recorded amounts are pumped or
delivered is with the collaborators.</p>
<figure><figcaption><strong>Figure 6.</strong> Profit strategy at each efficiency (lighter =
lower efficiency) against the trial's practice. Hover a bar for its value.</figcaption>{figs['efficiency']}</figure>
<figure><figcaption><strong>Figure 7.</strong> The profit strategy's best fixed schedule per
season, at each efficiency.</figcaption>{figs['eff_seasons']}</figure>
{eff_table(E)}
<p class="flag"><strong>Efficiency decides whether optimised irrigation beats the
trial.</strong></p>
<ul>
<li><strong>At 55 %</strong>, default delivers only 55 % of the trial's depths to the crop, so
its profit falls to {f(E[0.55]['strategies']['profit']['test_means']['default']['profit'], 0)}
$/ha/yr. The best fixed plan pumps
{f(E[0.55]['strategies']['profit']['fixed_vs_measured']['water']['mean'], 0)} mm/yr more than the
trial and recovers the trial's profit, but no more.</li>
<li><strong>At 90 %</strong>, the profit and balanced plans beat the trial on profit, water and
leaching at once. The balanced plan earns
{f(E[0.90]['strategies']['balanced']['fixed_vs_measured']['profit']['mean'], 0)} $/ha/yr more
with {f(-E[0.90]['strategies']['balanced']['fixed_vs_measured']['water']['mean'], 0)} mm/yr less
water.</li>
<li>Most of the leaching drop against measured comes from the change from the trial's large
events to monthly depths, which default already shows, not from water saved.</li>
</ul>

<h2 id="caveats">7. Caveats to state with any result</h2>
<ul>
<li><strong>Crop yields are Ai-SWAT-Plus's, copied verbatim</strong> (rule 11b). With rotation
as a lever, their biases no longer cancel. Validation errors (52A):</li>
</ul>
{bias_table()}
<ul>
<li><strong>Leaching is unvalidated on site</strong>; its magnitudes are within-model only. 52A
over-mineralises after the soil N fit, so its leaching is an upper bound.</li>
<li><strong>N₂O is IPCC 2019 Tier 1</strong>, computed from N inputs and leaching, never
simulated.</li>
<li><strong>Measured and generated water are at different efficiencies</strong> (100 % and the
case's value). Comparisons with <code>measured</code> carry that assumption; comparisons with
<code>default</code> do not.</li>
<li><strong>Effective n is {c1['test_ess']:.2f}</strong> (7 overlapping test windows). Seed
agreement is strong, but the test-window evidence is thin.</li>
<li><strong>Prices</strong> are 2022–2024 NASS averages, with potato at
{f(econ.prices().base.crop['pota'], 0)} $/Mg DM ({econ.POTA_DM:.3f} DM, USDA FDC 170027).
Operating costs come from UI enterprise budgets escalated to 2022–2024 dollars.</li>
</ul>

<h2 id="log">8. Run log</h2>
<ul>
<li>All five runs started together on 2026-10-27 at 22:53, each with an ES pool covering all
its jobs, so no core idled while a run's last jobs finished.</li>
<li>Speed-ups that leave results unchanged (each pinned by a test):
<ul>
<li>intermediate closed-loop steps simulate only up to the next decision's year;</li>
<li>a single-pass output reader;</li>
<li>numpy observations;</li>
<li>an output cache for Exp 1.</li>
</ul></li>
<li>Exp 1 stopped at 00:39 on a bug in that cache: a full cache could evict an entry the same
call still needed. The fix is pinned by <code>test_cache_smaller_than_a_call</code>. The
restart at 01:26 reused the 7 finished seeds and was done at 02:22 ({wall[1]:.1f} h).</li>
<li>Exp 2 was done at 03:26 ({wall[2]:.1f} h). The Exp 3 runs were done between 05:04 and 05:08
(80 %: {wall[3]:.1f} h). <strong>No SWAT+ run failed</strong> (0 failed genomes in all 60 ES
jobs).</li>
</ul>

<h2 id="next">9. Next</h2>
<ul>
<li><strong>Decide:</strong> add a profit floor to the sustainability strategy (for example,
profit no worse than default) so that "no irrigation" is excluded. That needs a rerun of Exp 3
sustainability only, about 3 h for all three efficiency cases.</li>
<li>Send the collaborator email (draft ready). It asks whether the recorded irrigation amounts
are pumped or delivered, which decides the measured replay's efficiency.</li>
<li>Rewrite PAPER.md for the LTM environment (it still describes GRACEnet).</li>
<li>Commit, when you ask.</li>
</ul>
<p class="note">Generated by <code>scripts/ltm_results.py</code> from <code>runs/ltm_exp{{1,2,3}}.json</code>
and <code>runs/ltm_exp3_eff{{55,90}}.json</code>.</p>
</body></html>
"""
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
