#!/usr/bin/env python3
"""Fit the LTM scenario's active-humus fraction to the trial's control plots.

Soil parameter only; crop parameters are untouched (CLAUDE.md rule 11b, user decision
2026-10-25). One parameter against one measured number:

  target   net seasonal inorganic-N mineralisation of the CONTROL plots in years without
           residue the previous fall (2013, 2014, 2016, 2018, 2020): 72 kg N/ha
           (Leytem et al. 2024, sec. 3.5; control and fertilizer did not differ)
  simulated  hru_nb_day act_nit_n of the control HRU, summed over each year's buried-bag
           window (P1 Table 4), averaged over the same years

Why the control: with no manure and straw baled off, its mineralisation is almost all active
humus (act_nit_n, which the engine reports). Manure organic N is split by rtof (0.5, the
default without a carbon file) between residue and humus, and rev 62 never adds residue
mineralisation to an output counter (nut_nminrl.f90), so manured plots have no reported
comparator. Known bias: simulated = whole profile, measured = 0-60 cm bags.

Parameter: the active share of soil humus N, f, set as both the initial fraction
(nutrients.sol fr_hum_act) and the equilibrium of the active<->stable transfer (n_act_frac via
calibration.cal), so the pool starts at equilibrium. The rate orgn_min stays at the inherited
0.0008: even at its 0.006 ceiling the default pool (f = 0.02) gives 33 of the 72 kg N/ha on the
stand-in weather, so pool size, not rate, is what is short.

Held out (scored by scripts/ltm_validate.py): control and fertilizer sugarbeet root DM, the
2020 soil-NO3 treatment contrast, 52A uptake, 52A humus mineralisation (a lower bound).

    uv run python scripts/ltm_calibrate_n.py          # report
    uv run python scripts/ltm_calibrate_n.py --apply  # -> model/ltm/n_cycle.json; rebuild
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swat_gym.fastrunner import FastRunner  # noqa: E402

TIO = ROOT / "model" / "ltm" / "TxtInOut"
DATA = ROOT / "data" / "ltm"
OUT_JSON = ROOT / "model" / "ltm" / "n_cycle.json"
TARGET, TARGET_YEARS = 72.0, (2013, 2014, 2016, 2018, 2020)
CONTROL_HRU = 3
BOUNDS = (0.02, 0.25)   # engine default .. top of Ai-SWAT-Plus's fr_hum_act range (0.010-0.250)
sys.path.insert(0, str(ROOT / "scripts"))
from build_ltm import set_active_fraction  # noqa: E402


def bag_windows() -> dict[int, tuple[date, date]]:
    nm = pd.read_csv(DATA / "nmin.csv").dropna(subset=["bag_in"])
    return {int(r.year): (date.fromisoformat(r.bag_in), date.fromisoformat(r.bag_out))
            for r in nm.itertuples()}


def seasonal(nb: pd.DataFrame, hru: int) -> pd.Series:
    d = nb[nb.unit == hru].copy()
    d["date"] = pd.to_datetime(dict(year=d.yr, month=d.mon, day=d.day))
    out = {}
    for y, (a, b) in bag_windows().items():
        w = d[(d.date >= pd.Timestamp(a)) & (d.date <= pd.Timestamp(b))]
        out[y] = float(w.act_nit_n.sum())
    return pd.Series(out)


class Model:
    def __init__(self) -> None:
        names = ("nutrients.sol", "calibration.cal", "cal_parms.cal")
        self.fr = FastRunner(TIO, trim_outputs=False, editable=frozenset(names))
        self.base = [(TIO / n).read_text() for n in names]
        self.cache: dict[float, pd.DataFrame] = {}

    def nb(self, x: float) -> pd.DataFrame:
        if x not in self.cache:
            nut, cal, cp = set_active_fraction(*self.base, x)
            self.fr.run({"nutrients.sol": nut, "calibration.cal": cal, "cal_parms.cal": cp})
            self.cache[x] = self.fr.read("hru_nb_day.txt")
        return self.cache[x]

    def control_mean(self, x: float) -> float:
        return float(seasonal(self.nb(x), CONTROL_HRU)[list(TARGET_YEARS)].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if OUT_JSON.exists():
        raise SystemExit(f"{OUT_JSON.relative_to(ROOT)} exists: the model already carries a fit. "  # noqa
                         "Delete it and rebuild (scripts/build_ltm.py) to refit from the base.")
    m = Model()
    x0 = float(m.base[0].splitlines()[2].split()[4])
    lo, hi = (m.control_mean(b) - TARGET for b in BOUNDS)
    print(f"f {x0} (engine default): control seasonal mineralisation "
          f"{m.control_mean(x0):.1f} kg N/ha vs measured {TARGET}")
    if lo > 0 or hi < 0:
        raise SystemExit(f"target not bracketed on {BOUNDS}: f = {lo + TARGET:.1f} .. "
                         f"{hi + TARGET:.1f}")
    x = brentq(lambda v: m.control_mean(v) - TARGET, *BOUNDS, xtol=1e-6)
    per_year = pd.DataFrame({
        "control_sim": seasonal(m.nb(x), CONTROL_HRU),
        "52A_sim_humus_only": seasonal(m.nb(x), 1),
        "52A_meas": pd.read_csv(DATA / "nmin.csv").query("treatment == '52A'")
                      .set_index("year").net_in_kg_ha})
    print(f"fitted f {x:.5f}: control mean {m.control_mean(x):.1f} over {TARGET_YEARS}")
    print(per_year.round(1).to_string())
    rec = {"f_act": round(x, 5), "default_f_act": x0, "orgn_min_kept": 0.0008, "target_kg_ha": TARGET,
           "target_years": TARGET_YEARS, "fitted": date.today().isoformat(),
           "engine_runs": m.fr.n_runs,
           "source": "Leytem et al. 2024 sec. 3.5 (control/fertilizer net INs, non-residue years)"}
    m.fr.close()
    if args.apply:
        OUT_JSON.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"wrote {OUT_JSON.relative_to(ROOT)}; now run scripts/build_ltm.py")


if __name__ == "__main__":
    main()
