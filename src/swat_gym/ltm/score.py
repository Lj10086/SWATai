"""Relative yield, sustainability terms, and the three strategies' score, per episode (per ha
per year). Profit is computed and reported, never scored (user decision 2026-10-30):

    profit = revenue - water - manure - mineral N - MAP P - passes - crop operating costs

Relative yield (ry): each season's harvested dry yield (alfalfa: all cuts of the year) divided
by its crop's reference yield, mean over the 8 seasons. A crop's reference is its train-window
mean yield when substituted into the default rotation (experiments.ltm.crop_reference).

Sustainability terms (CLAUDE.md "Locked design"), means over the 8 seasons:
  no3     NO3-N leached below the profile, kg/ha/yr      (hru_ls_yr percn)
  water   gross irrigation applied, mm/yr: hru_wb_yr irr (what reaches the soil, amt x eff)
          divided by that season's efficiency (render.season_irr_eff)
  p       P lost (surface soluble + sediment + leached) plus any net rise in profile solution P
          between Jan 2 of season 1 and Dec 30 of season 8 (mgt_out probes), kg/ha/yr
  n2o     IPCC Tier 1 on applied manure + mineral N and leached NO3 (ipcc_n2o below),
          kg N2O/ha/yr -- accounting only; SWAT+ does not simulate N2O

Strategy w in {1, 0.5, 0} (user decision 2026-10-25, retiring rule 1's frontier; yield in
place of profit 2026-10-30):
    score = w * ry / ry_ref - (1 - w) * mean_k(term_k / term_ref_k)
with *_ref the train-window means under measured 52A practice (``Reference``). Nothing prices
water or N at w = 1 (accepted, 2026-10-30): the pure-yield strategy may over-apply both.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .econ import LtmPrices
from .practice import MAP, MAP_N, MAP_P, N_PRODUCT, SPINUP_FERT
from .render import PROBE, EpisodePlan, manure_applied, season_irr_eff, season_of, stand_year
from .windows import SEASONS

# IPCC 2019 Tier 1 defaults. Accounting only: SWAT+ does not simulate N2O.
# EF5 is applied to simulated no3_rchg instead of FracLEACH x applied N.
# Verified 2026-10-25 against the 2019 Refinement to the 2006 IPCC Guidelines, Vol. 4 Ch. 11
# (Hergoualc'h et al. 2019), Table 11.1 (updated: aggregated EF1 0.010) and Table 11.3
# (updated: aggregated EF4 0.010, EF5 0.011, FracGASF 0.11, FracGASM 0.21). Kimberly is a dry
# climate there (annual P/PET < 1), whose disaggregated (Tier 2 option) EF1 and EF4 are 0.005.
IPCC_EF1 = 0.010        #: direct N2O-N per kg N applied
IPCC_EF4 = 0.010        #: N2O-N per kg N volatilised
IPCC_EF5 = 0.011        #: N2O-N per kg N leached
IPCC_FRACGASF = 0.11    #: volatilised fraction, synthetic N
IPCC_FRACGASM = 0.21    #: volatilised fraction, manure N
N2O_N_TO_N2O = 44.0 / 28.0


def ipcc_n2o(*, manure_n_kg: float, mineral_n_kg: float, no3_leached_kg: float) -> dict:
    """IPCC Tier 1 N2O (kg N2O/ha), split by pathway."""
    n_applied = float(manure_n_kg) + float(mineral_n_kg)
    direct = n_applied * IPCC_EF1
    volatilised = (float(mineral_n_kg) * IPCC_FRACGASF
                   + float(manure_n_kg) * IPCC_FRACGASM)
    indirect_vol = volatilised * IPCC_EF4
    indirect_leach = float(no3_leached_kg) * IPCC_EF5
    n2o_n = direct + indirect_vol + indirect_leach
    return {
        "n2o_kg": n2o_n * N2O_N_TO_N2O,
        "n2o_direct_kg": direct * N2O_N_TO_N2O,
        "n2o_volat_kg": indirect_vol * N2O_N_TO_N2O,
        "n2o_leach_kg": indirect_leach * N2O_N_TO_N2O,
        "n_applied_kg": n_applied,
    }


TERMS = ("no3", "water", "p", "n2o")
STRATEGIES = {"yield": 1.0, "balanced": 0.5, "sustainability": 0.0}
P_LOSS = ("surqsolp", "sedorgp", "sedminp", "lchlabp")


def _fert_events(mgt: pd.DataFrame) -> pd.DataFrame:
    f = mgt[mgt.operation == "FERT"]
    f = f[~f.name.isin([p for p, _ in SPINUP_FERT])]       # spin-up beans: not scored
    return f[~((f.name == PROBE) & (f.op_var == 0.0))]


def episode(plan: EpisodePlan, out: dict, prices: LtmPrices, start: int) -> dict:
    """Per-ha-per-year means over the 8 seasons, every term separable."""
    mgt = out["mgt"]
    years = range(start + 1, start + SEASONS + 1)
    hv = mgt[mgt.operation.isin(["HARV/KILL", "HARVEST"]) & mgt.yr.isin(years)]
    revenue = float(sum(r.op_var / 1000.0 * prices.base.crop[r.name] for r in hv.itertuples()))
    yields = {int(y): float(g.op_var.sum()) / 1000.0 for y, g in hv.groupby("yr")}

    operating = 0.0
    for k in range(1, SEASONS + 1):
        crop = season_of(plan, k).crop
        operating += prices.operating[crop]
        if crop == "alfa" and stand_year(plan.crops, k) == 1:
            operating += prices.alfa_seed

    wb, ls = out["hru_wb_yr.txt"], out["hru_ls_yr.txt"]
    net = wb.set_index("yr").irr
    irr_mm = sum(float(net.get(start + k, 0.0)) / season_irr_eff(plan, k)    # gross
                 for k in range(1, SEASONS + 1))

    fert = _fert_events(mgt)
    urea_n = float(fert.loc[fert.name == N_PRODUCT, "op_var"].sum())
    map_kg = float(fert.loc[fert.name == MAP, "op_var"].sum())
    mineral_n = urea_n + MAP_N * map_kg
    map_p = MAP_P * map_kg
    manure_dm, manure_n = manure_applied(plan)
    manure_fresh_mg = 0.0
    for r in fert[fert.name.str.startswith("ltm2")].itertuples():
        fall = int(r.name[3:]) - 2012
        dm_frac = _dm_fraction(fall)
        manure_fresh_mg += r.op_var / 1000.0 / dm_frac
    passes = int(fert.assign(d=fert.yr * 10000 + fert.mon * 100 + fert.day).d.nunique())

    b = prices.base
    costs = {"water_cost": irr_mm * b.water, "manure_cost": manure_fresh_mg * b.manure,
             "fert_n_cost": mineral_n * b.fert_n, "fert_p_cost": map_p * prices.fert_p,
             "pass_cost": passes * b.fert_op, "operating_cost": operating}
    profit = revenue - sum(costs.values())

    no3 = float(ls.loc[ls.yr.isin(years), "percn"].sum())
    p_loss = float(ls.loc[ls.yr.isin(years), list(P_LOSS)].sum().sum())
    pr = mgt[(mgt.name == PROBE) & (mgt.op_var == 0.0)]
    first = pr[(pr.yr == start + 1) & (pr.mon == 1) & (pr.day == 2)].soil_solp
    last = pr[(pr.yr == start + SEASONS) & (pr.mon == 12) & (pr.day == 30)].soil_solp
    p_build = max(0.0, float(last.iloc[0]) - float(first.iloc[0]))
    n2o = ipcc_n2o(manure_n_kg=manure_n, mineral_n_kg=mineral_n, no3_leached_kg=no3)

    per = 1.0 / SEASONS
    return {
        "start": start, "profit": profit * per, "revenue": revenue * per,
        **{k: v * per for k, v in costs.items()},
        "no3": no3 * per, "water": irr_mm * per, "p": (p_loss + p_build) * per,
        "p_loss": p_loss * per, "p_buildup": p_build * per, "n2o": n2o["n2o_kg"] * per,
        "mineral_n_kg": mineral_n * per, "manure_dm_mg": manure_dm / 1000.0 * per,
        "manure_n_kg": manure_n * per, "map_p_kg": map_p * per, "passes": passes * per,
        "yields_t": yields, "crops": list(plan.crops),
        "season_yield_t": [yields.get(y, 0.0) for y in years],
    }


def relative_yield(ep: dict, crop_ref: dict) -> float:
    """Mean over the seasons of harvested yield / the season's crop's reference yield."""
    return float(np.mean([y / crop_ref[c] for c, y in zip(ep["crops"], ep["season_yield_t"])]))


@lru_cache(maxsize=None)
def _dm_fraction(fall: int) -> float:
    from .practice import TRIAL_START, _management
    m = _management()
    r = m[(m.year == TRIAL_START + fall) & (m.op == "manure")].iloc[0]
    return float(r.dm_frac_fresh)


#: Sustainability's yield floor (profit floor 2026-10-28, yield floor 2026-10-30, user
#: decisions): an episode whose relative yield is below ``Reference.floor`` (default practice's
#: train-mean ry) loses FLOOR_PENALTY x the shortfall / reference ry. Graded, so CMA-ES can
#: steer back. Without a floor w = 0 picks "no irrigation" (Exp 3 v3: crops fail).
FLOOR_PENALTY = 10.0


@dataclass(frozen=True)
class Reference:
    """Train-window means of measured practice: the strategies' denominators. ``crop_ref``:
    t/ha per crop (relative_yield). ``floor``: the sustainability strategy's relative-yield
    floor, or None for no floor."""
    ry: float
    terms: dict
    crop_ref: dict
    floor: float | None = None

    def __post_init__(self) -> None:
        low = {c: v for c, v in self.crop_ref.items() if v < 1e-3}
        if low:
            raise ValueError(f"crop reference yields ~0: {low}: cannot normalise")
        if self.ry < 1e-3:
            raise ValueError(f"reference relative yield {self.ry:.4f} is ~0: cannot normalise")
        tiny = {k: v for k, v in self.terms.items() if abs(v) < 1e-3}
        if tiny:
            raise ValueError(f"reference terms ~0 under measured practice: {tiny}. A strategy "
                             "cannot be normalised by them; report this and decide.")

    @classmethod
    def from_episodes(cls, eps: list[dict], crop_ref: dict,
                      floor: float | None = None) -> Reference:
        return cls(float(np.mean([relative_yield(e, crop_ref) for e in eps])),
                   {k: float(np.mean([e[k] for e in eps])) for k in TERMS}, dict(crop_ref), floor)

    @classmethod
    def from_key(cls, key: dict) -> Reference:
        return cls(key["ry"], key["terms"], key["crop_ref"], key.get("floor"))

    def ry_of(self, ep: dict) -> float:
        return relative_yield(ep, self.crop_ref)

    def score(self, ep: dict, w: float) -> float:
        sust = float(np.mean([ep[k] / self.terms[k] for k in TERMS]))
        ry = self.ry_of(ep)
        s = w * ry / self.ry - (1.0 - w) * sust
        if w == 0.0 and self.floor is not None and ry < self.floor:
            s -= FLOOR_PENALTY * (self.floor - ry) / self.ry
        return s

    def key(self) -> dict:
        return {"ry": self.ry, "terms": dict(self.terms), "crop_ref": dict(self.crop_ref),
                "floor": self.floor, "floor_penalty": FLOOR_PENALTY}

