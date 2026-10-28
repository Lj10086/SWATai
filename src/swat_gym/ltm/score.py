"""Profit, sustainability terms, and the three strategies' score, per episode (per ha per year).

    profit = revenue - water - manure - mineral N - MAP P - passes - crop operating costs

Sustainability terms (CLAUDE.md "Locked design"), means over the 8 seasons:
  no3     NO3-N leached below the profile, kg/ha/yr      (hru_ls_yr percn)
  water   gross irrigation applied, mm/yr: hru_wb_yr irr (what reaches the soil, amt x eff)
          divided by that season's efficiency (render.season_irr_eff)
  p       P lost (surface soluble + sediment + leached) plus any net rise in profile solution P
          between Jan 2 of season 1 and Dec 30 of season 8 (mgt_out probes), kg/ha/yr
  n2o     IPCC Tier 1 on applied manure + mineral N and leached NO3 (rewarders.ipcc_n2o),
          kg N2O/ha/yr -- accounting only; SWAT+ does not simulate N2O

Strategy w in {1, 0.5, 0} (user decision 2026-10-25, retiring rule 1's frontier):
    score = w * profit / profit_ref - (1 - w) * mean_k(term_k / term_ref_k)
with *_ref the train-window means under measured 52A practice (``Reference``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..rewarders import ipcc_n2o
from .econ import LtmPrices
from .practice import MAP, MAP_N, MAP_P, N_PRODUCT, SPINUP_FERT
from .render import PROBE, EpisodePlan, manure_applied, season_irr_eff, season_of, stand_year
from .windows import SEASONS

TERMS = ("no3", "water", "p", "n2o")
STRATEGIES = {"profit": 1.0, "balanced": 0.5, "sustainability": 0.0}
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
    }


@lru_cache(maxsize=None)
def _dm_fraction(fall: int) -> float:
    from .practice import TRIAL_START, _management
    m = _management()
    r = m[(m.year == TRIAL_START + fall) & (m.op == "manure")].iloc[0]
    return float(r.dm_frac_fresh)


#: Sustainability's profit floor (2026-10-28, user decision): an episode earning less than
#: ``Reference.floor`` (default practice's train-mean profit) loses FLOOR_PENALTY x the
#: shortfall / reference profit. Graded, so CMA-ES can steer back; $100/ha costs ~0.9 score,
#: more than any plan gains on the footprint. Without it w = 0 picks "no irrigation" (Exp 3
#: v3: crops fail, -1,921 $/ha/yr).
FLOOR_PENALTY = 10.0


@dataclass(frozen=True)
class Reference:
    """Train-window means of measured practice: the strategies' denominators. ``floor``: the
    sustainability strategy's profit floor ($/ha/yr), or None for no floor."""
    profit: float
    terms: dict
    floor: float | None = None

    def __post_init__(self) -> None:
        if abs(self.profit) < 1.0:
            raise ValueError(f"reference profit {self.profit:.2f} $/ha/yr is ~0: cannot normalise")
        tiny = {k: v for k, v in self.terms.items() if abs(v) < 1e-3}
        if tiny:
            raise ValueError(f"reference terms ~0 under measured practice: {tiny}. A strategy "
                             "cannot be normalised by them; report this and decide.")

    @classmethod
    def from_episodes(cls, eps: list[dict], floor: float | None = None) -> Reference:
        return cls(float(np.mean([e["profit"] for e in eps])),
                   {k: float(np.mean([e[k] for e in eps])) for k in TERMS}, floor)

    @classmethod
    def from_key(cls, key: dict) -> Reference:
        return cls(key["profit"], key["terms"], key.get("floor"))

    def score(self, ep: dict, w: float) -> float:
        sust = float(np.mean([ep[k] / self.terms[k] for k in TERMS]))
        s = w * ep["profit"] / self.profit - (1.0 - w) * sust
        if w == 0.0 and self.floor is not None and ep["profit"] < self.floor:
            s -= FLOOR_PENALTY * (self.floor - ep["profit"]) / self.profit
        return s

    def key(self) -> dict:
        return {"profit": self.profit, "terms": dict(self.terms), "floor": self.floor,
                "floor_penalty": FLOOR_PENALTY}

