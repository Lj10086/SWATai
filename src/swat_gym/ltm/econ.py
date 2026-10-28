"""Prices and crop operating costs for the LTM environment.

    profit = revenue - water - manure - mineral N - mineral P - passes - crop operating costs

The first five terms are the levers' inputs, priced as in ``swat_gym.rewarders``. The last is
new with the rotation lever: each crop-season carries its budget's operating costs *other
than* those inputs (seed, pesticide, custom work, labour, machinery, insurance, the per-acre
water assessment). Without it, a pure-profit rotation would pick the highest-revenue crop.
Land rent and overhead are the same for every crop and are left out (they cancel in every
comparison here, and are not part of the reported profit level).

Sources (University of Idaho Extension, Southcentral Idaho / Magic Valley costs and returns
estimates; the 2019 set, potato 2017, from the UI AgBiz crop budget page):
  EBB3-MB-19 spring malting barley, EBB3-HRS-19 hard red spring wheat, EBB3-SU-19 sugarbeet,
  EBB3-AH-19 alfalfa hay, EBB3-AE2-19 alfalfa establishment (seed line), EBB3-CS-19 corn
  silage, southcentral-irrigated-crop-budgets-2017.xlsx sheet Potato2 (Russet Burbank,
  on-farm storage, no fumigation).

Each figure below is TOTAL OPERATING COSTS/ACRE minus Fertilizer, Irrigation Power,
Irrigation Repairs, Custom Fertilize and Interest on Operating Capital (corn silage also minus
its Custom Haul/Apply Manure line), in the budget's own year dollars ($/acre).
Escalated to the price years (user decision 2026-10-25) by USDA NASS's index of prices paid by
farmers (all farms, United States, 2011=100, monthly; www.nass.usda.gov/Charts_and_Maps/
graphics/data/allprpd.txt, retrieved 2026-10-25), calendar-year means: cost x
mean(PPI 2022-2024) / PPI(budget year).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..rewarders import Prices, _grain_to_mg_dm, average

ACRE_HA = 0.404686
KG_PER_CWT = 45.359237

#: $/acre, budget-year dollars (see module docstring for the line items removed)
OPERATING_USD_ACRE = {
    "barl": 379.93 - 53.10 - 42.68 - 11.66 - 7.25 - 9.29,             # EBB3-MB-19: 255.95
    "wwht": 414.80 - 81.10 - 42.68 - 11.66 - 7.25 - 11.16,            # EBB3-HRS-19: 260.95
    "sgbt": 940.29 - 131.95 - 67.90 - 18.55 - 7.75 - 33.13,           # EBB3-SU-19: 681.01
    "pota": (350.75 + 341.35 + 267.05 + 85.25 + 114.19 + 122.97 + 181.59 + 137.72 + 51.73)
            - 341.35 - 52.38 - 14.31 - (8.00 + 7.25) - 51.73,          # Potato2 2017: 1177.58
    "alfa": 596.87 - 69.95 - 69.84 - 19.08 - 8.75 - 12.34,            # EBB3-AH-19: 416.91
    "corn": 819.12 - 59.90 - 58.20 - 15.90 - 7.75 - 18.56 - 36.00,    # EBB3-CS-19: 622.81
}
#: Alfalfa establishment year: the production-year costs plus the seed (EBB3-AE2-19 "Alfalfa
#: Seed (pvt.): Inoculated", 20 lb x $4.25). The trial's spring seeding is cut in year 1
#: (Ai-SWAT-Plus spec), unlike AE2's fall seeding with no harvest.
ALFA_SEED_USD_ACRE = 85.00
COST_YEAR = {"barl": 2019, "wwht": 2019, "sgbt": 2019, "pota": 2017, "alfa": 2019, "corn": 2019}
#: NASS prices-paid index, calendar-year mean of the monthly values (12 months each)
PRICES_PAID = {2017: 106.93, 2019: 110.26, 2022: 137.43, 2023: 140.65, 2024: 139.87}


def escalation(cost_year: int) -> float:
    """Budget-year dollars -> mean 2022-2024 dollars."""
    return sum(PRICES_PAID[y] for y in YEARS) / len(YEARS) / PRICES_PAID[cost_year]

# Idaho marketing-year average prices, USDA NASS Crop Values: 2022-2023 from the 2024 Summary
# (Feb 2025), 2024 final from the 2025 Summary (Feb 2026), as rewarders._NASS.
#: Other spring wheat, $/bu (the trial grew hard red spring wheat)
SPRING_WHEAT_USD_BU = {2022: 8.20, 2023: 7.03, 2024: 5.70}
#: All potatoes, $/cwt fresh
POTATO_USD_CWT = {2022: 12.50, 2023: 9.63, 2024: 9.02}
#: Russet potato tuber dry-matter fraction (fresh -> DM for pricing): 1 - water, water 78.6 g per
#: 100 g in USDA-ARS FoodData Central SR Legacy FDC ID 170027 "Potatoes, russet, flesh and skin,
#: raw" (published 2019-04-01; the Foundation record FDC 2346401, russet without skin, raw,
#: 2022, agrees at 78.6 g). Market russets, cultivar unspecified; the trial reported no tuber DM.
POTA_DM = 1.0 - 0.786
POTA_DM_PROVISIONAL = POTA_DM   # old name, kept for callers

YEARS = (2022, 2023, 2024)


def _mean(d: dict) -> float:
    return sum(d[y] for y in YEARS) / len(YEARS)


@dataclass(frozen=True)
class LtmPrices:
    base: Prices
    operating: dict = field(default_factory=dict)   #: $/ha per crop-season
    alfa_seed: float = 0.0                           #: $/ha, establishment year only
    fert_p: float = 0.0                              #: $/kg P (as base.fert_p)
    pota_dm: float = POTA_DM

    def key(self) -> dict:
        return {**self.base.key(), "operating": dict(sorted(self.operating.items())),
                "alfa_seed": self.alfa_seed, "pota_dm": self.pota_dm}


def prices(*, pota_dm: float = POTA_DM) -> LtmPrices:
    """2022-2024 average prices with the LTM crops: wheat priced as spring wheat, potato added."""
    base = average()
    crop = dict(base.crop)
    crop["wwht"] = _grain_to_mg_dm(_mean(SPRING_WHEAT_USD_BU), lb_per_bu=60.0, moisture=0.135)
    crop["pota"] = _mean(POTATO_USD_CWT) / (KG_PER_CWT * pota_dm) * 1000.0
    base = replace(base, crop=crop, label=f"ltm_{base.label}")
    return LtmPrices(base=base,
                     operating={c: v * escalation(COST_YEAR[c]) / ACRE_HA
                                for c, v in OPERATING_USD_ACRE.items()},
                     alfa_seed=ALFA_SEED_USD_ACRE * escalation(COST_YEAR["alfa"]) / ACRE_HA,
                     fert_p=base.fert_p,
                     pota_dm=pota_dm)
