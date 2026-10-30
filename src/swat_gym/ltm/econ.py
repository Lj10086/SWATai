"""Prices and crop operating costs for the LTM environment.

    profit = revenue - water - manure - mineral N - mineral P - passes - crop operating costs

The first five terms are the levers' inputs (``Prices``, ``average``). The last is
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

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

#: short ton -> Mg
TON = 0.907185
#: 1 mm of water over 1 ha = 10 m**3
M3_PER_MM_HA = 10.0


def _hay_to_mg_dm(usd_per_ton: float, moisture: float = 0.12) -> float:
    """Hay $/short ton at market moisture -> $/Mg DM."""
    return usd_per_ton / (TON * (1.0 - moisture))


def _grain_to_mg_dm(usd_per_bu: float, lb_per_bu: float, moisture: float) -> float:
    """Grain $/bushel at market moisture -> $/Mg DM."""
    kg_dm = lb_per_bu * 0.453592 * (1.0 - moisture)
    return usd_per_bu / kg_dm * 1000.0


def _silage_from_grain(usd_per_bu: float, multiplier: float = 9.0, dm: float = 0.35) -> float:
    """Corn silage $/Mg DM derived from grain price (no NASS silage series)."""
    return usd_per_bu * multiplier / (TON * dm)


@dataclass(frozen=True)
class Prices:
    """Input and crop price vector. ``crop`` keyed by SWAT+ plant name."""

    crop: Mapping[str, float]   #: $/Mg dry matter
    water: float                #: $ per mm per ha
    manure: float               #: $ per Mg as applied (haul + spread)
    label: str = "unnamed"
    fert_n: float = 0.0         #: $ per kg mineral N
    fert_op: float = 0.0        #: $ per ha per application pass
    fert_p: float = 0.0         #: $ per kg P

    def key(self) -> dict:
        """Every value, for stage configs: a label alone hides a corrected price."""
        return {"label": self.label, "crop": dict(sorted(self.crop.items())),
                "water": self.water, "manure": self.manure, "fert_n": self.fert_n,
                "fert_op": self.fert_op, "fert_p": self.fert_p}


# Idaho marketing-year averages, USDA NASS Crop Values (2022-23: Feb 2025 summary; 2024:
# final values from the Feb 2026 summary, which revised the preliminary 154.00 / 6.70 / 5.80).
# Barley priced as grain.
_NASS = {  # year: (alfalfa hay $/ton, barley $/bu, corn grain $/bu)
    2022: (272.00, 7.61, 7.37),
    2023: (204.00, 7.76, 5.53),
    2024: (153.00, 6.44, 5.78),
}
# Same source; 2024 final (winter wheat prelim. was 5.55).
_NASS_CROPS = {  # year: (winter wheat $/bu, sugarbeet $/ton as delivered)
    2022: (7.97, 67.00),
    2023: (6.77, 74.40),
    2024: (5.48, 53.70),
}
#: Sugarbeet root dry-matter fraction: SWAT+ yields are dry, NASS prices are per delivered
#: (fresh) ton. Measured at Kimberly: root moisture 74.8 % in the synthetic-fertilizer plots
#: (Leytem et al. 2025, Crop Sci., doi:10.1002/csc2.70144, Table 5; 17.7 Mg DM of 70.7 Mg
#: fresh), matching the crop model's synthetic N. Manure plots run 76.6-77.8 % (DM 22-23 %).
SGBT_DM = 1.0 - 0.748

# Water $/mm/ha = WD01 rental (opportunity cost) + pumping energy.

M3_PER_ACRE_FOOT = 1233.48

#: WD01 2025 Rental Pool Procedures 5.3.101/5.3.103: common-pool Tier 1 (above Milner)
#: $25.18 + 10 % Board surcharge + $1.30 administrative fee, $/af.
WD01_RENTAL_AF = 25.18 * 1.10 + 1.30          # = 29.00
#: Same document 10.7.103/10.7.107(c): assignment-pool Tier 7, the top of the published range,
#: all-in like Tier 1 ($55.00 + $5.50 + $1.30). Was the bare $55.00 before 2026-10-24.
WD01_RENTAL_AF_TIER7 = 55.00 * 1.10 + 1.30    # = 61.80

WATER_RENTAL = WD01_RENTAL_AF / M3_PER_ACRE_FOOT * M3_PER_MM_HA     # 0.235 $/mm/ha

#: Idaho Power Schedule 24 tariff.
IPC_SCHED24_ENERGY_PER_KWH = 0.067222
IPC_SCHED24_DEMAND_PER_KW_MONTH = 16.50
#: Assumed, no published site value: total dynamic head of a canal-fed sprinkler (no well
#: lift), ~64 psi; demand-charge load factor. Both enter only the pumping term; breakevens.
PUMP_HEAD_M = 45.0
PUMP_LOAD_FACTOR = 0.60
#: Nebraska Pumping Plant Performance Criteria, electric: 75 % pump x 88 % motor (0.885 water
#: hp-h per kWh; e.g. NC State Extension, "Pumping Plant Performance").
PUMP_EFFICIENCY = 0.75 * 0.88
HOURS_PER_MONTH = 730.0

PUMP_KWH_PER_MM_HA = 9810.0 * M3_PER_MM_HA * PUMP_HEAD_M / (3.6e6 * PUMP_EFFICIENCY)  # 1.89
WATER_PUMPING = PUMP_KWH_PER_MM_HA * (
    IPC_SCHED24_ENERGY_PER_KWH
    + IPC_SCHED24_DEMAND_PER_KW_MONTH / (HOURS_PER_MONTH * PUMP_LOAD_FACTOR))  # 0.198 $/mm/ha

DEFAULT_WATER = WATER_RENTAL + WATER_PUMPING                                  # 0.433

#: Manure haul-and-spread $/Mg ($7.20/ton custom rate). Measured practice is manure-only,
#: so this lowers the human bar; report breakeven if a result depends on it.
DEFAULT_MANURE = 7.20 / 0.907185

#: Urea $644/short ton, 46 % N -> $/kg N. USDA AMS Inter-Mountain West, w/e 2025-04-04.
DEFAULT_FERT_N = 644.0 / (2000.0 * 0.46 * 0.453592)

#: MAP (11-52-0) $934.11/short ton, same report and week. The 11 % N is credited at the
#: urea price and the rest charged to P (P2O5 is 43.64 % P) -> $/kg P.
_MAP_KG = 2000.0 * 0.453592
DEFAULT_FERT_P = ((934.11 - 0.11 * _MAP_KG * DEFAULT_FERT_N)
                  / (0.52 * 0.4364 * _MAP_KG))

#: Per-pass application cost, $/ha ($15/acre). Without it, splitting N is free.
DEFAULT_FERT_OP = 15.00 / 0.404686


def nass(year: int, *, water: float = DEFAULT_WATER, manure: float = DEFAULT_MANURE,
         fert_n: float = DEFAULT_FERT_N, fert_op: float = DEFAULT_FERT_OP,
         fert_p: float = DEFAULT_FERT_P, silage_multiplier: float = 9.0) -> Prices:
    """Price vector for one NASS marketing year, in $/Mg DM."""
    hay, barl_bu, corn_bu = _NASS[year]
    wheat_bu, beet_ton = _NASS_CROPS[year]
    return Prices(
        crop={
            "alfa": _hay_to_mg_dm(hay),
            # Base moisture and bushel weight: Sadaka et al., Univ. Arkansas FSA1078 (2021),
            # barley 48 lb/bu at 14.5 %, wheat 60 lb/bu at 13.5 % (Tables 1-2).
            "barl": _grain_to_mg_dm(barl_bu, lb_per_bu=48.0, moisture=0.145),
            "corn": _silage_from_grain(corn_bu, multiplier=silage_multiplier),
            "wwht": _grain_to_mg_dm(wheat_bu, lb_per_bu=60.0, moisture=0.135),
            "sgbt": beet_ton / (TON * SGBT_DM),
        },
        water=water, manure=manure, fert_n=fert_n, fert_op=fert_op, fert_p=fert_p,
        label=f"NASS{year}",
    )


AVERAGE_YEARS = (2022, 2023, 2024)


def average(years: tuple[int, ...] = AVERAGE_YEARS, **kw) -> Prices:
    """Mean of converted per-year vectors; the fixed prices experiments run at."""
    vecs = [nass(y, **kw) for y in years]
    crop = {k: sum(v.crop[k] for v in vecs) / len(vecs) for k in vecs[0].crop}
    return replace(vecs[0], crop=crop, label=f"avg{years[0]}-{years[-1]}")


# -- LTM additions: spring wheat, potato, crop operating costs --------------------------------

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
# (Feb 2025), 2024 final from the 2025 Summary (Feb 2026), as _NASS above.
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
