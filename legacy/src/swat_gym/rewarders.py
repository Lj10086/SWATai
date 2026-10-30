"""Profit from a completed SWAT+ run; prices are an experimental axis.

    profit = revenue - irrigation - manure - mineral N - application passes

NO3 leaching is reported, priced only if no3_price > 0 (swept, never a single point).
N2O is IPCC Tier 1 accounting (ipcc_n2o), reported, never priced.
Crop prices are $/Mg dry matter (SWAT+ yields are dry Mg).
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

import pandas as pd

#: short ton -> Mg
TON = 0.907185
#: 1 mm of water over 1 ha = 10 m**3
M3_PER_MM_HA = 10.0

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
    """Price vector. ``crop`` keyed by SWAT+ plant name."""

    crop: Mapping[str, float]   #: $/Mg dry matter
    water: float                #: $ per mm per ha
    manure: float               #: $ per Mg as applied (haul + spread)
    # Fields after `label` are appended so positional callers don't break.
    label: str = "unnamed"
    fert_n: float = 0.0         #: $ per kg mineral N
    fert_op: float = 0.0        #: $ per ha per application pass
    fert_p: float = 0.0         #: $ per kg P (crop models' auto-P table; the rotation has none)

    def key(self) -> dict:
        """Every value, for stage configs: a label alone hides a corrected price."""
        return {"label": self.label, "crop": dict(sorted(self.crop.items())),
                "water": self.water, "manure": self.manure, "fert_n": self.fert_n,
                "fert_op": self.fert_op, "fert_p": self.fert_p}

    def alfalfa_ratio(self, numeraire: str = "corn") -> float:
        return self.crop["alfa"] / self.crop[numeraire]

    def at_ratio(self, ratio: float, numeraire: str = "corn") -> Prices:
        """Reprice alfalfa to ``ratio`` x the numeraire."""
        crop = dict(self.crop)
        crop["alfa"] = ratio * crop[numeraire]
        return replace(self, crop=crop, label=f"{self.label}@ratio={ratio:g}")


# Idaho marketing-year averages, USDA NASS Crop Values (2022-23: Feb 2025 summary; 2024:
# final values from the Feb 2026 summary, which revised the preliminary 154.00 / 6.70 / 5.80).
# Barley priced as grain.
_NASS = {  # year: (alfalfa hay $/ton, barley $/bu, corn grain $/bu)
    2022: (272.00, 7.61, 7.37),
    2023: (204.00, 7.76, 5.53),
    2024: (153.00, 6.44, 5.78),
}
# Crop models only (swat_gym.crop). Same source; 2024 final (winter wheat prelim. was 5.55).
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

#: Water price sweep range: rental only .. Tier 7 + pumping.
WATER_PRICE_RANGE = (WATER_RENTAL,
                     WD01_RENTAL_AF_TIER7 / M3_PER_ACRE_FOOT * M3_PER_MM_HA + WATER_PUMPING)

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


_FERT = re.compile(r"^\s*fert\s+\d+\s+\d+\s+\S+\s+(\S+)\s+\S+\s+([0-9.]+)", re.M)


def fert_applied(sch_text: str, frt_text: str) -> pd.DataFrame:
    """Every ``fert`` op in a schedule: mass, N, and kind (manure vs mineral by product name)."""
    from .schedule import MANURE_SOURCES

    comp = {}
    for line in frt_text.splitlines()[2:]:
        p = line.split()
        if len(p) >= 5:
            comp[p[0]] = float(p[1]) + float(p[3])  # min_n + org_n

    rows = [{"fert": name,
             "kind": "manure" if name in MANURE_SOURCES else "mineral",
             "mg_ha": float(kg) / 1000.0,
             "n_frac": comp.get(name, float("nan")),
             "n_kg_ha": float(kg) * comp.get(name, float("nan"))}
            for name, kg in _FERT.findall(sch_text)]
    return pd.DataFrame(rows, columns=["fert", "kind", "mg_ha", "n_frac", "n_kg_ha"])


def manure_applied(sch_text: str, frt_text: str) -> pd.DataFrame:
    df = fert_applied(sch_text, frt_text)
    return df[df["kind"] == "manure"].reset_index(drop=True)


def profit(runner, prices: Prices, *, manure_mg: float | None = None,
           fert_n_kg: float | None = None, no3_price: float = 0.0) -> dict:
    """Rotation-total profit ($/ha) with separable terms. ``no3_price`` defaults to 0."""
    yields = runner.yields()
    if yields.empty:
        # Even zero water grows ~4 t corn; all-zero means no plan reached the HRU (e.g. the
        # schedule name is not the one landuse.lum wires in). The engine exits 0 either way.
        from .engine import EngineError
        raise EngineError(f"no crop yielded anything in {runner.workdir}: is the schedule "
                          "wired to the HRU (landuse.lum mgt name) and the window simulated?")
    wb = runner.read("hru_wb_yr.txt")
    aqu = runner.read("basin_aqu_yr.txt")

    unpriced = set(yields["plant_name"]) - set(prices.crop)
    if unpriced:
        raise KeyError(f"no price for {sorted(unpriced)}; have {sorted(prices.crop)}")

    revenue = float(sum(r["yld(t)"] * prices.crop[r["plant_name"]]
                        for _, r in yields.iterrows()))
    irrigation_mm = float(wb["irr"].sum())
    # Parse the schedule when a total is unknown or passes are priced (pass count needs it).
    n_events = 0
    manure_n_kg = None
    if manure_mg is None or fert_n_kg is None or prices.fert_op:
        sch = (runner.workdir / "management.sch").read_text()
        frt = (runner.workdir / "fertilizer.frt").read_text()
        applied = fert_applied(sch, frt)
        n_events = int((applied["mg_ha"] > 0).sum())
        # For N2O only; manure is priced by mass.
        manure_n_kg = float(applied.loc[applied["kind"] == "manure", "n_kg_ha"].sum())
        if manure_mg is None:
            manure_mg = float(applied.loc[applied["kind"] == "manure", "mg_ha"].sum())
        if fert_n_kg is None:
            fert_n_kg = float(applied.loc[applied["kind"] == "mineral", "n_kg_ha"].sum())
    no3 = float(aqu["no3_rchg"].sum())

    water_cost = irrigation_mm * prices.water
    manure_cost = manure_mg * prices.manure
    fert_cost = fert_n_kg * prices.fert_n
    op_cost = n_events * prices.fert_op  # every pass, manure or mineral
    leach_cost = no3 * no3_price

    if manure_n_kg is None:
        manure_n_kg = 0.0
    n2o = ipcc_n2o(manure_n_kg=manure_n_kg, mineral_n_kg=float(fert_n_kg),
                   no3_leached_kg=no3)

    return {
        "label": prices.label,
        "revenue": revenue,
        "water_cost": water_cost,
        "manure_cost": manure_cost,
        "fert_cost": fert_cost,
        "op_cost": op_cost,
        "n_fert_events": n_events,
        "profit": revenue - water_cost - manure_cost - fert_cost - op_cost - leach_cost,
        "no3_leached_kg": no3,
        "leach_cost": leach_cost,
        **n2o,  # reported, never priced
        "irrigation_mm": irrigation_mm,
        "manure_mg": manure_mg,
        "fert_n_kg": fert_n_kg,
        "alfalfa_ratio": prices.alfalfa_ratio(),
    }
