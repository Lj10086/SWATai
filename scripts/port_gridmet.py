#!/usr/bin/env python3
"""Port gridMET daily weather for the Kimberly field into SWAT+ weather files.

gridMET (Abatzoglou 2013, Int. J. Climatol. 33:121-131, doi:10.1002/joc.3413) at the field
point (hru.con: 42.540 N, 114.320 W), 1995-2025, from the Northwest Knowledge Network THREDDS
NCSS point service. That service returns the *packed* integers; each variable is unpacked
with the scale_factor / add_offset its OPeNDAP .das publishes (checked below, and recorded):
tmmx and tmmn have different offsets (220 K vs 210 K).

===== ===================================== ==========================================
var   gridMET quantity                       SWAT+ series
===== ===================================== ==========================================
pr    precipitation (mm)                     pcp, mm
tmmx  max air temperature (K)                tmp, C
tmmn  min air temperature (K)                tmp, C
srad  mean shortwave radiation (W m-2)       slr, MJ m-2 d-1 (x 0.0864)
rmax  max relative humidity (%)              hmd: RH fraction, FAO-56 convention (below)
rmin  min relative humidity (%)              hmd
vs    mean wind speed at 10 m (m s-1)        wnd, m/s at 2 m: FAO-56 eq. 47 (x 0.748)
pet   grass reference ET, ASCE-PM (mm)       not an input; data/observed_et_gridmet.csv
===== ===================================== ==========================================

RH: actual vapour pressure from FAO-56 eq. 17, e_a = (e_s(Tmin) RHmax + e_s(Tmax) RHmin) / 2,
then RH = e_a / ((e_s(Tmax) + e_s(Tmin)) / 2), the same convention port_agrimet.py used for
the station record (Allen et al. 1998, FAO Irrigation and Drainage Paper 56, ch. 3).

    uv run python scripts/port_gridmet.py                 # fetch (cached) + write the rotation
    uv run python scripts/port_gridmet.py --crops         # also every model/crops/*/TxtInOut
    uv run python scripts/port_gridmet.py --refresh       # re-download
    uv run python scripts/port_gridmet.py --site ltm      # the long-term manure field -> model/ltm

The LTM field (Leytem et al. 2025: 42 33' 4.5" N, 114 21' 14.7" W) is ~3 km west of the
GRACEnet field and in the adjacent gridMET cell, so it has its own cache.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROTATION = ROOT / "model" / "rotation" / "TxtInOut"
CROPS = ROOT / "model" / "crops"
CACHE = ROOT / "data" / "gridmet_kimberly_1995_2025.csv"
META = ROOT / "data" / "gridmet_kimberly_1995_2025.json"
ET_CSV = ROOT / "data" / "observed_et_gridmet.csv"

LAT, LON = 42.540, -114.320   # the field (hru.con), not the AgriMet station
#: site: (lat, lon, cache stem, target TxtInOut)
SITES = {
    "gracenet": (42.540, -114.320, "gridmet_kimberly_1995_2025", ROTATION),
    "ltm": (42.5513, -114.3541, "gridmet_ltm_1995_2025", ROOT / "model" / "ltm" / "TxtInOut"),
}


def use_site(name: str) -> Path:
    """Point LAT/LON/CACHE/META at a site; returns its TxtInOut."""
    global LAT, LON, CACHE, META
    LAT, LON, stem, tio = SITES[name]
    CACHE, META = ROOT / "data" / f"{stem}.csv", ROOT / "data" / f"{stem}.json"
    return tio
YEARS = (1995, 2025)
THREDDS = "http://thredds.northwestknowledge.net:8080/thredds"
VARS = {  # short name: netCDF variable
    "pr": "precipitation_amount", "tmmx": "daily_maximum_temperature",
    "tmmn": "daily_minimum_temperature", "srad": "daily_mean_shortwave_radiation_at_surface",
    "rmax": "daily_maximum_relative_humidity", "rmin": "daily_minimum_relative_humidity",
    "vs": "daily_mean_wind_speed", "pet": "daily_mean_reference_evapotranspiration_grass",
}
FILL = 32767
#: FAO-56 eq. 47, 10 m -> 2 m: 4.87 / ln(67.8 z - 5.42).
WIND_10_TO_2 = 4.87 / np.log(67.8 * 10.0 - 5.42)
CHUNK = 5  # years per NCSS request (long ranges are slow)

SERIES = {  # kind: (data file, cli file, title)
    "pcp": ("tfccpcp.pcp", "pcp.cli", "Precipitation mm"),
    "tmp": ("tfcctmp.tem", "tmp.cli", "Temperature C"),
    "slr": ("tfccslr.slr", "slr.cli", "Solar radiation MJ/m2"),
    "hmd": ("tfcchmd.hmd", "hmd.cli", "Relative humidity"),
    "wnd": ("tfccwnd.wnd", "wnd.cli", "Wind speed m/s at 2 m"),
}


def _url(path: str) -> str:
    return f"{THREDDS}/{path}"


def _get(url: str, retries: int = 4) -> str:
    last = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read().decode()
        except Exception as e:  # network: retry with backoff, then raise
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"gridMET request failed after {retries} tries: {url}\n{last}")


def packing(var: str) -> dict:
    """scale_factor / add_offset / fill from the variable's OPeNDAP attributes."""
    das = _get(_url(f"dodsC/agg_met_{var}_1979_CurrentYear_CONUS.nc.das"))
    block = das[das.index(VARS[var]):]
    block = block[:block.index("}")]
    num = lambda key: float(re.search(rf"{key}\s+([-\d.Ee+]+)", block).group(1))  # noqa: E731
    return {"scale_factor": num("scale_factor"), "add_offset": num("add_offset"),
            "fill": int(num("_FillValue"))}


def fetch(var: str, y0: int, y1: int) -> pd.Series:
    url = _url(f"ncss/agg_met_{var}_1979_CurrentYear_CONUS.nc?var={VARS[var]}"
               f"&latitude={LAT}&longitude={LON}&time_start={y0}-01-01&time_end={y1}-12-31"
               "&accept=csv")
    rows = [line.split(",") for line in _get(url).splitlines()[1:] if line]
    return pd.Series({r[0][:10]: float(r[3]) for r in rows}, name=var)


def _parse(text: str, var: str) -> pd.Series:
    rows = [line.split(",") for line in text.splitlines()[1:] if line]
    return pd.Series({r[0][:10]: float(r[3]) for r in rows}, name=var)


def download(chunk_dir: Path | None = None) -> pd.DataFrame:
    """Fetch (or, with ``chunk_dir``, assemble pre-fetched ``<var>_<year>.csv`` NCSS responses,
    e.g. from a resumable shell loop when the service is flaky) and cache the unpacked record."""
    meta = {"source": "gridMET (Abatzoglou 2013, doi:10.1002/joc.3413)",
            "service": f"{THREDDS}/ncss/agg_met_<var>_1979_CurrentYear_CONUS.nc",
            "lat": LAT, "lon": LON, "years": YEARS, "retrieved": date.today().isoformat(),
            "packing": {}}
    chunks = [(y, min(y + CHUNK - 1, YEARS[1])) for y in range(YEARS[0], YEARS[1] + 1, CHUNK)]
    cols = {}
    with ThreadPoolExecutor(8) as pool:
        for var in VARS:
            pk = packing(var)
            meta["packing"][var] = pk
            if chunk_dir is None:
                parts = list(pool.map(lambda c: fetch(var, *c), chunks))
            else:
                parts = [_parse((Path(chunk_dir) / f"{var}_{y}.csv").read_text(), var)
                         for y in range(YEARS[0], YEARS[1] + 1)]
            raw = pd.concat(parts)
            if (raw == pk["fill"]).any():
                raise SystemExit(f"{var}: {int((raw == pk['fill']).sum())} fill values")
            cols[var] = raw * pk["scale_factor"] + pk["add_offset"]
            print(f"  {var}: {len(raw)} days, scale {pk['scale_factor']} offset {pk['add_offset']}")
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    expected = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[1]}-12-31", freq="D")
    if not df.index.equals(expected):
        raise SystemExit(f"gridMET returned {len(df)} days, expected {len(expected)}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.rename_axis("date").to_csv(CACHE, float_format="%.4f")
    META.write_text(json.dumps(meta, indent=2) + "\n")
    return df


def _es(t_c):
    """Saturation vapour pressure (kPa), FAO-56 eq. 11."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def convert(df: pd.DataFrame) -> pd.DataFrame:
    tmax, tmin = df["tmmx"] - 273.15, df["tmmn"] - 273.15
    if (tmin > tmax + 1e-9).any():
        raise SystemExit(f"{int((tmin > tmax).sum())} days with tmin > tmax: packing is wrong")
    e_a = (_es(tmin) * df["rmax"] + _es(tmax) * df["rmin"]) / 200.0
    rh = (e_a / ((_es(tmax) + _es(tmin)) / 2.0)).clip(0.05, 1.0)
    return pd.DataFrame({"year": df.index.year, "jday": df.index.dayofyear,
                         "pcp": df["pr"], "tmax": tmax, "tmin": tmin,
                         "slr": df["srad"] * 0.0864, "hmd": rh,
                         "wnd": df["vs"] * WIND_10_TO_2, "pet": df["pet"]})


def write(tio: Path, w: pd.DataFrame, elev: float) -> None:
    nbyr = w["year"].nunique()
    head = (f"nbyr     tstep       lat       lon      elev\n"
            f"{nbyr:>4}{0:>10}{LAT:>10.3f}{LON:>10.3f}{elev:>10.3f}\n")
    src = f"gridMET (Abatzoglou 2013) at {LAT:.3f}N {abs(LON):.3f}W, ported by swat-kimberly"
    for kind, (data, cli, title) in SERIES.items():
        with open(tio / data, "w") as f:
            f.write(f"{title} - {src}\n{head}")
            if kind == "tmp":
                for r in w.itertuples():
                    f.write(f"{r.year}{r.jday:>7}{r.tmax:>13.5f}{r.tmin:>13.5f}\n")
            else:
                for r in w.itertuples():
                    f.write(f"{r.year}{r.jday:>7}{getattr(r, kind):>13.5f}\n")
        (tio / cli).write_text(f"{cli}: written by swat-kimberly from gridMET\nfilename\n{data}\n")
        sta = (tio / "weather-sta.cli").read_text()
        if data not in sta:
            raise SystemExit(f"{tio}/weather-sta.cli does not name {data}; wire it first")
    shown = tio.relative_to(ROOT) if tio.is_relative_to(ROOT) else tio
    print(f"  wrote {shown} ({len(w)} days)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", action="store_true", help="also every model/crops/*/TxtInOut")
    ap.add_argument("--only-crops", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    ap.add_argument("--site", choices=SITES, default="gracenet")
    ap.add_argument("--from-chunks", type=Path, default=None,
                    help="assemble <var>_<year>.csv NCSS responses from this directory")
    args = ap.parse_args()
    site_tio = use_site(args.site)

    if args.from_chunks is not None:
        df = download(args.from_chunks)
    elif CACHE.is_file() and not args.refresh:
        df = pd.read_csv(CACHE, index_col="date", parse_dates=True)
    else:
        print("downloading gridMET ...")
        df = download()
    w = convert(df)
    if args.site == "gracenet":
        w[["year", "jday", "pet"]].rename(columns={"pet": "eto_mm"}).to_csv(ET_CSV, index=False)

    targets = [] if args.only_crops else [site_tio]
    if args.crops or args.only_crops:
        targets += sorted(p / "TxtInOut" for p in CROPS.iterdir() if (p / "TxtInOut").is_dir())
    for tio in targets:
        elev = float((tio / "hru.con").read_text().splitlines()[2].split()[6])
        write(tio, w, elev)
    annual = w.groupby("year")[["pcp", "pet"]].sum()
    print(f"annual precip {annual['pcp'].mean():.0f} mm, grass ETo {annual['pet'].mean():.0f} mm "
          f"(1995-2025 mean)")


if __name__ == "__main__":
    main()
