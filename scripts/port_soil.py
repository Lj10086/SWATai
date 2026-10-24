"""Port the reference Portneuf profile into soils.sol (block '80295', in place) and set the HRU to 1 ha."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIO = ROOT / "model" / "rotation" / "TxtInOut"
SOIL_CSV = ROOT / "data" / "soil_gracenet.csv"

SOIL_NAME = "80295"
HYD_GRP = "C"
ANION_EXCL, PERC_CRK = 0.5, 0.5

# Reference profile, one tuple per layer, in soils.sol column order:
#   dp (mm), bd, awc, soil_k (mm/hr), carbon (%), clay, silt, sand, rock (%),
#   alb, usle_k, ec, caco3, ph
LAYERS = [
    (150.0,  1.30, 0.20, 32.40, 1.50, 17.0, 69.3, 13.7, 0.0, 0.30, 0.43, 0.0,  8.0, 7.9),
    (310.0,  1.40, 0.18, 10.15, 0.75,  9.5, 69.3, 21.2, 0.0, 0.30, 0.64, 5.0, 23.0, 8.4),
    (610.0,  1.50, 0.18, 10.15, 0.75,  9.5, 69.3, 21.2, 0.0, 0.30, 0.64, 5.0,  0.0, 0.0),
    (1220.0, 1.40, 0.18, 10.15, 0.75,  9.5, 69.3, 21.2, 0.0, 0.30, 0.64, 5.0,  0.0, 0.0),
]

HRU_AREA_HA = 1.0
LAT, LON = 42.54, -114.32

BD_COL = 1   # index of bd within a LAYERS tuple


def measured_bd() -> list[float]:
    """Per-layer bulk density from measured GRACEnet plot means, thickness-weighted."""
    df = pd.read_csv(SOIL_CSV).dropna(subset=["bd_g_cm3"]).sort_values("depth_mm")
    tops = [0.0] + [float(d) for d in df["depth_mm"][:-1]]
    df = df.assign(top=tops, thick=lambda d: d["depth_mm"] - d["top"])

    out, lyr_top = [], 0.0
    for lyr in LAYERS:
        lyr_bot = lyr[0]
        span = df[(df["depth_mm"] > lyr_top) & (df["top"] < lyr_bot)]
        out.append(float((span["bd_g_cm3"] * span["thick"]).sum() / span["thick"].sum()))
        lyr_top = lyr_bot
    return out


def main() -> None:
    for lyr_i, bd in enumerate(measured_bd()):
        old = LAYERS[lyr_i][BD_COL]
        LAYERS[lyr_i] = LAYERS[lyr_i][:BD_COL] + (round(bd, 4),) + LAYERS[lyr_i][BD_COL + 1:]
        print(f"soils.sol layer {lyr_i + 1} (to {LAYERS[lyr_i][0]:.0f} mm): "
              f"bd {old} -> {round(bd, 4)} (GRACEnet measured)")

    sol = (TIO / "soils.sol").read_text().splitlines(keepends=True)
    # locate the 80295 header line and capture its indent style from the following layer line
    hdr_i = next(i for i, ln in enumerate(sol) if ln.split()[:1] == [SOIL_NAME])
    old_nly = int(sol[hdr_i].split()[1])
    layer_line = sol[hdr_i + 1]
    prefix = layer_line[: len(layer_line) - len(layer_line.lstrip(" "))]  # leading spaces

    # rebuild header: name nly hyd_grp dp_tot anion_excl perc_crk texture
    dp_tot = LAYERS[-1][0]
    texture = "-_".join(["SIL"] * len(LAYERS))
    new_hdr = (
        f"{SOIL_NAME:<30}{len(LAYERS):>6}{HYD_GRP:>18}{dp_tot:>14.5f}"
        f"{ANION_EXCL:>14.5f}{PERC_CRK:>14.5f}  {texture}\n"
    )
    new_block = [new_hdr] + [
        prefix + "".join(f"{v:>14.5f}" for v in lyr) + "  \n" for lyr in LAYERS
    ]
    sol[hdr_i : hdr_i + 1 + old_nly] = new_block
    (TIO / "soils.sol").write_text("".join(sol))
    print(f"soils.sol: {SOIL_NAME} -> reference Portneuf, {len(LAYERS)} layers to "
          f"{dp_tot:.0f} mm, hyd_grp {HYD_GRP} (was {old_nly} layers)")

    # set HRU area to 1 ha and coordinates to the Kimberly site
    hru = (TIO / "hru.con").read_text().splitlines(keepends=True)
    parts = hru[2].split()
    parts[3] = f"{HRU_AREA_HA:.5f}"   # area (ha)
    parts[4] = f"{LAT:.5f}"
    parts[5] = f"{LON:.5f}"
    # reassemble with generous spacing (SWAT+ reads free-format)
    hru[2] = "  ".join(parts) + "  \n"
    (TIO / "hru.con").write_text("".join(hru))
    print(f"hru.con: area -> {HRU_AREA_HA} ha, lat/lon -> {LAT} / {LON}")

    resize_containing_objects()


#: (file, first data line, area columns) to resize to 1 ha.
CONTAINERS = [
    ("object.cnt", 2, (1, 2)),
    ("aquifer.con", 2, (3,)),
    ("rout_unit.con", 2, (3,)),
    ("ls_unit.def", 3, (2,)),
    ("chandeg.con", 2, (3,)),
]


def resize_containing_objects() -> None:
    """Resize containing objects to 1 ha (else basin_aqu tables are diluted ~1432x)."""
    for name, line_i, cols in CONTAINERS:
        path = TIO / name
        lines = path.read_text().splitlines(keepends=True)
        changed = []
        for i in range(line_i, len(lines)):
            parts = lines[i].split()
            if len(parts) <= max(cols):
                continue
            changed.append(parts[1] if len(cols) == 1 else parts[0])
            for col in cols:
                parts[col] = f"{HRU_AREA_HA:.5f}"
            lines[i] = "  ".join(parts) + "  \n"
        path.write_text("".join(lines))
        print(f"{name}: area -> {HRU_AREA_HA} ha for {', '.join(changed)}")


if __name__ == "__main__":
    main()
