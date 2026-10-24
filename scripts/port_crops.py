"""Port workbook crop params (data/crop_params_swat.csv) into plants.plt for corn/barl/alfa.

COLMAP keys are 0-based whitespace token indices. Unmapped columns keep SWAT+ defaults.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from swat_gym.params import set_value

ROOT = Path(__file__).resolve().parents[1]
TIO = ROOT / "model" / "rotation" / "TxtInOut"
CROP_CSV = ROOT / "data" / "crop_params_swat.csv"

# SWAT crop code (CPNM in the site table) -> SWAT+ plants.plt name
CROP_MAP = {"CSIL": "corn", "BARL": "barl", "ALFA": "alfa"}

# SWAT+ plants.plt token index -> RuFaS field (with optional transform)
COLMAP = {
    5:  ("BIO_E", 1.0),          # bm_e        radiation-use efficiency
    6:  ("HVSTI", 1.0),          # harv_idx
    7:  ("BLAI", 1.0),           # lai_pot
    8:  ("FRGRW1", 1.0),         # frac_hu1
    9:  ("LAIMX1", 1.0),         # lai_max1
    10: ("FRGRW2", 1.0),         # frac_hu2
    11: ("LAIMX2", 1.0),         # lai_max2
    12: ("DLAI", 1.0),           # hu_lai_decl
    15: ("RDMX", 1.0),           # rt_dp_max (the site table is already in m)
    16: ("T_OPT", 1.0),          # tmp_opt
    17: ("T_BASE", 1.0),         # tmp_base
    18: ("CNYLD", 1.0),          # frac_n_yld
    19: ("CPYLD", 1.0),          # frac_p_yld
    20: ("BN1", 1.0),            # frac_n_em
    21: ("BN2", 1.0),            # frac_n_50
    22: ("BN3", 1.0),            # frac_n_mat
    23: ("BP1", 1.0),            # frac_p_em
    24: ("BP2", 1.0),            # frac_p_50
    25: ("BP3", 1.0),            # frac_p_mat
    26: ("WSYF", 1.0),           # harv_idx_ws
    14: ("CHTMX", 1.0),          # can_ht_max
    27: ("USLE_C", 1.0),         # usle_c_min
    28: ("GSI", 1.0),            # stcon_max   maximum stomatal conductance
    29: ("VPDFR", 1.0),          # vpd
    30: ("FRGMAX", 1.0),         # frac_stcon
    31: ("WAVP", 1.0),           # ru_vpd
    32: ("CO2HI", 1.0),          # co2_hi
    33: ("BIOEHI", 1.0),         # bm_e_hi
    34: ("RSDCO_PL", 1.0),       # plnt_decomp
    39: ("EXT_COEF", 1.0),       # ext_co      radiation extinction coefficient
    42: ("BM_DIEOFF", 1.0),      # bm_dieoff
}

#: Unfilled (all-zero) workbook columns; not ported.
UNFILLED = ("ALAI_MIN", "BIO_LEAF", "MAT_YRS", "BMX_TREES")

#: Spring barley must be warm_annual (cold_annual is for fall-sown crops and stunts it).
PLNT_TYP = {"barl": "warm_annual"}

#: (row, column) -> (value, justification), applied after COLMAP.
OVERRIDES: dict[tuple[str, str], tuple[float, str]] = {
    # Inert under rev 62; owned here only (not optimize.PARAMS). Asserted by check_param_state.
    ("alfa", "lai_min"): (1.75, "sourced; inert under rev 62 — dropped from PARAMS 2026-09-10"),
}


def main() -> None:
    table = pd.read_csv(CROP_CSV).set_index("CPNM")
    cfgs = {code: table.loc[code].to_dict() for code in CROP_MAP}
    lines = (TIO / "plants.plt").read_text().splitlines(keepends=True)

    for rufas_name, swat_name in CROP_MAP.items():
        cfg = cfgs[rufas_name]
        # find the plants.plt row whose first token == swat_name
        for i, ln in enumerate(lines):
            toks = ln.split()
            if toks[:1] == [swat_name]:
                for idx, (field, scale) in COLMAP.items():
                    toks[idx] = f"{cfg[field] * scale:.5f}"
                if swat_name in PLNT_TYP:
                    toks[1] = PLNT_TYP[swat_name]
                # rebuild: name left-justified, rest right-justified 14-wide, desc trailing
                desc = toks[-1] if not _is_float(toks[-1]) else ""
                nums = toks[1:-1] if desc else toks[1:]
                lines[i] = f"{swat_name:<22}" + "".join(f"{t:>14}" for t in nums) + (f"  {desc}" if desc else "") + "\n"
                print(f"{swat_name} <- {rufas_name} ({cfg['CROPNAME']}): "
                      f"bm_e={cfg['BIO_E']} lai_pot={cfg['BLAI']} "
                      f"harv_idx={cfg['HVSTI']} tmp_opt={cfg['T_OPT']} tmp_base={cfg['T_BASE']}")
                break
        else:
            print(f"WARNING: {swat_name} not found in plants.plt")

    (TIO / "plants.plt").write_text("".join(lines))

    # By column name: SWAT+-only columns not in COLMAP.
    text = (TIO / "plants.plt").read_text()
    for (row, column), (value, why) in OVERRIDES.items():
        text = set_value(text, "plants.plt", row, column, value)
        print(f"  override {row}.{column} = {value}  ({why})")
    (TIO / "plants.plt").write_text(text)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
