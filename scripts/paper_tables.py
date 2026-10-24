#!/usr/bin/env python3
"""Generate markdown + PNG result tables from runs/*.json.

    uv run python scripts/paper_tables.py [--out runs/paper_tables]

Needs runs/exp1_{irrigation,ceiling,controller}.json; PPO rows stay blank until PPO has run.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BASELINE = ROOT / "src" / "swat_gym" / "tests" / "fixtures" / "baseline.json"
REF_CSV = ROOT / "data" / "reference_hru119.csv"

GRACENET = {
    2013: 22.100,
    2014: 5.900,
    2015: 9.000,
    2016: 14.785,
    2017: 16.158,
    2018: 22.935,
    2019: 8.776,
}
CROP_NAME = {
    "corn": "Corn", "barl": "Barley", "alfa": "Alfalfa",
    "CSIL": "Corn", "BARL": "Barley", "ALFA": "Alfalfa",
}
REF_CROP = {"CSIL": "corn", "BARL": "barl", "ALFA": "alfa"}


def _fmt(x: float | None, digits: int = 0) -> str:
    if x is None:
        return "—"
    if digits == 0:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def _paired(p: dict) -> str:
    """mean ± se_ess (rule 7)."""
    return f"{p['mean']:+,.0f} ± {p['se_ess']:,.0f}"


def _ess_note(p: dict) -> str:
    """The sentence that has to travel with every interval in the paper (rule 7)."""
    return (f"± is one standard error over the n = {p['n']} held-out windows, divided by the "
            f"effective sample size {p['ess']:.2f} rather than by n: the windows are eight "
            f"years long and start one year apart, so they are not independent draws.")


def _md(title: str, headers: list[str], rows: list[list[str]], note: str = "") -> str:
    lines = [
        f"**{title}**",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    if note:
        lines += ["", note]
    lines.append("")
    return "\n".join(lines)


def save_table_png(
    path: Path,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    note: str = "",
    col_widths: list[float] | None = None,
) -> None:
    """Render a clean white table PNG for pasting into Google Docs."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    n_cols = len(headers)
    n_rows = len(rows)
    if col_widths is None:
        col_widths = []
        for j in range(n_cols):
            maxlen = len(headers[j])
            for row in rows:
                maxlen = max(maxlen, len(str(row[j])))
            col_widths.append(max(0.9, min(3.2, 0.13 * maxlen + 0.4)))
    fig_w = sum(col_widths) + 0.6
    fig_h = 0.7 + 0.42 * (n_rows + 1) + (0.5 if note else 0.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        0.0, 1.0, title, transform=ax.transAxes, fontsize=11, fontweight="bold",
        va="top", ha="left", color="#111111",
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        bbox=[0.0, 0.10 if note else 0.02, 1.0, 0.75 if note else 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    bold = FontProperties(weight="bold", size=9)
    total_w = sum(col_widths)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.6)
        cell.set_height(0.062 if n_rows < 8 else 0.052)
        cell.set_width(col_widths[c] / total_w)
        if r == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontproperties=bold, color="#111111")
        else:
            cell.set_facecolor("white")
            txt = str(rows[r - 1][c])
            if c == 0 and "This work" in txt:
                cell.set_text_props(fontproperties=bold)
            cell.set_text_props(color="#222222")
            if c > 0:
                raw = (
                    str(rows[r - 1][c])
                    .replace(",", "")
                    .replace("—", "")
                    .replace("*", "")
                    .replace("±", " ")
                    .strip()
                )
                if raw and re.fullmatch(r"[+\-−]?\d+(\.\d+)?(\s+\d+(\.\d+)?)?%?", raw):
                    cell.get_text().set_ha("right")

    if note:
        ax.text(
            0.0, 0.01, note.replace("\\*", "*"), transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left", color="#555555", style="italic",
        )

    fig.savefig(
        path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none",
        pad_inches=0.2,
    )
    plt.close(fig)


def data_env_compare() -> tuple[str, list[str], list[list[str]]]:
    title = "Table A. Crop-management RL environments (selected)."
    headers = [
        "Environment", "Dynamics engine", "Fertilizer / N", "Irrigation",
        "Hydrologic N routing",
    ]
    rows = [
        ["CropGym (Overweg et al., 2021)", "WOFOST / PCSE", "✓", "—", "limited"],
        ["gym-DSSAT (Gautron et al., 2022)", "DSSAT", "✓", "✓", "crop-centric"],
        ["CyclesGym (Turchetta et al., 2022)", "Cycles", "✓", "✓", "limited"],
        [
            "SWATgym (Madondo et al., 2023)",
            "Python reimplementation modeled after classic SWAT",
            "✓", "✓", "as reimplemented",
        ],
        [
            "This work",
            "Official SWAT+ rev 62.0.0 (TxtInOut)",
            "✓", "✓", "native SWAT+",
        ],
    ]
    return title, headers, rows


def data_yield():
    base = json.loads(BASELINE.read_text())
    sim = {int(r["year"]): r for r in base["yields"]}

    import pandas as pd
    ref = pd.read_csv(REF_CSV)
    ref_y = {}
    for _, row in ref.iterrows():
        y = int(row["year"])
        if y not in GRACENET:
            continue
        crop = REF_CROP[str(row["crop"])]
        ref_y[y] = (crop, float(row["yld_t_ha"]))

    year_headers = ["Year", "Crop", "SWAT+", "Reference", "GRACEnet"]
    year_rows: list[list[str]] = []
    by_crop: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_crop_ref: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for year in sorted(GRACENET):
        s = sim[year]
        crop = s["crop"]
        swat = float(s["yld_t"])
        grac = GRACENET[year]
        _, ref_v = ref_y[year]
        year_rows.append([
            str(year), CROP_NAME[crop], _fmt(swat, 2), _fmt(ref_v, 2), _fmt(grac, 2),
        ])
        by_crop[crop].append((swat, grac))
        by_crop_ref[crop].append((swat, ref_v))

    def pbias(pairs: list[tuple[float, float]]) -> float:
        obs = sum(o for _, o in pairs)
        return 100.0 * sum(s - o for s, o in pairs) / obs

    bias_headers = ["Crop", "Final bias vs GRACEnet", "vs reference"]
    bias_rows = []
    for crop, label in (("alfa", "Alfalfa"), ("corn", "Corn"), ("barl", "Barley")):
        n = len(by_crop[crop])
        bias_rows.append([
            label,
            f"{pbias(by_crop[crop]):+.1f} % (n = {n})",
            f"{pbias(by_crop_ref[crop]):+.1f} %",
        ])
    return (
        "Table 1. Annual dry-matter yield (Mg/ha).",
        year_headers, year_rows,
        "Table 1 (cont.). Per-crop bias.",
        bias_headers, bias_rows,
    )


def data_exp1_profit() -> tuple[str, list[str], list[list[str]], str]:
    ceiling = json.loads((RUNS / "exp1_ceiling.json").read_text())["result"]  # a stage file
    full = json.loads((RUNS / "exp1_irrigation.json").read_text())
    controller = json.loads((RUNS / "exp1_controller.json").read_text())
    grower_path = RUNS / "exp1_grower_rule.json"
    grower = json.loads(grower_path.read_text()) if grower_path.is_file() else None

    mp = full["mean_profit"]
    paired = full["paired_test"]
    ceil = ceiling["paired_test"]

    headers = ["Row", "Test profit", "vs measured", "vs default"]
    rows: list[list[str]] = [
        ["Logged schedule (replayed)", _fmt(mp["measured_test"]), "0", "—"],
        [
            "Grower rule (fitted to logs)",
            _fmt(grower["test"]) if grower and grower.get("scored") else "—*",
            _fmt(grower["test"] - mp["measured_test"])
            if grower and grower.get("scored") else "—*",
            "—",
        ],
        [
            "Generated monthly default",
            _fmt(mp["default_test"]),
            _paired(paired["default_vs_measured"]),
            "0",
        ],
        [
            "CMA-ES open-loop (fixed)",
            _fmt(mp["fixed_test"]),
            "—",
            _paired(paired["fixed_vs_default"]),
        ],
        ["CMA feedback controller", _fmt(controller["test"]), "—", "—"],
        [
            "Ceiling (oracle − shared)",
            _paired(ceil),
            "—",
            "—",
        ],
    ]

    if "policy_test" in mp:
        def seeds(k):
            vals = " / ".join(f"{v:+.0f}" for v in full[k].values())
            ok = full["sign_consistent"][k]
            return f"{vals}" + ("" if ok else " (signs differ: unresolvable)")

        rows += [
            ["PPO policy", _fmt(mp["policy_test"]), "—", "—"],
            ["Frozen plan", _fmt(mp["frozen_test"]), "—", "—"],
            ["Policy − fixed", _paired(paired["policy_vs_fixed"]), seeds("advantage_over_fixed"), ""],
            ["Frozen − policy", _paired(paired["frozen_vs_policy"]), seeds("adaptivity_value"), ""],
        ]
        note = _ess_note(paired["fixed_vs_default"]) + " Per-seed values in column 3."
    else:
        rows += [
            ["PPO policy", "—*", "—*", "—*"],
            ["Frozen plan", "—*", "—*", "—*"],
            ["Policy − fixed", "—*", "", ""],
            ["Policy − frozen (adaptivity)", "—*", "", ""],
        ]
        note = (
            "*TODO: run PPO for runs/exp1_irrigation.json "
            "(3 PPO seeds; mean ± SE for advantage and adaptivity). "
            + _ess_note(paired["fixed_vs_default"])
        )
    title = "Exp 1 test profit ($/ha), window starts 2013–2017."
    return title, headers, rows, note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RUNS / "paper_tables")
    ap.add_argument("--no-png", action="store_true", help="skip PNG export")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t, h, r = data_env_compare()
    (args.out / "table_env_compare.md").write_text(_md(t, h, r))
    if not args.no_png:
        save_table_png(
            args.out / "table_env_compare.png", t, h, r,
            col_widths=[2.6, 3.0, 1.1, 0.9, 1.4],
        )

    t1, h1, r1, t1b, h1b, r1b = data_yield()
    (args.out / "table_yield.md").write_text(_md(t1, h1, r1) + "\n" + _md(t1b, h1b, r1b))
    if not args.no_png:
        save_table_png(
            args.out / "table_yield.png", t1, h1, r1,
            col_widths=[0.8, 1.0, 1.0, 1.1, 1.1],
        )
        save_table_png(
            args.out / "table_bias.png", t1b, h1b, r1b,
            col_widths=[1.2, 2.2, 1.4],
        )

    t2, h2, r2, note2 = data_exp1_profit()
    (args.out / "table_exp1_profit.md").write_text(_md(t2, h2, r2, note=note2))
    if not args.no_png:
        save_table_png(
            args.out / "table_exp1_profit.png", t2, h2, r2, note=note2,
            col_widths=[2.8, 1.3, 1.4, 1.4],
        )

    (args.out / "all_tables.md").write_text(
        _md(t, h, r) + "\n" + _md(t1, h1, r1) + "\n" + _md(t1b, h1b, r1b)
        + "\n" + _md(t2, h2, r2, note=note2)
    )
    print(f"-> {args.out}/")
    for p in sorted(args.out.iterdir()):
        print(f"   {p.name}  ({p.stat().st_size // 1024} KB)" if p.suffix == ".png"
              else f"   {p.name}")


if __name__ == "__main__":
    main()
