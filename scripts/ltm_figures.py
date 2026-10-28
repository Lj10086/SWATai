"""Inline-SVG figures for runs/ltm_results.html (imported by scripts/ltm_results.py).

Plain SVG, no libraries: white surface, Calibri, hover via <title>. Colours: the dataviz
reference categorical slots, validated on #ffffff (strategies: slots 1-3, all-pairs pass;
crops: slots 1-6, adjacent pass). Aqua/yellow/magenta are < 3:1 on white, so every mark that
uses them is also direct-labelled or has its numbers in the page's tables.
"""
from __future__ import annotations

import html
import math

STRAT_COL = {"profit": "#2a78d6", "balanced": "#eb6834", "sustainability": "#1baf7a"}
CROP_COL = {"barl": "#2a78d6", "sgbt": "#eb6834", "wwht": "#1baf7a", "pota": "#eda100",
            "alfa": "#e87ba4", "corn": "#008300"}
CROP_INK = {"barl": "#fff", "corn": "#fff"}          # white text on the two dark fills
TRIAL = "#52514e"
DEFAULT = "#9a9893"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FONT = "font-family:Calibri,sans-serif"


def e(x) -> str:
    return html.escape(str(x))


def svg(w: int, h: int, body: str, label: str) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{e(label)}" style="{FONT};max-width:100%;height:auto">{body}</svg>')


def text(x, y, s, size=12, anchor="start", color=INK, weight="normal", extra="") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}" font-weight="{weight}" {extra}>{e(s)}</text>')


def ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.floor(lo / step + 1e-9) * step         # cover [lo, hi]: the domain is the ticks
    out = [round(t, 10)]
    while out[-1] < hi - 1e-9:
        out.append(round(out[-1] + step, 10))
    return out


def fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 10 or v == 0 else f"{v:g}"


class Axes:
    """A plot rectangle with linear x/y scales."""

    def __init__(self, x0, y0, w, h, xd, yd):
        self.x0, self.y0, self.w, self.h, self.xd, self.yd = x0, y0, w, h, xd, yd

    def x(self, v):
        return self.x0 + (v - self.xd[0]) / (self.xd[1] - self.xd[0]) * self.w

    def y(self, v):
        return self.y0 + self.h - (v - self.yd[0]) / (self.yd[1] - self.yd[0]) * self.h

    def grid(self, xt=None, yt=None, xlab="", ylab="", xfmt=fmt, yfmt=fmt) -> str:
        out = []
        for t in yt or []:
            out.append(f'<line x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.y(t):.1f}" '
                       f'y2="{self.y(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
            out.append(text(self.x0 - 6, self.y(t) + 4, yfmt(t), 11, "end", INK2))
        for t in xt or []:
            out.append(f'<line y1="{self.y0}" y2="{self.y0 + self.h}" x1="{self.x(t):.1f}" '
                       f'x2="{self.x(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
            out.append(text(self.x(t), self.y0 + self.h + 15, xfmt(t), 11, "middle", INK2))
        if xlab:
            out.append(text(self.x0 + self.w / 2, self.y0 + self.h + 32, xlab, 12, "middle", INK2))
        if ylab:
            cx, cy = self.x0 - 46, self.y0 + self.h / 2
            out.append(text(cx, cy, ylab, 12, "middle", INK2,
                            extra=f'transform="rotate(-90 {cx:.1f} {cy:.1f})"'))
        return "".join(out)


def legend(items: list[tuple[str, str, str]], x: float, y: float) -> str:
    """items: (label, colour, shape in {'dot','ring','square'})."""
    out, cx = [], x
    for label, col, shape in items:
        if shape == "dot":
            out.append(f'<circle cx="{cx + 5}" cy="{y - 4}" r="5" fill="{col}"/>')
        elif shape == "ring":
            out.append(f'<circle cx="{cx + 5}" cy="{y - 4}" r="5" fill="#fff" stroke="{col}" '
                       f'stroke-width="2"/>')
        else:
            out.append(f'<rect x="{cx}" y="{y - 9}" width="10" height="10" rx="2" fill="{col}"/>')
        out.append(text(cx + 15, y, label, 12, color=INK))
        cx += 22 + 6.3 * len(label)
    return "".join(out)


# ---------------------------------------------------------------------------------------------
def tradeoff(D: dict, es_mean, names: dict) -> str:
    """Profit vs nitrate leaching, one panel per experiment (own y scale)."""
    pw, ph, gap, top, left = 300, 260, 80, 56, 70
    W = left + 3 * pw + 2 * gap + 20
    H = top + ph + 60
    body = [legend([("trial practice (replayed)", TRIAL, "dot"), ("default", DEFAULT, "ring"),
                    ("best fixed plan", INK2, "dot"), ("learned policy", INK2, "ring")]
                   + [(s, c, "square") for s, c in STRAT_COL.items()], left, 16)]
    for i, n in enumerate((1, 2, 3)):
        d = D[n]
        pts = []
        tm0 = d["strategies"]["profit"]["test_means"]
        pts.append(("trial practice", TRIAL, "dot", tm0["measured"]))
        pts.append(("default", DEFAULT, "ring", tm0["default"]))
        if "auto" in tm0:
            pts.append(("auto-irrigation", DEFAULT, "tri", tm0["auto"]))
        for s, col in STRAT_COL.items():
            v = d["strategies"][s]
            pts.append((f"best fixed, {s}", col, "dot", v["test_means"]["fixed"]))
            pts.append((f"learned policy, {s}", col, "ring", es_mean(v)))
        # a lone loss-making point (Exp 3 sustainability: no irrigation) would squash the rest:
        # draw it pinned to the panel's bottom edge, labelled with its value
        ys = [p[3]["profit"] for p in pts]
        xs = [p[3]["no3"] for p in pts]
        off = [p for p in pts if p[3]["profit"] < 0]
        yt = ticks(0 if off else min(0, min(ys)), max(ys) * 1.12)
        xt = ticks(0, max(xs) * 1.05)
        ax = Axes(left + i * (pw + gap), top, pw, ph, (xt[0], xt[-1]), (yt[0], yt[-1]))
        body.append(text(ax.x0, top - 14, f"Exp {n}: {names[n]}", 13, weight="bold"))
        body.append(ax.grid(xt, yt, "NO₃-N leached, kg/ha/yr", "Profit, $/ha/yr" if i == 0 else ""))
        if yt[0] < 0:
            body.append(f'<line x1="{ax.x0}" x2="{ax.x0 + pw}" y1="{ax.y(0):.1f}" '
                        f'y2="{ax.y(0):.1f}" stroke="{INK2}" stroke-width="1"/>')
        body.append(text(ax.x0 + 6, top + 14, "↖ better: more profit, less leaching", 11,
                         color=INK2))
        # fixed dots first, rings on top (a ring around a dot = policy equals fixed)
        order = sorted(pts, key=lambda p: {"dot": 0, "tri": 0, "ring": 1}[p[2]])
        for label, col, shape, m in order:
            cx, cy = ax.x(m["no3"]), ax.y(max(m["profit"], yt[0]))
            tip = (f"<title>Exp {n}, {label}: profit {m['profit']:,.0f} $/ha/yr, NO₃ "
                   f"{m['no3']:.1f} kg/ha/yr, water {m['water']:.0f} mm/yr, score "
                   f"{m['score']:.2f}</title>")
            if shape == "dot":
                body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{col}" '
                            f'stroke="#fff" stroke-width="2">{tip}</circle>')
            elif shape == "tri":
                body.append(f'<path d="M{cx:.1f},{cy - 6:.1f} L{cx + 6:.1f},{cy + 5:.1f} '
                            f'L{cx - 6:.1f},{cy + 5:.1f} Z" fill="{col}">{tip}</path>')
            else:
                body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="none" '
                            f'stroke="{col}" stroke-width="2">{tip}</circle>')
            if m["profit"] < yt[0] and shape == "dot":
                body.append(text(cx + 12, cy - 6, f"↓ off scale: {label} "
                                 f"({m['profit']:,.0f} $/ha/yr)", 11, color=INK2))
            if label in ("trial practice", "default", "auto-irrigation"):
                dx = -10 if label == "trial practice" else 12
                body.append(text(cx + dx, cy + (14 if label == "default" else -8), label, 11,
                                 "end" if dx < 0 else "start", INK2))
    return svg(W, H, "".join(body), "Profit against nitrate leaching for every arm")


def forest(D: dict, names: dict) -> str:
    """Policy - fixed and adaptivity, per seed, with ci95_ess."""
    rows = [(n, s) for n in (1, 2, 3) for s in STRAT_COL]
    row_h, top, left, pw, gap = 30, 48, 230, 300, 50
    H = top + row_h * len(rows) + 50
    W = left + 2 * pw + gap + 20
    keys = [("policy_vs_fixed", "Learned policy − best fixed plan"),
            ("adaptivity_value", "Value of adapting (policy − its own frozen plan)")]
    allv = [x for n, s in rows for k, _ in keys for sd in D[n]["strategies"][s]["es"]
            for x in D[n]["strategies"][s]["es"][sd][k]["score"]["ci95_ess"]]
    lo, hi = min(allv + [0]), max(allv + [0])
    pad = 0.1 * (hi - lo)
    xt = ticks(lo - pad, hi + pad, 6)
    body = []
    for j, (k, title) in enumerate(keys):
        ax = Axes(left + j * (pw + gap), top, pw, row_h * len(rows), (xt[0], xt[-1]), (0, 1))
        body.append(text(ax.x0, top - 22, title, 13, weight="bold"))
        body.append(ax.grid(xt, None, "score difference (test)", "", xfmt=lambda t: f"{t:+.2f}"))
        body.append(text(ax.x(0) - 4, top - 6, "← fixed / frozen better", 11, "end", INK2))
        body.append(text(ax.x(0) + 4, top - 6, "policy better →", 11, "start", INK2))
        body.append(f'<line x1="{ax.x(0):.1f}" x2="{ax.x(0):.1f}" y1="{top}" '
                    f'y2="{top + row_h * len(rows)}" stroke="{INK}" stroke-width="1.5"/>')
        for r, (n, s) in enumerate(rows):
            es = D[n]["strategies"][s]["es"]
            yc = top + r * row_h + row_h / 2
            if j == 0:
                lab = f"Exp {n} {names[n].split(':')[0].lower()}, {s}"
                body.append(text(left - 10, yc + 4, lab, 12, "end"))
            if r and rows[r - 1][0] != n:
                body.append(f'<line x1="{20 if j == 0 else ax.x0}" x2="{ax.x0 + pw}" '
                            f'y1="{yc - row_h / 2:.1f}" y2="{yc - row_h / 2:.1f}" '
                            f'stroke="{GRID}" stroke-width="1"/>')
            for q, sd in enumerate(es):
                c = es[sd][k]["score"]
                yy = yc + (q - 1.5) * 5
                a, b = c["ci95_ess"]
                tip = (f"<title>Exp {n}, {s}, seed {sd}: {c['mean']:+.3f} "
                       f"[{a:+.3f}, {b:+.3f}]</title>")
                body.append(f'<line x1="{ax.x(a):.1f}" x2="{ax.x(b):.1f}" y1="{yy:.1f}" '
                            f'y2="{yy:.1f}" stroke="{STRAT_COL[s]}" stroke-width="2" '
                            f'stroke-linecap="round">{tip}</line>')
                body.append(f'<circle cx="{ax.x(c["mean"]):.1f}" cy="{yy:.1f}" r="3.5" '
                            f'fill="{STRAT_COL[s]}" stroke="#fff" stroke-width="1">{tip}</circle>')
    return svg(W, H, "".join(body), "Policy minus fixed plan and adaptivity per seed")


def rotations(rows: list[tuple[str, list[str]]], names: dict) -> str:
    """Crop strips: one row per plan, one cell per season, crop name in the cell."""
    cw, ch, left, top = 92, 30, 250, 26
    W = left + 8 * cw + 10
    H = top + len(rows) * (ch + 8) + 10
    body = [text(left + i * cw + cw / 2, 16, f"year {i + 1}", 11, "middle", INK2) for i in range(8)]
    for r, (label, crops) in enumerate(rows):
        y = top + r * (ch + 8)
        body.append(text(left - 10, y + ch / 2 + 4, label, 12, "end"))
        for i, c in enumerate(crops):
            x = left + i * cw
            body.append(f'<rect x="{x + 1}" y="{y}" width="{cw - 2}" height="{ch}" rx="4" '
                        f'fill="{CROP_COL[c]}"><title>{e(label)}, year {i + 1}: '
                        f'{e(names[c])}</title></rect>')
            body.append(text(x + cw / 2, y + ch / 2 + 4, names[c], 12, "middle",
                             CROP_INK.get(c, INK), extra='pointer-events="none"'))
    return svg(W, H, "".join(body), "Rotations chosen in Exp 1")


def hbars(panels: list[tuple[str, str, list[tuple[str, str, float]]]]) -> str:
    """Side-by-side horizontal bar panels: (title, unit, [(label, colour, value)])."""
    left, pw, gap, bh, top = 230, 260, 60, 20, 34
    n = len(panels[0][2])
    H = top + n * (bh + 8) + 40
    W = left + len(panels) * pw + (len(panels) - 1) * gap + 70
    body = []
    for j, (title, unit, items) in enumerate(panels):
        vmax = max(v for _, _, v in items) or 1
        xt = ticks(0, vmax * 1.05, 4)
        ax = Axes(left + j * (pw + gap), top, pw, n * (bh + 8), (0, xt[-1]), (0, 1))
        body.append(text(ax.x0, top - 14, title, 13, weight="bold"))
        body.append(ax.grid(xt, None, unit))
        for r, (label, col, v) in enumerate(items):
            y = top + r * (bh + 8) + 4
            if j == 0:
                body.append(text(left - 10, y + bh / 2 + 4, label, 12, "end"))
            wpx = max(0.0, ax.x(v) - ax.x0)
            body.append(f'<rect x="{ax.x0}" y="{y}" width="{wpx:.1f}" height="{bh}" rx="4" '
                        f'fill="{col}"><title>{e(label)}: {v:,.1f} {e(unit)}</title></rect>')
            body.append(text(ax.x0 + wpx + 5, y + bh / 2 + 4, f"{v:,.1f}", 11, color=INK2))
    return svg(W, H, "".join(body), "Nitrogen inputs per plan")


def season_bars(seasons: list[str], series: list[tuple[str, str, list[float]]], unit: str) -> str:
    """Grouped vertical bars, one group per season."""
    left, top, gw, ph = 70, 40, 104, 240
    W = left + gw * len(seasons) + 20
    H = top + ph + 58
    vmax = max(max(v) for _, _, v in series)
    yt = ticks(0, vmax * 1.08, 5)
    ax = Axes(left, top, gw * len(seasons), ph, (0, len(seasons)), (0, yt[-1]))
    body = [legend([(lab, col, "square") for lab, col, _ in series], left, 16),
            ax.grid(None, yt, "", unit)]
    k = len(series)
    bw = (gw - 20 - 2 * (k - 1)) / k
    for i, s in enumerate(seasons):
        gx = left + i * gw + 10
        body.append(text(left + i * gw + gw / 2, top + ph + 16, f"{i + 1}: {s}", 11, "middle", INK2))
        for j, (lab, col, vals) in enumerate(series):
            v = vals[i]
            x = gx + j * (bw + 2)
            y = ax.y(v)
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                        f'height="{ax.y(0) - y:.1f}" rx="3" fill="{col}"><title>season {i + 1} '
                        f'({e(s)}), {e(lab)}: {v:,.0f} {e(unit)}</title></rect>')
    body.append(f'<line x1="{left}" x2="{left + gw * len(seasons)}" y1="{ax.y(0):.1f}" '
                f'y2="{ax.y(0):.1f}" stroke="{INK2}" stroke-width="1"/>')
    return svg(W, H, "".join(body), "Irrigation per season")
