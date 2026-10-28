"""Closed-loop LTM episodes: one step per decision point of a lever (practice.decisions).

As ``swat_gym.crop_env``: every step re-simulates the whole window with the decisions so far
(undecided monthly amounts = 0; undecided seasons of the rotation = the trial's crop, or
the first feasible of FILL_ORDER where that would break a rule), so the final step's outputs are exactly the open-loop
outputs of the final plan (tests/test_ltm.py: episode == open loop, every term), and an
observation only reads state at or before the decision it precedes (CLAUDE.md rule 4).

Actions: R picks a crop index (forced inside an alfalfa stand; masked by the rules); N and I
take a unit-box value per decision. Policies act in residuals around ``measured_x`` (N, I)
or with a bonus on the trial's crop (R), so the zero policy replays measured practice.
"""
from __future__ import annotations

import numpy as np

from ..plan import n_amount
from .practice import (ALFA_STAND, CROPS, N_MONTH_MAX, ROTATION_GAP, ROTATION_NO_REPEAT,
                       TRIAL_ROTATION, decisions, irr_cap, irr_months, measured_x, n_months,
                       trial_season)
from .render import EpisodePlan
from .windows import SEASONS

OBS_DIM = 23
KINDS = ("crop", "manure", "monthly")


def feasible(history: list[str], k: int) -> np.ndarray:
    """Mask over CROPS for season k given the crops of seasons 1..k-1 (render.validate's
    rules: an alfalfa stand runs its 3 years; no crop two years running outside a stand; potato
    at most once in any ROTATION_GAP years; a stand must fit before season 8 ends)."""
    if history and history[-1] == "alfa" and _stand_len(history) % ALFA_STAND:
        return np.array([c == "alfa" for c in CROPS])
    prev = history[-1] if history else None
    ok = np.ones(len(CROPS), bool)
    for i, c in enumerate(CROPS):
        if c in ROTATION_NO_REPEAT and c == prev:
            ok[i] = False
        gap = ROTATION_GAP.get(c, 0)
        if gap and c in history[max(0, k - gap):]:
            ok[i] = False
        if c == "alfa" and k + ALFA_STAND - 1 > SEASONS:
            ok[i] = False
    return ok


#: Fill order for undecided seasons when the trial's crop is infeasible there.
FILL_ORDER = ("barl", "wwht", "sgbt", "corn", "pota")


def fill_rotation(chosen: list[str]) -> tuple[str, ...]:
    """Decided seasons + a valid fill for the rest (the trial's crop, else the first feasible
    of FILL_ORDER; alfalfa only to finish a stand)."""
    crops = list(chosen)
    while len(crops) < SEASONS:
        k = len(crops) + 1
        ok = dict(zip(CROPS, feasible(crops, k)))
        if crops and crops[-1] == "alfa" and _stand_len(crops) % ALFA_STAND:
            crops.append("alfa")
            continue
        c = TRIAL_ROTATION[k - 1]
        if not ok[c] or c == "alfa":
            c = next(x for x in FILL_ORDER if ok[x])
        crops.append(c)
    return tuple(crops)


def _stand_len(crops: list[str]) -> int:
    n = 0
    for c in reversed(crops):
        if c != "alfa":
            break
        n += 1
    return n


def plan_from_x(lever: str, x: np.ndarray, t_done: int | None = None) -> EpisodePlan:
    """Unit-box decision vector (R: crop indices) -> plan; decisions >= t_done undecided."""
    ds = decisions(lever)
    t_done = len(ds) if t_done is None else t_done
    probes = ds
    if lever == "R":
        chosen = [CROPS[int(x[i])] for i in range(t_done)]
        # Exp 1 auto-irrigates every season (user decision 2026-10-25): one rule for all crops
        return EpisodePlan(crops=fill_rotation(chosen), probes=probes, irr_auto=True)
    manure = [1.0] * SEASONS
    n_kg = [None] * SEASONS
    irr = [None] * SEASONS
    caps = irr_cap()
    monthly: dict[int, dict[int, float]] = {}
    for i, d in enumerate(ds):
        v = float(x[i]) if i < t_done else 0.0
        if d.kind == "manure":
            manure[d.k] = v
        else:
            monthly.setdefault(d.k, {})[d.mon] = v
    for k in range(1, SEASONS + 1):
        s = trial_season(k)
        m = monthly.get(k, {})
        if lever == "N":
            n_kg[k - 1] = tuple(float(n_amount(m.get(mon, 0.0), N_MONTH_MAX))
                                for mon in n_months(s))
        else:
            irr[k - 1] = tuple(float(np.clip(m.get(mon, 0.0), 0.0, 1.0)) * caps.get(mon, 0.0)
                               for mon in irr_months(s))
    if lever == "N":
        return EpisodePlan(manure_x=tuple(manure), n_kg=tuple(n_kg), probes=probes)
    return EpisodePlan(irr_mm=tuple(irr), probes=probes)


class Episode:
    """One closed-loop episode; the runner is supplied by the caller."""

    def __init__(self, lever: str) -> None:
        self.lever = lever
        self.ds = decisions(lever)
        self.x0 = measured_x(lever)
        self.reset(0)

    def reset(self, start: int) -> np.ndarray:
        self.start, self.t = int(start), 0
        self.x = np.zeros(len(self.ds))
        self.chosen: list[str] = []
        self.out: dict | None = None
        return self.obs()

    @property
    def done(self) -> bool:
        return self.t >= len(self.ds)

    # -- actions ------------------------------------------------------------------------------
    def mask(self) -> np.ndarray:
        return feasible(self.chosen, self.ds[self.t].k)

    def absolute(self, a) -> float:
        """Policy output -> this decision's value (R: crop index from logits)."""
        if self.lever == "R":
            logits = np.asarray(a, float).ravel()[:len(CROPS)].copy()
            logits[CROPS.index(TRIAL_ROTATION[self.ds[self.t].k - 1])] += R_BONUS
            logits[~self.mask()] = -np.inf
            return float(int(np.argmax(logits)))
        r = float(np.clip(np.asarray(a, float).ravel()[0], -1.0, 1.0))
        return float(np.clip(self.x0[self.t] + r, 0.0, 1.0))

    def write(self, value: float) -> EpisodePlan:
        if self.lever == "R":
            ok = self.mask()
            if not ok[int(value)]:
                raise ValueError(f"crop {CROPS[int(value)]} infeasible at season "
                                 f"{self.ds[self.t].k} after {self.chosen}")
            self.chosen.append(CROPS[int(value)])
        self.x[self.t] = value
        return plan_from_x(self.lever, self.x, self.t + 1)

    def observe(self, out: dict) -> np.ndarray:
        self.out = out
        self.t += 1
        return self.obs()

    # -- observation --------------------------------------------------------------------------
    def obs(self) -> np.ndarray:
        o = np.zeros(OBS_DIM, np.float32)
        T = len(self.ds)
        d = self.ds[min(self.t, T - 1)]
        o[0] = self.t / T
        o[1] = d.k / SEASONS
        o[2 + KINDS.index("monthly" if d.kind in ("n", "irr") else d.kind)] = 1.0
        o[5], o[6] = np.sin(2 * np.pi * d.mon / 12), np.cos(2 * np.pi * d.mon / 12)
        if self.lever == "R":
            crop = self.chosen[-1] if self.chosen else None
        else:
            crop = TRIAL_ROTATION[d.k - 1] if d.k >= 1 else None
        if crop:
            o[7 + CROPS.index(crop)] = 1.0
        o[13:23] = self._state(d)
        return o

    def _state(self, d) -> np.ndarray:
        """Soil and last-month weather at the decision date, from the previous step's run.
        (numpy masks over the tables' columns: the same selections as pandas, ~10x faster.)"""
        s = np.array([0.5, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
        if self.out is None or self.done:
            return s
        c = _cols(self.out)
        year = self.start + d.k
        m = c["mgt"]
        i = np.flatnonzero(m["probe"] & (m["yr"] == year) & (m["mon"] == d.mon) & (m["day"] == d.day))
        if len(i):
            j = i[0]
            s[0], s[1], s[2] = (m["soil_water"][j] / 300.0, m["soil_no3"][j] / 500.0,
                                m["soil_solp"][j] / 500.0)
        py, pm = (year, d.mon - 1) if d.mon > 1 else (year - 1, 12)
        wb, pw, nb = c["wb"], c["pw"], c["nb"]
        w = np.flatnonzero((wb["yr"] == py) & (wb["mon"] == pm))
        p = np.flatnonzero((pw["yr"] == py) & (pw["mon"] == pm))
        if len(w):
            s[3], s[4] = wb["precip"][w[0]] / 50.0, wb["pet"][w[0]] / 200.0
        if len(p):
            s[5], s[6] = pw["strsn"][p[0]] / 31.0, pw["strsw"][p[0]] / 31.0
        s[7] = float(wb["irr"][(wb["yr"] == year) & (wb["mon"] < d.mon)].sum()) / 800.0
        s[8] = float(nb["fertn"][(nb["yr"] == year) & (nb["mon"] < d.mon)].sum()) / 300.0
        s[9] = float(m["op_var"][m["harv"] & (m["yr"] == year - 1)].sum()) / 1000.0 / 20.0
        return s


def _cols(out: dict) -> dict:
    """The observation's columns as numpy arrays, built once per output dict."""
    c = out.get("_obs_cols")
    if c is None:
        mgt = out["mgt"]
        m = {k: mgt[k].to_numpy() for k in ("yr", "mon", "day", "op_var", "soil_water",
                                              "soil_no3", "soil_solp")}
        m["probe"] = (mgt.name.to_numpy() == "elem_n") & (m["op_var"] == 0.0)
        m["harv"] = mgt.operation.isin(["HARV/KILL", "HARVEST"]).to_numpy()
        tab = {"wb": ("hru_wb_mon.txt", ("yr", "mon", "precip", "pet", "irr")),
               "pw": ("hru_pw_mon.txt", ("yr", "mon", "strsn", "strsw")),
               "nb": ("hru_nb_mon.txt", ("yr", "mon", "fertn"))}
        c = {"mgt": m, **{k: {col: out[t][col].to_numpy() for col in cols}
                          for k, (t, cols) in tab.items()}}
        out["_obs_cols"] = c
    return c


#: R: logit bonus on the trial's crop. Small enough that sigma-0.02 ES perturbations of the
#: 64-unit last hidden layer (~0.02 x 8 = 0.16 in a logit) can overturn it.
R_BONUS = 0.1


def run_policy(policy, lever: str, runner, n: int, start: int,
               truncate: bool = True) -> tuple[list, list, list]:
    """Closed-loop episodes of ``n`` policies on one window, all in the same engine runs.

    ``policy(obs[n, D]) -> raw outputs [n, A]``. Returns (final plans, final outputs, x).
    ``truncate``: an intermediate step simulates only through Dec 31 of the next decision's
    year -- everything its observation reads (SWAT+ is causal, so those days are identical to
    the full run's; tests/test_ltm.py::test_truncated_steps_observe_the_same). The final step
    always runs the whole window."""
    eps = [Episode(lever) for _ in range(n)]
    obs = np.stack([e.reset(start) for e in eps])
    plans: list = [None] * n
    outs: list = [None] * n
    ds = eps[0].ds
    while not eps[0].done:
        raw = policy(obs)
        plans = [e.write(e.absolute(a)) for e, a in zip(eps, raw)]
        t = eps[0].t
        last = start + ds[t + 1].k if truncate and t + 1 < len(ds) else None
        outs = runner.run(plans, start, last)
        obs = np.stack([e.observe(o) for e, o in zip(eps, outs)])
    return plans, outs, [e.x.copy() for e in eps]
