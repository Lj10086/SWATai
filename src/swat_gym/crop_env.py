"""Envs for monoculture crop models: one step = one lever month of the scored season.

Like the rotation's ``SwatEnv``: each step re-simulates the whole window with the decisions
so far (undecided months = 0; SWAT+ has no restart), so episode profit equals the open-loop
profit of the final x, and observations read only the month just decided.

``CropEnv.step`` takes absolute unit-box actions; the gym/vec interfaces take residuals in
[-1, 1] around ``default_x`` so an untrained policy starts where CMA-ES does.

``CropVecEnv`` steps all its envs in one engine run (K HRUs) when the spec is batchable.
One engine run has one time.sim, so the envs share a weather window per episode round;
the window is resampled every round.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import numpy as np

from .crop import CACHE, CROP_RENDER_VERSION, CropRunner, Plan, lever_months, profit, to_unit
from .engine import EngineError
from .fastrunner import model_hash
from .rewarders import Prices
from .spec import CropSpec
from .windows import CROP_TRAIN_YEARS

OBS_DIM = 9
REWARD_SCALE = 1e-3
INITIAL_STATE = {"sw": 250.0, "strsn": 0.0, "strsw": 0.0, "precip": 20.0, "pet": 100.0,
                 "n_so_far": 0.0, "irr_so_far": 0.0}


# -- default_x: the decision tables' monthly average on train windows -----------------

def default_runs(spec: CropSpec, windows: Sequence[int] = CROP_TRAIN_YEARS,
                 runner: CropRunner | None = None) -> list[dict]:
    """The unmodified model (its own decision tables) on each window: per-HRU outputs."""
    own = runner is None
    runner = runner or CropRunner(spec)
    try:
        return [runner.run([Plan()], sy)[0] for sy in windows]
    finally:
        if own:
            runner.close()


def monthly_amounts(spec: CropSpec, lever: str, out: dict) -> np.ndarray:
    """mm (I) or kg N (N) applied in each lever month of the scored season."""
    table, col = ("hru_wb_mon.txt", "irr") if lever == "I" else ("hru_nb_mon.txt", "fertn")
    df = out[table]
    return np.array([float(df.loc[df["mon"] == m.mon, col].sum())
                     for m in lever_months(spec, lever)])


def default_x(spec: CropSpec, lever: str) -> np.ndarray:
    """Mean monthly amount of the model's decision tables on train windows, in the unit box.

    The CMA-ES start, PPO's residual centre and every lever plan's spin-up year. Cached by
    model hash and render version under runs/cache/.
    """
    key = (f"default_x_{spec.name}_{lever}_{model_hash(spec.txtinout)}"
           f"_r{CROP_RENDER_VERSION[lever]}")
    path = CACHE / f"{key}.json"
    if path.is_file():
        return np.asarray(json.loads(path.read_text())["x"])
    amounts = np.mean([monthly_amounts(spec, lever, o) for o in default_runs(spec)], axis=0)
    x = to_unit(spec, lever, amounts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"x": list(map(float, x)), "windows": CROP_TRAIN_YEARS,
                                "monthly": list(map(float, amounts))}, indent=1))
    return x


# -- one episode ------------------------------------------------------------------------

class _Episode:
    """State of one episode; the runner is supplied by the caller (single or batched)."""

    def __init__(self, spec: CropSpec, lever: str, prices: Prices, no3_price: float,
                 x0: np.ndarray) -> None:
        self.spec, self.lever, self.prices, self.no3_price = spec, lever, prices, no3_price
        self.months = lever_months(spec, lever)
        self.default = np.asarray(x0, dtype=float)
        self.start_year = 0
        self.t = 0
        self.x = np.zeros(len(self.months))
        self.cum_profit = 0.0
        self.last = dict(INITIAL_STATE)

    def reset(self, start_year: int) -> np.ndarray:
        self.start_year, self.t = int(start_year), 0
        self.x = np.zeros(len(self.months))
        self.cum_profit = 0.0
        self.last = dict(INITIAL_STATE)
        return self.obs()

    def absolute(self, residual) -> float:
        r = float(np.clip(np.asarray(residual, dtype=float).ravel()[0], -1.0, 1.0))
        return float(np.clip(self.default[self.t] + r, 0.0, 1.0))

    def write(self, action) -> Plan:
        self.x[self.t] = float(np.clip(np.asarray(action, dtype=float).ravel()[0], 0.0, 1.0))
        return Plan(self.lever, tuple(self.x), tuple(self.default))

    def observe(self, out: dict):
        d = profit(self.spec, out, self.prices, no3_price=self.no3_price)
        d["start_year"] = self.start_year
        d["x"] = self.x.copy()  # the plan so far (the final plan in the terminal info)
        reward = (d["profit"] - self.cum_profit) * REWARD_SCALE
        self.cum_profit = d["profit"]
        self.last = self._state(out, self.months[self.t].mon)
        self.t += 1
        return self.obs(), reward, self.t >= len(self.months), d

    @staticmethod
    def _state(out: dict, mon: int) -> dict:
        """End-of-month state of the decided month (scored season only: nyskip = 1)."""
        wb, pw, nb = out["hru_wb_mon.txt"], out["hru_pw_mon.txt"], out["hru_nb_mon.txt"]
        row = {}
        for name, df in (("hru_wb_mon", wb), ("hru_pw_mon", pw)):
            hit = df[df["mon"] == mon]
            if len(hit) != 1:
                raise EngineError(f"{name}: {len(hit)} rows for month {mon}, expected 1")
            row[name] = hit.iloc[0]
        w, p = row["hru_wb_mon"], row["hru_pw_mon"]
        return {"sw": float(w["sw_final"]), "precip": float(w["precip"]),
                "pet": float(w["pet"]), "strsn": float(p["strsn"]), "strsw": float(p["strsw"]),
                "n_so_far": float(nb.loc[nb["mon"] <= mon, "fertn"].sum()),
                "irr_so_far": float(wb.loc[wb["mon"] <= mon, "irr"].sum())}

    def obs(self) -> np.ndarray:
        s, T = self.last, len(self.months)
        mon = self.months[min(self.t, T - 1)].mon
        return np.array([self.t / T, s["sw"] / 300.0, s["strsn"] / 31.0, s["strsw"] / 31.0,
                         s["precip"] / 50.0, s["pet"] / 200.0, s["n_so_far"] / 250.0,
                         s["irr_so_far"] / 500.0, mon / 12.0], dtype=np.float32)


class CropEnv:
    """Single-HRU crop env with absolute actions (rollouts, frozen plans, parity tests)."""

    def __init__(self, spec: CropSpec, lever: str, *, prices: Prices, no3_price: float = 0.0,
                 train_years: Sequence[int] = CROP_TRAIN_YEARS, seed: int | None = None,
                 runner: CropRunner | None = None) -> None:
        self.spec, self.lever = spec, lever
        self.train_years = list(train_years)
        self.rng = np.random.default_rng(seed)
        self._own = runner is None
        self.runner = runner or CropRunner(spec)
        self.ep = _Episode(spec, lever, prices, no3_price, default_x(spec, lever))
        self.reset(start_year=self.train_years[0])

    @property
    def x(self) -> np.ndarray:
        return self.ep.x

    @property
    def last(self) -> dict:
        return self.ep.last

    @property
    def t(self) -> int:
        return self.ep.t

    @property
    def n_steps(self) -> int:
        return len(self.ep.months)

    def reset(self, *, seed: int | None = None, start_year: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        sy = start_year if start_year is not None else int(self.rng.choice(self.train_years))
        return self.ep.reset(sy), {}

    def absolute(self, residual) -> list[float]:
        return [self.ep.absolute(residual)]

    def step(self, action):
        plan = self.ep.write(action)
        out = self.runner.run([plan], self.ep.start_year)[0]
        obs, reward, done, d = self.ep.observe(out)
        return obs, reward, done, False, d

    def close(self) -> None:
        if self._own:
            self.runner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def rollout(env: CropEnv, policy: Callable[[CropEnv, np.ndarray], Sequence[float]],
            start_year: int) -> dict:
    """One episode; ``policy(env, obs) -> absolute action``. Final profit terms + x."""
    obs, _ = env.reset(start_year=start_year)
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy(env, obs))
    return {**info, "x": env.x.copy()}


# -- vectorised (PPO) -------------------------------------------------------------------

def make_vec_env(spec: CropSpec, lever: str, *, prices: Prices, n_envs: int, seed: int,
                 no3_price: float = 0.0, train_years: Sequence[int] = CROP_TRAIN_YEARS):
    """SB3 VecEnv of ``n_envs`` episodes sharing one window per round (one engine run per
    step when the spec is batchable)."""
    from gymnasium import spaces
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    class CropVecEnv(VecEnv):
        def __init__(self) -> None:
            self.render_mode = None
            self.runner = CropRunner(spec, k=n_envs)
            x0 = default_x(spec, lever)
            self.eps = [_Episode(spec, lever, prices, no3_price, x0) for _ in range(n_envs)]
            self.rng = np.random.default_rng(seed)
            self.train_years = list(train_years)
            self._actions = None
            super().__init__(n_envs, spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
                             spaces.Box(-1.0, 1.0, (1,), np.float32))

        def _new_round(self) -> np.ndarray:
            sy = int(self.rng.choice(self.train_years))
            return np.stack([e.reset(sy) for e in self.eps])

        def reset(self):
            if self._seeds[0] is not None:
                self.rng = np.random.default_rng(self._seeds[0])
            self._seeds = [None] * self.num_envs
            return self._new_round()

        def step_async(self, actions) -> None:
            self._actions = actions

        def step_wait(self):
            plans = [e.write([e.absolute(a)]) for e, a in zip(self.eps, self._actions)]
            outs = self.runner.run(plans, self.eps[0].start_year)
            res = [e.observe(o) for e, o in zip(self.eps, outs)]
            obs = np.stack([r[0] for r in res])
            rewards = np.array([r[1] for r in res], dtype=np.float32)
            dones = np.array([r[2] for r in res])
            infos = [dict(r[3]) for r in res]
            if dones.all():  # equal-length episodes: all end together
                for i, info in enumerate(infos):
                    info["terminal_observation"] = obs[i]
                    info["TimeLimit.truncated"] = False
                obs = self._new_round()
            elif dones.any():
                raise RuntimeError("crop episodes ended out of step")
            return obs, rewards, dones, infos

        def runner_stats(self) -> dict:
            return {"n_runs": self.runner.n_runs, "n_skipped": self.runner.n_skipped,
                    "seasons": self.runner.n_seasons}

        def close(self) -> None:
            self.runner.close()

        def get_attr(self, attr_name, indices=None):
            if attr_name == "render_mode":
                return [None] * self.num_envs
            raise AttributeError(attr_name)

        def set_attr(self, attr_name, value, indices=None) -> None:
            raise AttributeError(attr_name)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            raise AttributeError(method_name)

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * self.num_envs

    return CropVecEnv()

