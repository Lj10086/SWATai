"""Exp 1: irrigation lever (monthly). Rows: measured, default, fixed (CMA-ES), PPO, frozen.

Run after exp1_ceiling, exp1_controller, exp1_grower_rule.

    uv run python -m swat_gym.experiments.exp1_irrigation --budget 300000 --ppo-seeds 3
"""
from .common import RUNS, run


def main(argv=None) -> dict:
    return run("I", RUNS / "exp1_irrigation.json", argv)


if __name__ == "__main__":
    main()
