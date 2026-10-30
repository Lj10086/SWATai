"""Exp 2: nitrogen lever (monthly mineral N + April manure; irrigation at measured practice).

    uv run python -m swat_gym.experiments.exp2_nitrogen --budget 300000 --ppo-seeds 3
"""
from .common import RUNS, run


def main(argv=None) -> dict:
    return run("N", RUNS / "exp2_nitrogen.json", argv)


if __name__ == "__main__":
    main()
