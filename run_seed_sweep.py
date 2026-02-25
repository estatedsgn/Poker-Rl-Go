"""Run repeated synthetic training over multiple seeds for stability checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from train import TrainConfig, run_training


def parse_args():
    p = argparse.ArgumentParser(description="Run synthetic training sweep across seeds")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out-root", type=str, default="sweep_runs")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]

    for s in seeds:
        out_dir = Path(a.out_root) / f"seed_{s}"
        cfg = TrainConfig(steps=a.steps, seed=s, out_dir=str(out_dir), backend="synthetic", ppo_epochs=1)
        metrics = run_training(cfg)
        print(f"seed={s} metrics={metrics}")
