"""Evaluate a pool of checkpoints on RLCard and rank by avg payoff/hand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from evaluate import EvalConfig, run_evaluation


def evaluate_checkpoint_pool(checkpoints_dir: str, hands: int = 200, obs_dim: int = 128, num_actions: int = 16, seed: int = 42) -> List[Tuple[str, float]]:
    ckpts = sorted(Path(checkpoints_dir).glob("*.pt"))
    if not ckpts:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")

    results = []
    for ckpt in ckpts:
        score = run_evaluation(
            EvalConfig(
                checkpoint=str(ckpt),
                obs_dim=obs_dim,
                num_actions=num_actions,
                hands=hands,
                seed=seed,
            )
        )
        results.append((str(ckpt), score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Rank checkpoint pool by average payoff per hand")
    p.add_argument("--checkpoints-dir", type=str, required=True)
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--obs-dim", type=int, default=128)
    p.add_argument("--num-actions", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    ranking = evaluate_checkpoint_pool(
        checkpoints_dir=a.checkpoints_dir,
        hands=a.hands,
        obs_dim=a.obs_dim,
        num_actions=a.num_actions,
        seed=a.seed,
    )
    for i, (path, score) in enumerate(ranking, 1):
        print(f"{i:02d}. {path} -> {score:.6f}")
