"""Evaluate a pool of checkpoints on RLCard and rank by avg payoff/hand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Literal, Tuple

from evaluate import run_evaluation_over_seeds

RankMetric = Literal["mean", "lcb"]


def _build_seeds(seed_start: int, num_seeds: int) -> List[int]:
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    return [seed_start + i for i in range(num_seeds)]


def _ranking_value(mean_payoff: float, ci95_half_width: float, rank_metric: RankMetric, ci_penalty: float) -> float:
    if rank_metric == "mean":
        return mean_payoff
    if rank_metric == "lcb":
        return mean_payoff - ci_penalty * ci95_half_width
    raise ValueError(f"Unsupported rank_metric: {rank_metric}")


def evaluate_checkpoint_pool(
    checkpoints_dir: str,
    hands: int = 200,
    obs_dim: int = 128,
    num_actions: int = 16,
    seed_start: int = 42,
    num_seeds: int = 3,
    rank_metric: RankMetric = "mean",
    ci_penalty: float = 1.0,
) -> List[Tuple[str, float, float, float, int]]:
    ckpts = sorted(Path(checkpoints_dir).glob("*.pt"))
    if not ckpts:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")

    seeds = _build_seeds(seed_start, num_seeds)

    results = []
    for ckpt in ckpts:
        summary = run_evaluation_over_seeds(
            checkpoint=str(ckpt),
            seeds=seeds,
            hands=hands,
            obs_dim=obs_dim,
            num_actions=num_actions,
        )
        results.append((str(ckpt), summary.mean_payoff, summary.std_payoff, summary.ci95_half_width, summary.num_seeds))

    results.sort(
        key=lambda x: _ranking_value(x[1], x[3], rank_metric=rank_metric, ci_penalty=ci_penalty),
        reverse=True,
    )
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Rank checkpoint pool by average payoff per hand")
    p.add_argument("--checkpoints-dir", type=str, required=True)
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--obs-dim", type=int, default=128)
    p.add_argument("--num-actions", type=int, default=16)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--rank-metric", type=str, choices=["mean", "lcb"], default="mean")
    p.add_argument("--ci-penalty", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    ranking = evaluate_checkpoint_pool(
        checkpoints_dir=a.checkpoints_dir,
        hands=a.hands,
        obs_dim=a.obs_dim,
        num_actions=a.num_actions,
        seed_start=a.seed_start,
        num_seeds=a.num_seeds,
        rank_metric=a.rank_metric,
        ci_penalty=a.ci_penalty,
    )
    print(f"Ranking metric: {a.rank_metric}, ci_penalty={a.ci_penalty}")
    for i, (path, mu, sigma, ci95, n) in enumerate(ranking, 1):
        score = _ranking_value(mu, ci95, rank_metric=a.rank_metric, ci_penalty=a.ci_penalty)
        print(
            f"{i:02d}. {path} -> mean={mu:.6f}, std={sigma:.6f}, "
            f"ci95=±{ci95:.6f}, seeds={n}, rank_score={score:.6f}"
        )
