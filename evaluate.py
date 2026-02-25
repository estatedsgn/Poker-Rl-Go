"""Evaluation entrypoint for NLHE coursework scaffold."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Iterable, List

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from src.rlcard_nlhe_env import RLCardConfig, RLCardNLHECollector
from train import TinyActorCritic, choose_action_from_logits


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: str | None = None
    obs_dim: int = 128
    num_actions: int = 16
    hands: int = 200
    seed: int = 42
    seed_start: int = 42
    num_seeds: int = 1

    @staticmethod
    def from_dict(data: dict) -> "EvalConfig":
        return EvalConfig(**data)


@dataclass(frozen=True)
class EvalSummary:
    mean_payoff: float
    std_payoff: float
    ci95_half_width: float
    num_seeds: int


def summarize_scores(scores: Iterable[float]) -> EvalSummary:
    values = list(scores)
    if not values:
        raise ValueError("scores cannot be empty")

    mu = float(mean(values))
    if len(values) == 1:
        return EvalSummary(mean_payoff=mu, std_payoff=0.0, ci95_half_width=0.0, num_seeds=1)

    sigma = float(stdev(values))
    ci95 = 1.96 * sigma / math.sqrt(len(values))
    return EvalSummary(mean_payoff=mu, std_payoff=sigma, ci95_half_width=ci95, num_seeds=len(values))


def build_policy_fn(model: TinyActorCritic):
    def _policy(obs, _mask, legal_actions):
        obs_t = torch.tensor([obs], dtype=torch.float32)
        with torch.no_grad():
            logits, _ = model(obs_t)
        return choose_action_from_logits(logits[0], legal_actions)

    return _policy


def run_evaluation(cfg: EvalConfig) -> float:
    collector = RLCardNLHECollector(
        RLCardConfig(seed=cfg.seed, obs_dim=cfg.obs_dim, num_actions=cfg.num_actions)
    )

    if cfg.checkpoint is None:
        return collector.evaluate_policy(num_hands=cfg.hands, policy_fn=None)

    if torch is None:
        raise ImportError("PyTorch is required for checkpoint evaluation")

    model = TinyActorCritic(obs_dim=cfg.obs_dim, num_actions=cfg.num_actions)
    state = torch.load(cfg.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return collector.evaluate_policy(num_hands=cfg.hands, policy_fn=build_policy_fn(model))


def run_evaluation_over_seeds(
    checkpoint: str | None,
    seeds: List[int],
    hands: int = 200,
    obs_dim: int = 128,
    num_actions: int = 16,
) -> EvalSummary:
    if not seeds:
        raise ValueError("seeds cannot be empty")

    scores = []
    for s in seeds:
        cfg = EvalConfig(
            checkpoint=checkpoint,
            obs_dim=obs_dim,
            num_actions=num_actions,
            hands=hands,
            seed=s,
        )
        scores.append(run_evaluation(cfg))
    return summarize_scores(scores)


def load_eval_config(path: str) -> EvalConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "eval" in payload:
        payload = payload["eval"]
    return EvalConfig.from_dict(payload)


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate policy in RLCard no-limit-holdem")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--obs-dim", type=int, default=128)
    p.add_argument("--num-actions", type=int, default=16)
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--config", type=str, default=None)
    a = p.parse_args()

    if a.config:
        return load_eval_config(a.config)

    return EvalConfig(
        checkpoint=a.checkpoint,
        obs_dim=a.obs_dim,
        num_actions=a.num_actions,
        hands=a.hands,
        seed=a.seed,
        seed_start=a.seed_start,
        num_seeds=a.num_seeds,
    )


def format_evaluation_output(cfg: EvalConfig) -> str:
    if cfg.num_seeds > 1:
        seeds = [cfg.seed_start + i for i in range(cfg.num_seeds)]
        summary = run_evaluation_over_seeds(
            checkpoint=cfg.checkpoint,
            seeds=seeds,
            hands=cfg.hands,
            obs_dim=cfg.obs_dim,
            num_actions=cfg.num_actions,
        )
        return (
            "Average payoff per hand (player-0): "
            f"mean={summary.mean_payoff:.6f}, std={summary.std_payoff:.6f}, "
            f"95%CI=±{summary.ci95_half_width:.6f}, n={summary.num_seeds}"
        )

    avg_payoff = run_evaluation(cfg)
    return f"Average payoff per hand (player-0): {avg_payoff:.6f}"


if __name__ == "__main__":
    cfg = parse_args()
    try:
        print(format_evaluation_output(cfg))
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
