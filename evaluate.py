"""Evaluation entrypoint for NLHE coursework scaffold."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

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

    @staticmethod
    def from_dict(data: dict) -> "EvalConfig":
        return EvalConfig(**data)


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
    )


if __name__ == "__main__":
    cfg = parse_args()
    avg_payoff = run_evaluation(cfg)
    print(f"Average payoff per hand (player-0): {avg_payoff:.6f}")
