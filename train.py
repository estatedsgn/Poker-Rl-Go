"""Minimal PPO-style training entrypoint for NLHE coursework scaffold.

Supports:
- synthetic backend (default),
- RLCard backend (real no-limit-holdem environment collection).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from src.rl_training_utils import backpropagation_step, total_actor_critic_loss
from src.self_play_league import MatchResult, PolicySnapshot, SelfPlayLeague


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    batch_size: int = 128
    obs_dim: int = 128
    num_actions: int = 16
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip_norm: float = 1.0
    checkpoint_interval: int = 50
    out_dir: str = "checkpoints"
    seed: int = 42
    backend: str = "synthetic"  # synthetic | rlcard
    gae_gamma: float = 0.99
    gae_lambda: float = 0.95

    @staticmethod
    def from_dict(data: Dict) -> "TrainConfig":
        return TrainConfig(**data)


if torch is not None:
    _BaseModule = torch.nn.Module
else:  # pragma: no cover
    _BaseModule = object


class TinyActorCritic(_BaseModule):  # type: ignore[misc]
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.policy = torch.nn.Linear(256, num_actions)
        self.value = torch.nn.Linear(256, 1)

    def forward(self, obs):
        h = self.backbone(obs)
        return self.policy(h), self.value(h)


def choose_action_from_logits(logits: "torch.Tensor", legal_actions: List[int]) -> int:
    if torch is None:
        raise ImportError("PyTorch is required")
    if not legal_actions:
        return 0

    mask = torch.full_like(logits, float("-inf"))
    mask[legal_actions] = 0.0
    masked = logits + mask
    probs = torch.softmax(masked, dim=-1)
    action = int(torch.argmax(probs).item())
    if action not in legal_actions:
        return int(legal_actions[0])
    return action


def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """Compute GAE-Lambda advantages and returns."""
    adv = torch.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + values
    return adv, returns


class SyntheticNLHEBatchGenerator:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def sample(self) -> Dict[str, "torch.Tensor"]:
        obs = torch.randn(self.cfg.batch_size, self.cfg.obs_dim)

        action_mask = torch.zeros(self.cfg.batch_size, self.cfg.num_actions, dtype=torch.bool)
        for i in range(self.cfg.batch_size):
            legal_count = torch.randint(low=2, high=self.cfg.num_actions + 1, size=(1,)).item()
            idx = torch.randperm(self.cfg.num_actions)[:legal_count]
            action_mask[i, idx] = True

        actions = torch.zeros(self.cfg.batch_size, dtype=torch.long)
        for i in range(self.cfg.batch_size):
            legal = torch.nonzero(action_mask[i], as_tuple=False).squeeze(-1)
            actions[i] = legal[torch.randint(low=0, high=len(legal), size=(1,)).item()]

        rewards = torch.randn(self.cfg.batch_size)
        dones = (torch.rand(self.cfg.batch_size) < 0.1).float()

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "action_mask": action_mask,
        }


class RLCardBatchGenerator:
    """Collector backed by real RLCard NLHE environment with policy-driven actions."""

    def __init__(self, cfg: TrainConfig, model: TinyActorCritic):
        from src.rlcard_nlhe_env import RLCardConfig, RLCardNLHECollector

        self.batch_size = cfg.batch_size
        self.model = model
        self.collector = RLCardNLHECollector(
            RLCardConfig(seed=cfg.seed, obs_dim=cfg.obs_dim, num_actions=cfg.num_actions)
        )

    def _policy_fn(self, obs: List[float], _legal_mask: List[bool], legal_actions: List[int]) -> int:
        obs_t = torch.tensor([obs], dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.model(obs_t)
        return choose_action_from_logits(logits[0], legal_actions)

    def sample(self) -> Dict[str, "torch.Tensor"]:
        batch = self.collector.sample_batch(batch_size=self.batch_size, policy_fn=self._policy_fn)
        return {
            "obs": torch.tensor(batch["obs"], dtype=torch.float32),
            "actions": torch.tensor(batch["actions"], dtype=torch.long),
            "rewards": torch.tensor(batch["rewards"], dtype=torch.float32),
            "dones": torch.tensor(batch["dones"], dtype=torch.float32),
            "action_mask": torch.tensor(batch["action_mask"], dtype=torch.bool),
        }


def _save_checkpoint(model: TinyActorCritic, out_dir: Path, step: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"policy_step_{step:06d}.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def run_training(cfg: TrainConfig) -> Dict[str, float]:
    if torch is None:
        raise ImportError("PyTorch is required to run train.py")

    torch.manual_seed(cfg.seed)

    model = TinyActorCritic(obs_dim=cfg.obs_dim, num_actions=cfg.num_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.backend == "rlcard":
        batch_gen = RLCardBatchGenerator(cfg, model)
    else:
        batch_gen = SyntheticNLHEBatchGenerator(cfg)

    league = SelfPlayLeague()
    bootstrap = PolicySnapshot(snapshot_id="bootstrap", path="bootstrap.pt", step=0)
    league.add_snapshot(bootstrap, set_active=True)

    latest_metrics: Dict[str, float] = {}
    checkpoints: List[Path] = []

    for step in range(1, cfg.steps + 1):
        batch = batch_gen.sample()

        logits, values = model(batch["obs"])
        baseline_values = values.squeeze(-1).detach()
        advantages, returns = compute_gae(
            rewards=batch["rewards"],
            values=baseline_values,
            dones=batch["dones"],
            gamma=cfg.gae_gamma,
            lam=cfg.gae_lambda,
        )

        adv_std = advantages.std().clamp(min=1e-6)
        advantages = (advantages - advantages.mean()) / adv_std

        loss, parts = total_actor_critic_loss(
            logits=logits,
            actions=batch["actions"],
            advantages=advantages,
            values=values,
            returns=returns,
            action_mask=batch["action_mask"],
            value_coef=cfg.value_coef,
            entropy_coef=cfg.entropy_coef,
        )
        train_metrics = backpropagation_step(model, optimizer, loss, grad_clip_norm=cfg.grad_clip_norm)

        latest_metrics = {
            "step": float(step),
            "loss": train_metrics["loss"],
            "grad_norm": train_metrics["grad_norm"],
            "policy_loss": float(parts["policy_loss"].detach().item()),
            "value_loss": float(parts["value_loss"].detach().item()),
            "entropy": float(parts["entropy"].detach().item()),
        }

        if step % cfg.checkpoint_interval == 0 or step == cfg.steps:
            ckpt_path = _save_checkpoint(model, Path(cfg.out_dir), step)
            checkpoints.append(ckpt_path)

            sid = f"step_{step:06d}"
            league.add_snapshot(PolicySnapshot(snapshot_id=sid, path=str(ckpt_path), step=step), elo=1500.0)
            league.update_elo_batch([MatchResult(a_id=sid, b_id="bootstrap", score_a=0.55)], k_factor=8.0)

    latest_metrics["num_checkpoints"] = float(len(checkpoints))
    latest_metrics["league_size"] = float(len(league.entries))
    return latest_metrics


def load_train_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "train" in payload:
        payload = payload["train"]
    return TrainConfig.from_dict(payload)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Run minimal masked actor-critic training scaffold")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--obs-dim", type=int, default=128)
    p.add_argument("--num-actions", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--checkpoint-interval", type=int, default=50)
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", type=str, choices=["synthetic", "rlcard"], default="synthetic")
    p.add_argument("--gae-gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    if args.config:
        return load_train_config(args.config)

    return TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        obs_dim=args.obs_dim,
        num_actions=args.num_actions,
        lr=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        out_dir=args.out_dir,
        seed=args.seed,
        backend=args.backend,
        gae_gamma=args.gae_gamma,
        gae_lambda=args.gae_lambda,
    )


if __name__ == "__main__":
    cfg = parse_args()
    metrics = run_training(cfg)
    print("Training finished:", {k: round(v, 6) for k, v in metrics.items()})
