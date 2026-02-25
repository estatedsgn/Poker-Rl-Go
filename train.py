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
    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 2
    mini_batch_size: int = 64

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


def masked_log_prob_actions(logits: "torch.Tensor", actions: "torch.Tensor", action_mask: "torch.Tensor") -> "torch.Tensor":
    """Return log-probabilities of selected actions under masked logits."""
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(action_mask.bool(), logits, neg_inf)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def ppo_policy_loss(
    new_log_probs: "torch.Tensor",
    old_log_probs: "torch.Tensor",
    advantages: "torch.Tensor",
    clip_ratio: float,
) -> "torch.Tensor":
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()


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


def _iter_minibatches(total_size: int, mini_batch_size: int):
    perm = torch.randperm(total_size)
    for start in range(0, total_size, mini_batch_size):
        yield perm[start : start + mini_batch_size]


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

        with torch.no_grad():
            old_logits, old_values = model(batch["obs"])
            old_log_probs = masked_log_prob_actions(old_logits, batch["actions"], batch["action_mask"])
            advantages, returns = compute_gae(
                rewards=batch["rewards"],
                values=old_values.squeeze(-1),
                dones=batch["dones"],
                gamma=cfg.gae_gamma,
                lam=cfg.gae_lambda,
            )
            adv_std = advantages.std().clamp(min=1e-6)
            advantages = (advantages - advantages.mean()) / adv_std

        last_parts = None
        last_train_metrics = None
        for _ in range(cfg.ppo_epochs):
            for mb_idx in _iter_minibatches(batch["obs"].shape[0], cfg.mini_batch_size):
                mb_obs = batch["obs"][mb_idx]
                mb_actions = batch["actions"][mb_idx]
                mb_mask = batch["action_mask"][mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]

                logits, values = model(mb_obs)
                new_log_probs = masked_log_prob_actions(logits, mb_actions, mb_mask)
                ppo_pol = ppo_policy_loss(new_log_probs, mb_old_lp, mb_adv, cfg.ppo_clip_ratio)

                _, parts = total_actor_critic_loss(
                    logits=logits,
                    actions=mb_actions,
                    advantages=mb_adv,
                    values=values,
                    returns=mb_ret,
                    action_mask=mb_mask,
                    value_coef=cfg.value_coef,
                    entropy_coef=cfg.entropy_coef,
                )

                total_loss = ppo_pol + cfg.value_coef * parts["value_loss"] - cfg.entropy_coef * parts["entropy"]
                train_metrics = backpropagation_step(model, optimizer, total_loss, grad_clip_norm=cfg.grad_clip_norm)
                last_parts = parts
                last_train_metrics = train_metrics

        assert last_parts is not None and last_train_metrics is not None
        latest_metrics = {
            "step": float(step),
            "loss": last_train_metrics["loss"],
            "grad_norm": last_train_metrics["grad_norm"],
            "policy_loss": float(ppo_pol.detach().item()),
            "value_loss": float(last_parts["value_loss"].detach().item()),
            "entropy": float(last_parts["entropy"].detach().item()),
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
    p.add_argument("--ppo-clip-ratio", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--mini-batch-size", type=int, default=64)
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
        ppo_clip_ratio=args.ppo_clip_ratio,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
    )


if __name__ == "__main__":
    cfg = parse_args()
    metrics = run_training(cfg)
    print("Training finished:", {k: round(v, 6) for k, v in metrics.items()})
