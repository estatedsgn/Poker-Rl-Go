"""RLCard integration for real No-Limit Hold'em environment collection."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class RLCardConfig:
    game_name: str = "no-limit-holdem"
    seed: int = 42
    obs_dim: int = 128
    num_actions: int = 16


def _flatten_obs(obs) -> List[float]:
    if isinstance(obs, (list, tuple)):
        out: List[float] = []
        for x in obs:
            out.extend(_flatten_obs(x))
        return out
    try:
        return [float(obs)]
    except Exception:
        return [0.0]


def _fit_obs_dim(obs: List[float], obs_dim: int) -> List[float]:
    if len(obs) == obs_dim:
        return obs
    if len(obs) > obs_dim:
        return obs[:obs_dim]
    return obs + [0.0] * (obs_dim - len(obs))


PolicyFn = Callable[[List[float], List[bool], List[int]], int]


class RLCardNLHECollector:
    """Collect transitions from RLCard NLHE for training/evaluation."""

    def __init__(self, cfg: RLCardConfig):
        self.cfg = cfg
        try:
            import rlcard  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "rlcard is required for backend='rlcard'. Install with `pip install rlcard`."
            ) from exc

        self.env = rlcard.make(cfg.game_name, config={"seed": cfg.seed})
        state, _ = self.env.reset()
        self.current_state = state

    def _extract_legal_actions(self, state: Dict) -> List[int]:
        legal = [int(a) for a in state.get("legal_actions", {}).keys()]
        legal = [a for a in legal if 0 <= a < self.cfg.num_actions]
        if not legal:
            legal = [0]
        return legal

    def _extract_legal_mask(self, legal_actions: List[int]) -> List[bool]:
        mask = [False] * self.cfg.num_actions
        for a in legal_actions:
            mask[a] = True
        return mask

    def _choose_action(self, obs: List[float], legal_mask: List[bool], legal_actions: List[int], policy_fn: Optional[PolicyFn]) -> int:
        if policy_fn is None:
            return int(random.choice(legal_actions))
        action = int(policy_fn(obs, legal_mask, legal_actions))
        if action not in legal_actions:
            return int(random.choice(legal_actions))
        return action

    def sample_batch(self, batch_size: int, policy_fn: Optional[PolicyFn] = None) -> Dict[str, List]:
        obs_list: List[List[float]] = []
        action_mask_list: List[List[bool]] = []
        actions_list: List[int] = []
        rewards_list: List[float] = []
        dones_list: List[float] = []

        while len(obs_list) < batch_size:
            state = self.current_state
            obs = _fit_obs_dim(_flatten_obs(state.get("obs", [])), self.cfg.obs_dim)
            legal_actions = self._extract_legal_actions(state)
            action_mask = self._extract_legal_mask(legal_actions)
            action = self._choose_action(obs, action_mask, legal_actions, policy_fn)

            next_state, _ = self.env.step(action)
            done = float(self.env.is_over())
            if done:
                payoffs = self.env.get_payoffs()
                reward = float(payoffs[0]) if len(payoffs) > 0 else 0.0
                reset_state, _ = self.env.reset()
                self.current_state = reset_state
            else:
                reward = 0.0
                self.current_state = next_state

            obs_list.append(obs)
            action_mask_list.append(action_mask)
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)

        return {
            "obs": obs_list,
            "action_mask": action_mask_list,
            "actions": actions_list,
            "rewards": rewards_list,
            "dones": dones_list,
        }

    def evaluate_policy(self, num_hands: int, policy_fn: Optional[PolicyFn] = None) -> float:
        """Return average payoff per finished hand for player-0."""
        if num_hands <= 0:
            raise ValueError("num_hands must be > 0")

        finished = 0
        total_payoff = 0.0

        while finished < num_hands:
            state = self.current_state
            obs = _fit_obs_dim(_flatten_obs(state.get("obs", [])), self.cfg.obs_dim)
            legal_actions = self._extract_legal_actions(state)
            action_mask = self._extract_legal_mask(legal_actions)
            action = self._choose_action(obs, action_mask, legal_actions, policy_fn)

            next_state, _ = self.env.step(action)
            if self.env.is_over():
                payoffs = self.env.get_payoffs()
                total_payoff += float(payoffs[0]) if len(payoffs) > 0 else 0.0
                finished += 1
                state, _ = self.env.reset()
                self.current_state = state
            else:
                self.current_state = next_state

        return total_payoff / num_hands
