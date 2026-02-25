"""Utilities to map policy outputs and enumerate legal NLHE actions."""

from dataclasses import dataclass
from typing import Iterable, List, Literal

Action = Literal["fold", "check", "call", "raise"]

MIN_POT_FRACTION = 0.01
MAX_POT_FRACTION = 1.0
MAX_PLAYERS = 8


@dataclass(frozen=True)
class ActionDecision:
    action: Action
    amount: int = 0


@dataclass(frozen=True)
class BettingContext:
    legal_actions: Iterable[Action]
    min_raise_to: int
    max_raise_to: int


@dataclass(frozen=True)
class GameConfig:
    max_players: int = MAX_PLAYERS

    def __post_init__(self) -> None:
        if self.max_players < 2 or self.max_players > MAX_PLAYERS:
            raise ValueError(f"max_players must be in [2, {MAX_PLAYERS}]")


def _build_raise_sizes(
    ctx: BettingContext,
    pot: int,
    step: float = 0.01,
    min_fraction: float = MIN_POT_FRACTION,
    max_fraction: float = MAX_POT_FRACTION,
) -> List[int]:
    """Enumerate legal raise sizes from min_fraction..max_fraction of the pot + all-in.

    Returned sizes are unique and sorted ascending.
    """
    if step <= 0:
        raise ValueError("step must be > 0")

    min_fraction = max(MIN_POT_FRACTION, min_fraction)
    max_fraction = min(MAX_POT_FRACTION, max_fraction)

    sizes = set()
    safe_pot = max(pot, 1)

    current = min_fraction
    # +1e-9 avoids float precision issues when reaching the right boundary.
    while current <= max_fraction + 1e-9:
        target = int(round(current * safe_pot))
        amount = min(max(target, ctx.min_raise_to), ctx.max_raise_to)
        if ctx.min_raise_to <= amount <= ctx.max_raise_to:
            sizes.add(amount)
        current += step

    sizes.add(ctx.max_raise_to)  # explicit all-in option
    return sorted(sizes)


def enumerate_legal_decisions(
    ctx: BettingContext,
    pot: int,
    raise_step: float = 0.01,
) -> List[ActionDecision]:
    """Return all legal actions including raise sizes from 0.01x pot to 1.0x pot and all-in."""
    legal = set(ctx.legal_actions)
    decisions: List[ActionDecision] = []

    for action in ("fold", "check", "call"):
        if action in legal:
            decisions.append(ActionDecision(action))

    if "raise" in legal and ctx.min_raise_to <= ctx.max_raise_to:
        for amount in _build_raise_sizes(ctx, pot=pot, step=raise_step):
            decisions.append(ActionDecision("raise", amount))

    if not decisions:
        raise ValueError("No legal action available in context")

    return decisions


def map_discrete_action(
    action_id: int,
    ctx: BettingContext,
    raise_buckets: List[float] | None = None,
    pot: int = 0,
) -> ActionDecision:
    """Map a discrete action index into a legal action.

    action_id:
      0 -> fold
      1 -> check/call (prefers check when legal)
      >=2 -> raise buckets based on `raise_buckets` * pot
    """
    legal = set(ctx.legal_actions)

    if action_id == 0:
        if "fold" in legal:
            return ActionDecision("fold")
        if "check" in legal:
            return ActionDecision("check")
        if "call" in legal:
            return ActionDecision("call")

    if action_id == 1:
        if "check" in legal:
            return ActionDecision("check")
        if "call" in legal:
            return ActionDecision("call")

    if raise_buckets is None:
        raise_buckets = [0.01, 0.1, 0.25, 0.33, 0.5, 0.75, 1.0]

    if "raise" in legal and action_id >= 2:
        bucket_idx = min(action_id - 2, len(raise_buckets) - 1)
        target = int(round(raise_buckets[bucket_idx] * max(pot, 1)))
        amount = min(max(target, ctx.min_raise_to), ctx.max_raise_to)
        return ActionDecision("raise", amount)

    if "check" in legal:
        return ActionDecision("check")
    if "call" in legal:
        return ActionDecision("call")
    if "fold" in legal:
        return ActionDecision("fold")

    raise ValueError("No legal action available in context")
