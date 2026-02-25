"""Competitive NLHE environment configuration helpers.

This module does not implement the full game engine. It defines a stricter, closer-to-
competitive configuration surface that can be consumed by a real simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

MAX_PLAYERS = 8


@dataclass(frozen=True)
class BlindsConfig:
    small_blind: int
    big_blind: int
    ante: int = 0

    def __post_init__(self) -> None:
        if self.small_blind <= 0 or self.big_blind <= 0:
            raise ValueError("Blinds must be positive")
        if self.small_blind >= self.big_blind:
            raise ValueError("small_blind must be < big_blind")
        if self.ante < 0:
            raise ValueError("ante must be >= 0")


@dataclass(frozen=True)
class RakeConfig:
    enabled: bool = False
    percent: float = 0.0
    cap_bb: float = 0.0

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.percent < 0 or self.percent > 0.1:
            raise ValueError("rake percent must be in [0, 0.1]")
        if self.cap_bb < 0:
            raise ValueError("rake cap_bb must be >= 0")


@dataclass(frozen=True)
class ActionAbstraction:
    pot_fractions: Tuple[float, ...] = (
        0.01,
        0.05,
        0.10,
        0.20,
        0.33,
        0.50,
        0.66,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
    )
    include_all_in: bool = True

    def __post_init__(self) -> None:
        if len(self.pot_fractions) == 0:
            raise ValueError("pot_fractions cannot be empty")
        if any(x <= 0 for x in self.pot_fractions):
            raise ValueError("pot fractions must be > 0")


@dataclass(frozen=True)
class CompetitiveEnvConfig:
    players: int = 6
    initial_stack_bb: int = 100
    blinds: BlindsConfig = field(default_factory=lambda: BlindsConfig(50, 100, 0))
    rake: RakeConfig = field(default_factory=RakeConfig)
    abstraction: ActionAbstraction = field(default_factory=ActionAbstraction)
    history_length: int = 16

    def __post_init__(self) -> None:
        if self.players < 2 or self.players > MAX_PLAYERS:
            raise ValueError(f"players must be in [2, {MAX_PLAYERS}]")
        if self.initial_stack_bb < 20:
            raise ValueError("initial_stack_bb should be >= 20 for stable training")
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")


def build_raise_ladder(pot: int, min_raise_to: int, max_raise_to: int, abstraction: ActionAbstraction) -> List[int]:
    """Create legal raise targets using pot-fraction abstraction (+ optional all-in)."""
    if min_raise_to > max_raise_to:
        return []

    safe_pot = max(1, pot)
    amounts = {
        min(max(int(round(frac * safe_pot)), min_raise_to), max_raise_to)
        for frac in abstraction.pot_fractions
    }
    if abstraction.include_all_in:
        amounts.add(max_raise_to)
    return sorted(amounts)


def estimate_rake(pot: int, big_blind: int, rake: RakeConfig) -> int:
    """Estimate rake in chips according to configuration."""
    if not rake.enabled:
        return 0
    raw = int(round(pot * rake.percent))
    cap_chips = int(round(rake.cap_bb * big_blind))
    return min(raw, cap_chips)
