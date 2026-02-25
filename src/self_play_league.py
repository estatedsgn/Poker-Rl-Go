"""Self-play league utilities for stronger NLHE training.

This module implements lightweight but practical league mechanics:
- policy snapshots,
- Elo tracking,
- prioritized fictitious self-play (PFSP)-style opponent sampling,
- promotion logic for best checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class PolicySnapshot:
    snapshot_id: str
    path: str
    step: int
    winrate_hint: float = 0.0


@dataclass
class LeagueEntry:
    snapshot: PolicySnapshot
    elo: float = 1500.0
    games_played: int = 0


@dataclass
class MatchResult:
    a_id: str
    b_id: str
    # 1.0 if a wins, 0.5 split/tie, 0.0 if b wins
    score_a: float


@dataclass
class SelfPlayLeague:
    entries: Dict[str, LeagueEntry] = field(default_factory=dict)
    active_id: str | None = None

    def add_snapshot(self, snapshot: PolicySnapshot, elo: float = 1500.0, set_active: bool = False) -> None:
        self.entries[snapshot.snapshot_id] = LeagueEntry(snapshot=snapshot, elo=elo)
        if set_active or self.active_id is None:
            self.active_id = snapshot.snapshot_id

    def get_active(self) -> LeagueEntry:
        if self.active_id is None or self.active_id not in self.entries:
            raise ValueError("No active policy in league")
        return self.entries[self.active_id]

    def sorted_entries(self) -> List[LeagueEntry]:
        return sorted(self.entries.values(), key=lambda e: e.elo, reverse=True)

    def top_k_ids(self, k: int = 5) -> List[str]:
        return [e.snapshot.snapshot_id for e in self.sorted_entries()[: max(1, k)]]

    def promote_if_strong(self, candidate: PolicySnapshot, candidate_elo: float, margin: float = 35.0) -> bool:
        """Promote candidate to active if it beats current active Elo by margin."""
        active = self.get_active()
        self.add_snapshot(candidate, elo=candidate_elo)
        if candidate_elo >= active.elo + margin:
            self.active_id = candidate.snapshot_id
            return True
        return False

    def update_elo_batch(self, results: Sequence[MatchResult], k_factor: float = 20.0) -> None:
        for r in results:
            if r.a_id not in self.entries or r.b_id not in self.entries:
                continue
            a = self.entries[r.a_id]
            b = self.entries[r.b_id]

            expected_a = 1.0 / (1.0 + 10 ** ((b.elo - a.elo) / 400.0))
            expected_b = 1.0 - expected_a

            a.elo += k_factor * (r.score_a - expected_a)
            b.elo += k_factor * ((1.0 - r.score_a) - expected_b)
            a.games_played += 1
            b.games_played += 1

    def sample_opponent_pfsp(
        self,
        anchor_id: str,
        rng: random.Random | None = None,
        temperature: float = 1.0,
        min_games: int = 0,
    ) -> str:
        """Sample an opponent with PFSP-like weighting.

        Weight heuristic targets informative opponents:
        - higher for close Elo matches,
        - slightly boosted for opponents with fewer games (exploration).
        """
        if anchor_id not in self.entries:
            raise ValueError(f"Unknown anchor_id: {anchor_id}")
        pool = [e for sid, e in self.entries.items() if sid != anchor_id and e.games_played >= min_games]
        if not pool:
            raise ValueError("No opponents available for sampling")

        anchor = self.entries[anchor_id]
        weights = []
        for e in pool:
            delta = abs(anchor.elo - e.elo)
            closeness = math.exp(-delta / max(1e-6, 200.0 * temperature))
            exploration = 1.0 / math.sqrt(1.0 + e.games_played)
            w = 0.8 * closeness + 0.2 * exploration
            weights.append(max(w, 1e-8))

        rng = rng or random.Random()
        idx = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        return pool[idx].snapshot.snapshot_id


def create_initial_league(checkpoint_paths: Sequence[str]) -> SelfPlayLeague:
    """Create a seed league from checkpoint paths."""
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths cannot be empty")
    league = SelfPlayLeague()
    for i, path in enumerate(checkpoint_paths):
        sid = f"seed_{i:03d}"
        league.add_snapshot(PolicySnapshot(snapshot_id=sid, path=path, step=0), set_active=(i == 0))
    return league
