"""No-Limit Hold'em equity estimation utilities.

This module implements a lightweight 5/7-card hand evaluator and a Monte-Carlo
showdown equity estimator for up to 8 players (hero + 7 opponents).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random
from typing import Iterable, List, Sequence, Tuple

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_TO_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
MAX_PLAYERS = 8

Card = str
Score = Tuple[int, Tuple[int, ...]]


@dataclass(frozen=True)
class EquityResult:
    equity: float
    win_prob: float
    tie_prob: float
    loss_prob: float
    simulations: int


def _validate_card(card: Card) -> None:
    if len(card) != 2 or card[0] not in RANK_TO_VALUE or card[1] not in SUITS:
        raise ValueError(f"Invalid card: {card}")


def _full_deck() -> List[Card]:
    return [r + s for r in RANKS for s in SUITS]


def _straight_high(values: Sequence[int]) -> int | None:
    unique = sorted(set(values), reverse=True)
    if 14 in unique:
        unique.append(1)  # wheel support A-2-3-4-5
    for i in range(len(unique) - 4):
        window = unique[i : i + 5]
        if window[0] - window[4] == 4 and len(set(window)) == 5:
            return window[0]
    return None


def evaluate_five(cards: Sequence[Card]) -> Score:
    """Evaluate 5-card poker hand.

    Returns a comparable tuple: (category, tiebreakers), higher is better.
    Categories:
      8 straight-flush, 7 quads, 6 full house, 5 flush,
      4 straight, 3 trips, 2 two pair, 1 one pair, 0 high card.
    """
    if len(cards) != 5:
        raise ValueError("evaluate_five expects exactly 5 cards")
    for card in cards:
        _validate_card(card)

    ranks = [RANK_TO_VALUE[c[0]] for c in cards]
    suits = [c[1] for c in cards]

    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    counts_sorted = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    is_flush = len(set(suits)) == 1
    straight_hi = _straight_high(ranks)

    if is_flush and straight_hi is not None:
        return (8, (straight_hi,))

    if counts_sorted[0][1] == 4:
        quad = counts_sorted[0][0]
        kicker = max(r for r in ranks if r != quad)
        return (7, (quad, kicker))

    if counts_sorted[0][1] == 3 and counts_sorted[1][1] == 2:
        return (6, (counts_sorted[0][0], counts_sorted[1][0]))

    if is_flush:
        return (5, tuple(sorted(ranks, reverse=True)))

    if straight_hi is not None:
        return (4, (straight_hi,))

    if counts_sorted[0][1] == 3:
        trips = counts_sorted[0][0]
        kickers = sorted((r for r in ranks if r != trips), reverse=True)
        return (3, (trips, *kickers))

    if counts_sorted[0][1] == 2 and counts_sorted[1][1] == 2:
        pair_high = max(counts_sorted[0][0], counts_sorted[1][0])
        pair_low = min(counts_sorted[0][0], counts_sorted[1][0])
        kicker = max(r for r in ranks if r not in (pair_high, pair_low))
        return (2, (pair_high, pair_low, kicker))

    if counts_sorted[0][1] == 2:
        pair = counts_sorted[0][0]
        kickers = sorted((r for r in ranks if r != pair), reverse=True)
        return (1, (pair, *kickers))

    return (0, tuple(sorted(ranks, reverse=True)))


def evaluate_seven(cards: Sequence[Card]) -> Score:
    if len(cards) != 7:
        raise ValueError("evaluate_seven expects exactly 7 cards")
    best: Score | None = None
    for combo in combinations(cards, 5):
        score = evaluate_five(combo)
        if best is None or score > best:
            best = score
    if best is None:
        raise RuntimeError("Unreachable: no 5-card combinations")
    return best


def estimate_equity(
    hole_cards: Sequence[Card],
    board_cards: Sequence[Card],
    num_opponents: int,
    simulations: int = 5000,
    seed: int | None = None,
) -> EquityResult:
    """Monte-Carlo equity for hero hand vs random ranges.

    - Supports up to 8 players total (hero + 7 opponents).
    - board_cards can be 0..5 cards.
    """
    if len(hole_cards) != 2:
        raise ValueError("hole_cards must contain exactly 2 cards")
    if len(board_cards) > 5:
        raise ValueError("board_cards cannot exceed 5 cards")
    if num_opponents < 1 or num_opponents > MAX_PLAYERS - 1:
        raise ValueError("num_opponents must be in [1, 7]")
    if simulations <= 0:
        raise ValueError("simulations must be > 0")

    known = list(hole_cards) + list(board_cards)
    for c in known:
        _validate_card(c)
    if len(set(known)) != len(known):
        raise ValueError("Duplicate cards in known cards")

    rng = random.Random(seed)
    deck = [c for c in _full_deck() if c not in known]

    wins = 0.0
    ties = 0.0

    board_missing = 5 - len(board_cards)

    for _ in range(simulations):
        need = board_missing + 2 * num_opponents
        drawn = rng.sample(deck, need)

        runout = list(board_cards) + drawn[:board_missing]
        opp_cards = drawn[board_missing:]

        hero_score = evaluate_seven(list(hole_cards) + runout)

        opp_scores = []
        for i in range(num_opponents):
            c1 = opp_cards[2 * i]
            c2 = opp_cards[2 * i + 1]
            opp_scores.append(evaluate_seven([c1, c2] + runout))

        best_opp = max(opp_scores)
        if hero_score > best_opp:
            wins += 1.0
        elif hero_score == best_opp:
            n_tied = 1 + sum(1 for s in opp_scores if s == hero_score)
            ties += 1.0 / n_tied

    equity = (wins + ties) / simulations
    win_prob = wins / simulations
    tie_prob = max(0.0, equity - win_prob)
    loss_prob = max(0.0, 1.0 - equity)

    return EquityResult(
        equity=equity,
        win_prob=win_prob,
        tie_prob=tie_prob,
        loss_prob=loss_prob,
        simulations=simulations,
    )
