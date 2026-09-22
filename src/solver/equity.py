"""Preflop hand-vs-hand equity for the 169 canonical hands"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

from solver.eval import compare
from solver.cards import (
    RANKS,
    SUITS,
    CanonicalHand,
    canonical_hands,
    expand_to_combos,
    hand_by_label,
)

FULL_DECK = [r + s for r in RANKS for s in SUITS]


@dataclass(frozen=True)
class EquityResult:
    win: float
    tie: float
    loss: float

    @property
    def equity(self) -> float:
        return self.win + 0.5 * self.tie


def _random_combo(hand: CanonicalHand, rng: random.Random) -> tuple[str, str]:
    combos = expand_to_combos(hand)
    return combos[rng.randrange(len(combos))]


def hand_vs_hand_equity(
    hand_a: CanonicalHand | str,
    hand_b: CanonicalHand | str,
    n_samples: int = 2000,
    seed: int = 0,
) -> EquityResult:
    """Monte Carlo equity of hand_a vs hand_b,
    averaged over card-removal-respecting combos and random boards.
    """
    if isinstance(hand_a, str):
        hand_a = hand_by_label(hand_a)
    if isinstance(hand_b, str):
        hand_b = hand_by_label(hand_b)

    rng = random.Random(seed)
    wins = ties = losses = 0
    n = 0
    attempts = 0
    max_attempts = n_samples * 20  # guard against pathological conflict rates

    while n < n_samples and attempts < max_attempts:
        attempts += 1
        combo_a = _random_combo(hand_a, rng)
        combo_b = _random_combo(hand_b, rng)
        if set(combo_a) & set(combo_b):
            continue  # card removal conflict, resample

        remaining = [c for c in FULL_DECK if c not in combo_a and c not in combo_b]
        board = rng.sample(remaining, 5)

        result = compare(list(combo_a) + board, list(combo_b) + board)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            ties += 1
        n += 1

    if n == 0:
        raise RuntimeError(f"could not sample any valid matchups for {hand_a} vs {hand_b}")

    return EquityResult(win=wins / n, tie=ties / n, loss=losses / n)


@lru_cache(maxsize=None)
def _cached_equity(label_a: str, label_b: str, n_samples: int, seed: int) -> EquityResult:
    return hand_vs_hand_equity(label_a, label_b, n_samples=n_samples, seed=seed)


def equity(label_a: str, label_b: str, n_samples: int = 2000, seed: int = 0) -> float:
    """Cached convenience wrapper returning just the equity of a vs b."""
    return _cached_equity(label_a, label_b, n_samples, seed).equity


def build_equity_table(n_samples: int = 2000, seed: int = 0) -> dict[tuple[str, str], EquityResult]:
    """Full 169x169 table (as a dict keyed by (label_a, label_b)).

    Symmetric: table[(a, b)].equity == 1 - table[(b, a)].equity
    (up to MC noise), so we compute the upper triangle and mirror it.
    """
    hands = canonical_hands()
    table: dict[tuple[str, str], EquityResult] = {}
    for i, ha in enumerate(hands):
        for hb in hands[i:]:
            result = hand_vs_hand_equity(ha, hb, n_samples=n_samples, seed=seed)
            table[(ha.label, hb.label)] = result
            table[(hb.label, ha.label)] = EquityResult(
                win=result.loss, tie=result.tie, loss=result.win
            )
    return table
