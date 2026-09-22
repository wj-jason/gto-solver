from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}
SUITS = "shdc"


@dataclass(frozen=True)
class CanonicalHand:
    """
    Collapse the state of specific suits (AcKd = AKo)
    """

    label: str  # e.g. "AKs", "AKo", "TT"
    high: str  # higher rank, e.g. "A"
    low: str  # lower (or equal) rank, e.g. "K"
    suited: bool | None  # None for pairs, True/False otherwise
    n_combos: int  # 6 for pairs, 4 for suited, 12 for offsuit

    def __str__(self) -> str:
        return self.label


def canonical_hands() -> list[CanonicalHand]:
    hands: list[CanonicalHand] = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i < j:
                continue  # only emit each unordered rank-pair once
            high, low = r1, r2
            if i == j:
                hands.append(CanonicalHand(f"{high}{low}", high, low, None, 6))
            else:
                hands.append(CanonicalHand(f"{high}{low}s", high, low, True, 4))
                hands.append(CanonicalHand(f"{high}{low}o", high, low, False, 12))
    return hands


def total_combos(hands: list[CanonicalHand] | None = None) -> int:
    hands = hands if hands is not None else canonical_hands()
    return sum(h.n_combos for h in hands)


def expand_to_combos(hand: CanonicalHand) -> list[tuple[str, str]]:
    """Expand a canonical hand into its actual (card, card) combinations."""
    combos: list[tuple[str, str]] = []
    if hand.suited is None:  # pair
        for s1, s2 in combinations(SUITS, 2):
            combos.append((hand.high + s1, hand.low + s2))
    elif hand.suited:
        for s in SUITS:
            combos.append((hand.high + s, hand.low + s))
    else:
        for s1, s2 in product(SUITS, SUITS):
            if s1 != s2:
                combos.append((hand.high + s1, hand.low + s2))
    return combos


def hands_conflict(combo_a: tuple[str, str], combo_b: tuple[str, str]) -> bool:
    return bool(set(combo_a) & set(combo_b))


def hand_by_label(label: str) -> CanonicalHand:
    for h in canonical_hands():
        if h.label == label:
            return h
    raise KeyError(f"unknown canonical hand label: {label!r}")
