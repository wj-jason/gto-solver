from __future__ import annotations

try:
    from solver import _poker_solver_cpp as _ext
except ImportError as e:
    raise ImportError(
        "the compiled C++ extension (_poker_solver_cpp) isn't built yet. "
        "Run `pip install -e .` from the repo root after cmake/pybind11 "
        "are available (see environment.yml) to build it."
    ) from e

RANKS = "23456789TJQKA"
SUITS = "shdc"

_RANK_INDEX = {r: i for i, r in enumerate(RANKS)}
_SUIT_INDEX = {s: i for i, s in enumerate(SUITS)}


def card_to_bit(card: str) -> int:
    rank, suit = card[0], card[1]
    r = _RANK_INDEX[rank]
    s = _SUIT_INDEX[suit]
    return 1 << (s * 13 + r)


def cards_to_mask(cards: list[str]) -> int:
    mask = 0
    for c in cards:
        mask |= card_to_bit(c)
    return mask


def score_5card(cards: list[str]) -> int:
    return _ext.score_5card(cards_to_mask(cards))


def best_hand_score(cards: list[str]) -> int:
    return _ext.best_hand_score(cards_to_mask(cards))


def compare(cards_a: list[str], cards_b: list[str]) -> int:
    score_a = best_hand_score(cards_a)
    score_b = best_hand_score(cards_b)
    if score_a > score_b:
        return 1
    if score_a < score_b:
        return -1
    return 0
