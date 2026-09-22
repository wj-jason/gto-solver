from __future__ import annotations

from collections import Counter
from itertools import combinations

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}

HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8


def _rank_of(card: str) -> int:
    return RANK_VALUE[card[0]]


def _suit_of(card: str) -> str:
    return card[1]


def _straight_high(sorted_ranks_desc: list[int]) -> int | None:
    """Given distinct ranks sorted descending, return the high card of a
    straight if one exists among them, else None. Handles the wheel."""
    ranks = sorted(set(sorted_ranks_desc), reverse=True)
    if 14 in ranks:
        ranks = ranks + [1]  # ace can play low for the wheel
    for i in range(len(ranks) - 4):
        window = ranks[i : i + 5]
        if window[0] - window[4] == 4:
            return window[0]
    return None


def score_5card(cards: list[str]) -> tuple:
    """returns (hand type, *tiebreakers)"""
    assert len(cards) == 5
    ranks = [_rank_of(c) for c in cards]
    suits = [_suit_of(c) for c in cards]
    rank_counts = Counter(ranks)
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(sorted(ranks, reverse=True))

    if straight_high is not None and is_flush:
        return (STRAIGHT_FLUSH, straight_high)

    counts_desc = sorted(rank_counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    if counts_desc[0][1] == 4:
        quad_rank = counts_desc[0][0]
        kicker = max(r for r in ranks if r != quad_rank)
        return (QUADS, quad_rank, kicker)

    if counts_desc[0][1] == 3 and counts_desc[1][1] == 2:
        return (FULL_HOUSE, counts_desc[0][0], counts_desc[1][0])

    if is_flush:
        return (FLUSH, *sorted(ranks, reverse=True))

    if straight_high is not None:
        return (STRAIGHT, straight_high)

    if counts_desc[0][1] == 3:
        trips_rank = counts_desc[0][0]
        kickers = sorted((r for r in ranks if r != trips_rank), reverse=True)
        return (TRIPS, trips_rank, *kickers)

    if counts_desc[0][1] == 2 and counts_desc[1][1] == 2:
        pair_ranks = sorted([counts_desc[0][0], counts_desc[1][0]], reverse=True)
        kicker = max(r for r in ranks if r not in pair_ranks)
        return (TWO_PAIR, *pair_ranks, kicker)

    if counts_desc[0][1] == 2:
        pair_rank = counts_desc[0][0]
        kickers = sorted((r for r in ranks if r != pair_rank), reverse=True)
        return (PAIR, pair_rank, *kickers)

    return (HIGH_CARD, *sorted(ranks, reverse=True))


def best_hand_score(cards: list[str]) -> tuple:
    assert 5 <= len(cards) <= 7
    if len(cards) == 5:
        return score_5card(cards)
    return max(score_5card(list(combo)) for combo in combinations(cards, 5))


def compare(cards_a: list[str], cards_b: list[str]) -> int:
    score_a = best_hand_score(cards_a)
    score_b = best_hand_score(cards_b)
    if score_a > score_b:
        return 1
    if score_a < score_b:
        return -1
    return 0
