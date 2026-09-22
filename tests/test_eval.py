from solver.eval import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    best_hand_score,
    compare,
    score_5card,
)


def test_high_card_category():
    score = score_5card(["2h", "5c", "9d", "Js", "Ac"])
    assert score[0] == HIGH_CARD


def test_pair_category():
    score = score_5card(["2h", "2c", "9d", "Js", "Ac"])
    assert score[0] == PAIR
    assert score[1] == 2  # pair rank


def test_two_pair_category():
    score = score_5card(["2h", "2c", "9d", "9s", "Ac"])
    assert score[0] == TWO_PAIR


def test_trips_category():
    score = score_5card(["2h", "2c", "2d", "9s", "Ac"])
    assert score[0] == TRIPS


def test_straight_category():
    score = score_5card(["5h", "6c", "7d", "8s", "9c"])
    assert score[0] == STRAIGHT
    assert score[1] == 9  # high card of straight


def test_wheel_straight_ace_plays_low():
    score = score_5card(["Ah", "2c", "3d", "4s", "5c"])
    assert score[0] == STRAIGHT
    assert score[1] == 5  # 5-high straight (the wheel), not ace-high


def test_flush_category():
    score = score_5card(["2h", "5h", "9h", "Jh", "Ah"])
    assert score[0] == FLUSH


def test_full_house_category():
    score = score_5card(["2h", "2c", "2d", "9s", "9c"])
    assert score[0] == FULL_HOUSE
    assert score[1] == 2  # trips rank
    assert score[2] == 9  # pair rank


def test_quads_category():
    score = score_5card(["2h", "2c", "2d", "2s", "9c"])
    assert score[0] == QUADS


def test_straight_flush_category():
    score = score_5card(["5h", "6h", "7h", "8h", "9h"])
    assert score[0] == STRAIGHT_FLUSH
    assert score[1] == 9


def test_hand_category_ordering():
    """Sanity check the category ranking itself: each category should beat
    every lower one regardless of kickers."""
    high_card = score_5card(["2h", "5c", "9d", "Js", "Ac"])
    pair = score_5card(["3h", "3c", "9d", "Js", "Ac"])
    two_pair = score_5card(["3h", "3c", "4d", "4s", "Ac"])
    trips = score_5card(["3h", "3c", "3d", "Js", "Ac"])
    straight = score_5card(["3h", "4c", "5d", "6s", "7c"])
    flush = score_5card(["2h", "5h", "9h", "Jh", "Ah"])
    full_house = score_5card(["3h", "3c", "3d", "4s", "4c"])
    quads = score_5card(["3h", "3c", "3d", "3s", "4c"])
    straight_flush = score_5card(["3h", "4h", "5h", "6h", "7h"])

    ordered = [
        high_card,
        pair,
        two_pair,
        trips,
        straight,
        flush,
        full_house,
        quads,
        straight_flush,
    ]
    for weaker, stronger in zip(ordered, ordered[1:]):
        assert weaker < stronger


def test_best_hand_score_picks_best_5_of_7():
    # 7 cards containing a flush among the 5-card subsets
    cards = ["2h", "5h", "9h", "Jh", "Ah", "2c", "3d"]
    score = best_hand_score(cards)
    assert score[0] == FLUSH


def test_best_hand_score_ignores_worse_5_card_subsets():
    # trips on the board plus two useless side cards -- best 5 should be trips, not worse
    cards = ["9h", "9c", "9d", "2s", "3c", "4h", "5d"]
    score = best_hand_score(cards)
    assert score[0] == TRIPS


def test_compare_higher_hand_wins():
    aces = ["Ah", "Ac", "9d", "Js", "2c"]
    kings = ["Kh", "Kc", "9d", "Js", "2c"]
    assert compare(aces, kings) == 1
    assert compare(kings, aces) == -1


def test_compare_tie():
    hand_a = ["2h", "5c", "9d", "Js", "Ac"]
    hand_b = ["2c", "5d", "9h", "Js", "Ad"]  # same ranks, different suits
    assert compare(hand_a, hand_b) == 0
