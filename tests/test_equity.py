import pytest

from solver.equity import hand_vs_hand_equity

N_SAMPLES = 3000
SEED = 42
TOLERANCE = 0.05  # +/- 5 percentage points, generous for MC noise at this sample size


def _log(label: str, mc_equity: float, expected: float) -> None:
    diff = mc_equity - expected
    print(f"{label:20s}  mc={mc_equity:.4f}  expected={expected:.4f}  " f"diff={diff:+.4f}")


def test_equity_result_win_tie_loss_sum_to_one():
    result = hand_vs_hand_equity("AA", "KK", n_samples=N_SAMPLES, seed=SEED)
    total = result.win + result.tie + result.loss
    print(
        f"AA vs KK  win={result.win:.4f} "
        f"tie={result.tie:.4f} loss={result.loss:.4f} total={total:.4f}"
    )
    assert abs(total - 1.0) < 1e-9


def test_pair_vs_lower_pair_favors_higher_pair():
    expected = 0.82
    result = hand_vs_hand_equity("AA", "KK", n_samples=N_SAMPLES, seed=SEED)
    _log("AA vs KK", result.equity, expected)
    assert result.equity == pytest.approx(expected, abs=TOLERANCE)


def test_equity_is_symmetric():
    ab = hand_vs_hand_equity("AA", "KK", n_samples=N_SAMPLES, seed=SEED)
    ba = hand_vs_hand_equity("KK", "AA", n_samples=N_SAMPLES, seed=SEED)
    print(
        f"AA vs KK equity={ab.equity:.4f} "
        f"KK vs AA equity={ba.equity:.4f}  sum={ab.equity + ba.equity:.4f}"
    )
    assert (ab.equity + ba.equity) == pytest.approx(1.0, abs=TOLERANCE)


def test_identical_hand_label_is_roughly_coinflip():
    expected = 0.5
    # AKs vs AKs (different suits/combos, no card overlap) should be ~50/50
    result = hand_vs_hand_equity("AKs", "AKs", n_samples=N_SAMPLES, seed=SEED)
    _log("AKs vs AKs", result.equity, expected)
    assert result.equity == pytest.approx(expected, abs=TOLERANCE)


def test_dominant_pair_vs_weak_unrelated_hand():
    expected = 0.88
    result = hand_vs_hand_equity("AA", "72o", n_samples=N_SAMPLES, seed=SEED)
    _log("AA vs 72o", result.equity, expected)
    assert result.equity == pytest.approx(expected, abs=TOLERANCE)


def test_close_coinflip_style_matchup():
    expected = 0.46
    # a classic "coinflip" spot: overcards vs. a slightly lower pair
    result = hand_vs_hand_equity("AKs", "QQ", n_samples=N_SAMPLES, seed=SEED)
    _log("AKs vs QQ", result.equity, expected)
    assert result.equity == pytest.approx(expected, abs=TOLERANCE)


def test_seed_reproducibility():
    a = hand_vs_hand_equity("AA", "KK", n_samples=500, seed=123)
    b = hand_vs_hand_equity("AA", "KK", n_samples=500, seed=123)
    print(f"seed=123 run1 equity={a.equity:.4f}  run2 equity={b.equity:.4f}")
    assert a == b


@pytest.mark.parametrize(
    "hand_a,hand_b,expected",
    [
        ("AA", "KK", 0.82),
        ("KK", "QQ", 0.82),
        ("AKs", "QQ", 0.46),
        ("AKo", "QQ", 0.43),
        ("AA", "72o", 0.88),
        ("22", "AKo", 0.53),
        ("JJ", "AKs", 0.54),
    ],
)
def test_known_matchup_reference_table(hand_a, hand_b, expected):
    """A small table of commonly-cited reference equities, all logged
    together so you can eyeball MC accuracy across several spot types at
    once (pair vs pair, overcards vs pair, dominant vs weak)."""
    result = hand_vs_hand_equity(hand_a, hand_b, n_samples=N_SAMPLES, seed=SEED)
    _log(f"{hand_a} vs {hand_b}", result.equity, expected)
    assert result.equity == pytest.approx(expected, abs=TOLERANCE)
