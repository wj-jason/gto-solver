from __future__ import annotations

from dataclasses import dataclass

try:
    from solver import _poker_solver_cpp as _ext
except ImportError as e:
    raise ImportError(
        "the compiled C++ extension (_poker_solver_cpp) isn't built yet. "
        "Run `pip install -e .` from the repo root to build it."
    ) from e

from solver.eval.eval_cpp import _RANK_INDEX, _SUIT_INDEX
from solver.game.cards import CanonicalHand, expand_to_combos, hand_by_label


@dataclass(frozen=True)
class EquityResult:
    win: float
    tie: float
    loss: float
    n_trials: int  # actual trials collected -- may be < requested n_samples

    @property
    def equity(self) -> float:
        return self.win + 0.5 * self.tie


def _card_to_index(card: str) -> int:
    """0-51 index matching the C++ bit layout: index = suit*13 + rank."""
    rank, suit = card[0], card[1]
    return _SUIT_INDEX[suit] * 13 + _RANK_INDEX[rank]


def _combos_as_indices(hand: CanonicalHand) -> list[tuple[int, int]]:
    return [
        (_card_to_index(c1), _card_to_index(c2)) for c1, c2 in expand_to_combos(hand)
    ]


def hand_vs_hand_equity(
    hand_a: CanonicalHand | str,
    hand_b: CanonicalHand | str,
    n_samples: int = 2000,
    seed: int = 0,
) -> EquityResult:
    """Preflop equity -- same signature/semantics as
    eval.equity.hand_vs_hand_equity, backed entirely by the C++ MC loop."""
    return equity_on_board(hand_a, hand_b, board=[], n_samples=n_samples, seed=seed)


def equity_on_board(
    hand_a: CanonicalHand | str,
    hand_b: CanonicalHand | str,
    board: list[str] | None = None,
    n_samples: int = 2000,
    seed: int = 0,
) -> EquityResult:
    """Street-aware equity -- same signature/semantics as
    eval.equity.equity_on_board, backed entirely by the C++ MC loop.

    NOTE: unlike eval.equity, this doesn't (yet) support pinning one side
    to an exact hole-card combo -- both hand_a/hand_b are always canonical
    hand labels, sampled to a combo internally by the C++ RNG. Add that if
    you need it later (the C++ side already takes explicit combo lists, so
    it's a small change to pass a single-combo list for a pinned hand).
    """
    if isinstance(hand_a, str):
        hand_a = hand_by_label(hand_a)
    if isinstance(hand_b, str):
        hand_b = hand_by_label(hand_b)
    board = board or []

    combos_a = _combos_as_indices(hand_a)
    combos_b = _combos_as_indices(hand_b)
    board_indices = [_card_to_index(c) for c in board]

    counts = _ext.simulate_equity(combos_a, combos_b, board_indices, n_samples, seed)
    n = counts.n_trials
    if n == 0:
        raise RuntimeError(
            f"could not sample any valid matchups for {hand_a} vs {hand_b}"
        )

    return EquityResult(
        win=counts.wins / n, tie=counts.ties / n, loss=counts.losses / n, n_trials=n
    )
