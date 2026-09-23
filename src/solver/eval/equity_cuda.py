from __future__ import annotations

from solver.eval.equity_cpp import EquityResult, _combos_as_indices, _card_to_index
from solver.game.cards import CanonicalHand, hand_by_label

try:
    from solver import _poker_solver_cpp as _ext
except ImportError as e:
    raise ImportError(
        "the compiled C++ extension (_poker_solver_cpp) isn't built yet. "
        "Run `pip install -e .` from the repo root to build it."
    ) from e

if not hasattr(_ext, "simulate_equity_cuda"):
    raise ImportError(
        "_poker_solver_cpp was built WITHOUT CUDA support (no "
        "simulate_equity_cuda symbol). Rebuild with CUDA enabled, e.g.:\n"
        "  pip install -e . --config-settings=cmake.define.POKER_SOLVER_ENABLE_CUDA=ON\n"
        "(requires a CUDA toolchain and GPU on this machine)."
    )


def hand_vs_hand_equity_cuda(
    hand_a: CanonicalHand | str,
    hand_b: CanonicalHand | str,
    n_samples: int = 2000,
    seed: int = 0,
) -> EquityResult:
    """Preflop equity, single hand pair, naive CUDA kernel."""
    return equity_on_board_cuda(
        hand_a, hand_b, board=[], n_samples=n_samples, seed=seed
    )


def equity_on_board_cuda(
    hand_a: CanonicalHand | str,
    hand_b: CanonicalHand | str,
    board: list[str] | None = None,
    n_samples: int = 2000,
    seed: int = 0,
) -> EquityResult:
    if isinstance(hand_a, str):
        hand_a = hand_by_label(hand_a)
    if isinstance(hand_b, str):
        hand_b = hand_by_label(hand_b)
    board = board or []

    combos_a = _combos_as_indices(hand_a)
    combos_b = _combos_as_indices(hand_b)
    board_indices = [_card_to_index(c) for c in board]

    counts = _ext.simulate_equity_cuda(
        combos_a, combos_b, board_indices, n_samples, seed
    )
    n = counts.n_trials
    if n == 0:
        raise RuntimeError(
            f"could not sample any valid matchups for {hand_a} vs {hand_b}"
        )

    return EquityResult(
        win=counts.wins / n, tie=counts.ties / n, loss=counts.losses / n, n_trials=n
    )
