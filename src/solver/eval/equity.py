from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from tqdm import tqdm

from solver.eval.eval import compare
from solver.eval.equity_cpp import hand_vs_hand_equity as hand_vs_hand_equity_cpp
from solver.game.cards import (
    RANKS,
    SUITS,
    CanonicalHand,
    canonical_hands,
    expand_to_combos,
    hand_by_label,
)

FULL_DECK = [r + s for r in RANKS for s in SUITS]

HoleSpec = "CanonicalHand | str | tuple[str, str]"  # doc-only alias, see resolve_hole()


class Street(Enum):
    PREFLOP = 0
    FLOP = 3
    TURN = 4
    RIVER = 5


def _validate_board(board: list[str]) -> Street:
    n = len(board)
    for street in Street:
        if street.value == n:
            if len(set(board)) != n:
                raise ValueError(f"board contains duplicate cards: {board}")
            return street
    raise ValueError(
        f"invalid board length {n} (must be 0=preflop, 3=flop, 4=turn, 5=river): {board}"
    )


@dataclass(frozen=True)
class EquityResult:
    win: float
    tie: float
    loss: float

    @property
    def equity(self) -> float:
        """Equity = win probability + half of tie probability."""
        return self.win + 0.5 * self.tie


def _resolve_hole(hole: HoleSpec) -> CanonicalHand | tuple[str, str]:
    """Normalize a hole-card spec: an explicit (card, card) combo is used
    as-is (fixed, exact); a canonical hand (label or CanonicalHand) is kept
    unresolved so a fresh combo gets sampled per Monte Carlo trial."""
    if isinstance(hole, tuple):
        return hole
    if isinstance(hole, str):
        return hand_by_label(hole)
    return hole  # already a CanonicalHand


def _sample_combo(spec: CanonicalHand | tuple[str, str], rng: random.Random) -> tuple[str, str]:
    if isinstance(spec, tuple):
        return spec
    combos = expand_to_combos(spec)
    return combos[rng.randrange(len(combos))]


def equity_on_board(
    hole_a: HoleSpec,
    hole_b: HoleSpec,
    board: list[str] | None = None,
    n_samples: int = 2000,
    seed: int = 0,
) -> EquityResult:
    """Equity of hole_a vs hole_b given a (possibly partial) board.

    hole_a / hole_b can each be:
      - an explicit 2-card combo, e.g. ("As", "Kd")
      - a canonical hand label, e.g. "AKs"

    board (optional) is a list of 0, 3, 4, or 5 card strings
    (preflop, flop, turn, river respectively)
    """
    board = board or []
    _validate_board(board)
    board_set = set(board)
    remaining_needed = 5 - len(board)

    rng = random.Random(seed)
    hole_a_spec = _resolve_hole(hole_a)
    hole_b_spec = _resolve_hole(hole_b)

    wins = ties = losses = 0
    n = 0
    attempts = 0
    max_attempts = n_samples * 30  # dont really need this

    while n < n_samples and attempts < max_attempts:
        attempts += 1
        combo_a = _sample_combo(hole_a_spec, rng)
        combo_b = _sample_combo(hole_b_spec, rng)

        used = set(combo_a) | set(combo_b) | board_set
        if len(used) != 4 + len(board):
            continue  # invalid, resample

        if remaining_needed > 0:
            deck = [c for c in FULL_DECK if c not in used]
            runout = rng.sample(deck, remaining_needed)
        else:
            runout = []

        full_board = board + runout
        result = compare(list(combo_a) + full_board, list(combo_b) + full_board)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            ties += 1
        n += 1

    if n == 0:
        raise RuntimeError(
            f"could not sample any valid matchups for {hole_a} vs {hole_b} on board {board}"
        )

    return EquityResult(win=wins / n, tie=ties / n, loss=losses / n)


def hand_vs_hand_equity(
    hand_a: CanonicalHand | str,
    hand_b: CanonicalHand | str,
    n_samples: int = 2000,
    seed: int = 0,
) -> EquityResult:
    return equity_on_board(hand_a, hand_b, board=[], n_samples=n_samples, seed=seed)


@lru_cache(maxsize=None)
def _cached_equity(label_a: str, label_b: str, n_samples: int, seed: int) -> EquityResult:
    return hand_vs_hand_equity(label_a, label_b, n_samples=n_samples, seed=seed)


def equity(label_a: str, label_b: str, n_samples: int = 2000, seed: int = 0) -> float:
    """Cached convenience wrapper returning just the preflop equity of a vs b."""
    return _cached_equity(label_a, label_b, n_samples, seed).equity


def build_equity_table(
    n_samples: int = 2000,
    seed: int = 0,
    hands: list[CanonicalHand] | None = None,
    show_progress: bool = False,
) -> dict[tuple[str, str], EquityResult]:
    """Preflop equity table (as a dict keyed by (label_a, label_b)).

    Symmetric: table[(a, b)].equity == 1 - table[(b, a)].equity

    `hands` lets you restrict to a subset (e.g. for a fast smoke test);
    defaults to the full 169 canonical hands.

    NOTE:
    the full 169x169 table at a few thousand samples/pair takes several minutes in this pure-
    Python evaluator -- see save_equity_table/load_equity_table below to
    avoid recomputing it every run
    """
    hands = hands if hands is not None else canonical_hands()
    progress = None
    n = len(hands)
    total_pairs = n * (n + 1) // 2
    if show_progress:
        progress = tqdm(total=total_pairs, desc="Building equity table", unit="pair")
    table: dict[tuple[str, str], EquityResult] = {}
    for i, ha in enumerate(hands):
        for hb in hands[i:]:
            result = hand_vs_hand_equity_cpp(ha, hb, n_samples=n_samples, seed=seed)
            table[(ha.label, hb.label)] = result
            table[(hb.label, ha.label)] = EquityResult(
                win=result.loss, tie=result.tie, loss=result.win
            )
            if progress:
                progress.update(1)
    if progress:
        progress.close()
    return table


def save_equity_table(table: dict[tuple[str, str], EquityResult], path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(table, f)


def load_equity_table(path: str | Path) -> dict[tuple[str, str], EquityResult]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_or_load_equity_table(
    path: str | Path,
    n_samples: int = 2000,
    seed: int = 0,
    hands: list[CanonicalHand] | None = None,
    show_progress: bool = False,
) -> dict[tuple[str, str], EquityResult]:
    """Load a cached table from disk if it exists, else build it and save
    it for next time. This is the function most callers actually want."""
    path = Path(path)
    if path.exists():
        return load_equity_table(path)
    table = build_equity_table(
        n_samples=n_samples, seed=seed, hands=hands, show_progress=show_progress
    )
    save_equity_table(table, path)
    return table
