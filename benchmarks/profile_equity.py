"""Profiling driver for the CUDA equity path.

Runs a warm-up call (so CUDA context creation doesn't pollute the profile),
then builds the equity table for a small subset of hands using the current
one-launch-per-hand-pair path.

Usage:
    python profile_equity.py            # default: first 20 hands -> 210 pairs
    python profile_equity.py 40 2000    # 40 hands, 2000 samples per pair
"""

from __future__ import annotations

import sys
import time

from solver.eval.equity import build_equity_table
from solver.eval.equity_cuda import hand_vs_hand_equity_cuda
from solver.game.cards import canonical_hands


def main() -> None:
    n_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    hands = canonical_hands()[:n_hands]
    n_pairs = n_hands * (n_hands + 1) // 2

    # Warm-up: first CUDA call pays for context creation and module loading.
    hand_vs_hand_equity_cuda("AA", "KK", n_samples=n_samples)

    start = time.perf_counter()
    build_equity_table(n_samples=n_samples, hands=hands)
    elapsed = time.perf_counter() - start

    print(f"{n_pairs} pairs x {n_samples} samples: {elapsed:.3f} s "
          f"({elapsed / n_pairs * 1e3:.3f} ms per pair)")


if __name__ == "__main__":
    main()
