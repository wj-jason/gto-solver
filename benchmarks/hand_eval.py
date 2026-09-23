from __future__ import annotations

import argparse
import random
import time

from solver.eval.eval_cpp import compare as compare_cpp
from solver.eval.eval import compare as compare_py
from solver.game.cards import (
    RANKS,
    SUITS,
    canonical_hands,
    expand_to_combos,
    hand_by_label,
)

FULL_DECK = [r + s for r in RANKS for s in SUITS]


def sample_trials(hand_a_label: str, hand_b_label: str, n_samples: int, seed: int):
    """Pre-sample all trials (combo_a, combo_b, board) once -- shared
    between both evaluators so the comparison is apples-to-apples."""
    hand_a = hand_by_label(hand_a_label)
    hand_b = hand_by_label(hand_b_label)
    combos_a = expand_to_combos(hand_a)
    combos_b = expand_to_combos(hand_b)

    rng = random.Random(seed)
    trials = []
    attempts = 0
    max_attempts = n_samples * 30
    while len(trials) < n_samples and attempts < max_attempts:
        attempts += 1
        ca = combos_a[rng.randrange(len(combos_a))]
        cb = combos_b[rng.randrange(len(combos_b))]
        if set(ca) & set(cb):
            continue
        remaining = [c for c in FULL_DECK if c not in ca and c not in cb]
        board = rng.sample(remaining, 5)
        trials.append((ca, cb, board))
    return trials


def run_evaluator(trials, compare_fn) -> list[int]:
    return [compare_fn(list(ca) + board, list(cb) + board) for ca, cb, board in trials]


def equity_from_results(results: list[int]) -> float:
    wins = sum(1 for r in results if r == 1)
    ties = sum(1 for r in results if r == 0)
    return (wins + 0.5 * ties) / len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hero", default="AA", help="the fixed hand (default: AA)")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    villains = [h for h in canonical_hands() if h.label != args.hero]
    print(
        f"{args.hero} vs full range ({len(villains)} hands), "
        f"{args.n_samples} samples/hand, seed={args.seed}"
    )
    print()

    total_mismatches = 0
    py_total_time = 0.0
    cpp_total_time = 0.0
    equities_py: dict[str, float] = {}
    equities_cpp: dict[str, float] = {}

    for h in villains:
        trials = sample_trials(args.hero, h.label, args.n_samples, args.seed)

        t0 = time.perf_counter()
        results_py = run_evaluator(trials, compare_py)
        py_total_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        results_cpp = run_evaluator(trials, compare_cpp)
        cpp_total_time += time.perf_counter() - t0

        mismatches = sum(1 for a, b in zip(results_py, results_cpp) if a != b)
        total_mismatches += mismatches
        if mismatches:
            print(
                f"  MISMATCH: {args.hero} vs {h.label} -- "
                f"{mismatches}/{args.n_samples} trials disagree"
            )

        equities_py[h.label] = equity_from_results(results_py)
        equities_cpp[h.label] = equity_from_results(results_cpp)

    total_trials = len(villains) * args.n_samples
    print()
    print(f"total trials evaluated (each evaluator): {total_trials:,}")
    if total_mismatches > 0:
        print(
            f"correctness: {total_mismatches} MISMATCHES FOUND -- evaluators disagree, investigate"
        )
    print()
    print(f"python evaluator total time: {py_total_time:.3f}s")
    print(f"c++    evaluator total time: {cpp_total_time:.3f}s")
    print(f"speedup: {py_total_time / cpp_total_time:.1f}x")
    print()


if __name__ == "__main__":
    main()
