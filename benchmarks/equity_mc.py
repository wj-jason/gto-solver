from __future__ import annotations

import argparse
import time

from solver.eval.equity import hand_vs_hand_equity as hand_vs_hand_equity_py
from solver.eval.equity_cpp import hand_vs_hand_equity_cpp
from solver.eval.equity_cuda import hand_vs_hand_equity_cuda

from solver.game.cards import canonical_hands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hero", default="AA")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    villains = [h for h in canonical_hands() if h.label != args.hero]
    print(
        f"{args.hero} vs full range ({len(villains)} hands), "
        f"{args.n_samples} samples/hand, seed={args.seed}"
    )
    print()

    py_total_time = 0.0
    cpp_total_time = 0.0
    cuda_total_time = 0.0
    for h in villains:
        t0 = time.perf_counter()
        _ = hand_vs_hand_equity_py(
            args.hero, h.label, n_samples=args.n_samples, seed=args.seed
        )
        py_total_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = hand_vs_hand_equity_cpp(
            args.hero, h.label, n_samples=args.n_samples, seed=args.seed
        )
        cpp_total_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = hand_vs_hand_equity_cuda(
            args.hero, h.label, n_samples=args.n_samples, seed=args.seed
        )
        cuda_total_time += time.perf_counter() - t0

    total_trials = len(villains) * args.n_samples
    print(f"total trials evaluated (each language): {total_trials:,}")
    print()
    print(f"python | total time: {py_total_time:.3f}s")
    print(f"c++    | total time: {cpp_total_time:.3f}s")
    print(f"cuda   | total time: {cuda_total_time:.3f}s")
    print(f"speedup (python -> c++): {py_total_time / cpp_total_time:.1f}x")
    print(f"speedup (python -> cuda): {py_total_time / cuda_total_time:.1f}x")


if __name__ == "__main__":
    main()
