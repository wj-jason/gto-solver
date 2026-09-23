"""
Usage:
    full 169-hand grid vs random, default samples
        `python visualize_equity_table.py`
    vs a specific hand
        `python visualize_equity_table.py --vs AA`
    specify mc samples
        `python visualize_equity_table.py --n-samples 200`
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from solver.eval.equity import build_or_load_equity_table
from solver.game.cards import RANKS, canonical_hands

RANKS_DESC = RANKS[::-1]  # 'AKQJT98765432'

CMAP = "Blues"


def grid_label(row_rank: str, col_rank: str) -> str:
    """Standard grid convention: diagonal = pair; above the diagonal
    (row rank higher than column rank) = suited; below = offsuit."""
    i = RANKS_DESC.index(row_rank)
    j = RANKS_DESC.index(col_rank)
    if i == j:
        return row_rank + col_rank
    hi, lo = (row_rank, col_rank) if i < j else (col_rank, row_rank)
    return f"{hi}{lo}s" if i < j else f"{hi}{lo}o"


def avg_equity_vs_random(label: str, equity_table: dict, hands_by_label: dict) -> float:
    """Combo-weighted average equity of `label` against every other
    canonical hand -- i.e. equity vs a uniformly random opponent hand."""
    total_weight = 0.0
    weighted_sum = 0.0
    for other_label, other_hand in hands_by_label.items():
        if other_label == label:
            continue  # can't be dealt the exact same canonical hand as "the opponent" here
        w = other_hand.n_combos
        weighted_sum += w * equity_table[(label, other_label)].equity
        total_weight += w
    return weighted_sum / total_weight


def build_grid(
    equity_table: dict, hands_by_label: dict, vs_label: str | None
) -> np.ndarray:
    n = len(RANKS_DESC)
    grid = np.zeros((n, n))
    for i, row_rank in enumerate(RANKS_DESC):
        for j, col_rank in enumerate(RANKS_DESC):
            label = grid_label(row_rank, col_rank)
            if vs_label is None:
                grid[i, j] = avg_equity_vs_random(label, equity_table, hands_by_label)
            else:
                grid[i, j] = equity_table[(label, vs_label)].equity
    return grid


def plot_grid(grid: np.ndarray, title: str, out_path: str) -> None:
    n = len(RANKS_DESC)
    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(grid, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(list(RANKS_DESC))
    ax.set_yticklabels(list(RANKS_DESC))
    ax.set_xlabel("second card")
    ax.set_ylabel("first card")
    ax.set_title(title, fontsize=13, pad=12)

    # grid lines between cells (thin, recessive -- not competing with data)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # per-cell labels: hand name + equity %, with text color flipped for
    # readability against the light-vs-dark cell background
    for i, row_rank in enumerate(RANKS_DESC):
        for j, col_rank in enumerate(RANKS_DESC):
            label = grid_label(row_rank, col_rank)
            equity_pct = grid[i, j] * 100
            text_color = "white" if grid[i, j] > 0.55 else "black"
            ax.text(
                j,
                i - 0.15,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
                fontweight="bold",
            )
            ax.text(
                j,
                i + 0.22,
                f"{equity_pct:.1f}%",
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("equity", rotation=270, labelpad=15)
    cbar.ax.yaxis.set_major_formatter(lambda x, _: f"{x*100:.0f}%")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vs",
        dest="vs_label",
        default=None,
        help="opponent hand label (e.g. AA, 72o). Omit for equity vs a random hand.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=500, help="MC samples per hand pair"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-path",
        default="equity_table_cache.pkl",
        help="where to cache the built equity table (reused on future runs)",
    )
    parser.add_argument("--out", default="preflop_equity_grid.png")
    args = parser.parse_args()

    hands = canonical_hands()
    hands_by_label = {h.label: h for h in hands}

    print(
        f"building/loading equity table (n_samples={args.n_samples}, cache={args.cache_path})..."
    )
    t0 = time.perf_counter()
    equity_table = build_or_load_equity_table(
        args.cache_path,
        n_samples=args.n_samples,
        seed=args.seed,
        hands=hands,
        show_progress=True,
    )
    print(f"equity table ready in {time.perf_counter() - t0:.1f}s")

    grid = build_grid(equity_table, hands_by_label, args.vs_label)

    if args.vs_label is None:
        title = "Preflop equity vs. a random hand"
    else:
        title = f"Preflop equity vs. {args.vs_label}"
    plot_grid(grid, title, args.out)


if __name__ == "__main__":
    main()
