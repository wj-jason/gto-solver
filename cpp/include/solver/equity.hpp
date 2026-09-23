#pragma once
#include <cstdint>
#include <utility>
#include <vector>

namespace poker_solver {

struct EquityCounts {
    int wins;
    int ties;
    int losses;
    int n_trials;  // actually collected (<= n_samples, in case of pathological conflict rates)
};

// Runs the FULL Monte Carlo equity loop entirely in C++: sampling combos,
// checking card-removal conflicts, sampling the remaining board, and
// evaluating -- all without crossing back into Python per trial. Only one
// call per hand pair, unlike calling the fast evaluator from a Python
// sampling loop (which still pays Python-level overhead for the RNG,
// conflict checks, and list rebuilding every trial -- see project notes).
//
// combos_a / combos_b: each entry is a pair of card indices (0-51, where
// index = suit*13 + rank, suit order s,h,d,c -- matches the bit layout in
// hand_evaluator.hpp). board: 0, 3, 4, or 5 existing board card indices
// (empty = preflop).
EquityCounts simulate_equity(
    const std::vector<std::pair<int, int>>& combos_a,
    const std::vector<std::pair<int, int>>& combos_b,
    const std::vector<int>& board,
    int n_samples,
    uint64_t seed
);

}  // namespace poker_solver
