#pragma once
// Host-side declaration only -- safe to include from bindings.cpp (a plain
// .cpp file compiled by the regular C++ compiler, not nvcc). The kernel
// itself lives in equity_mc_cuda.cu and is never exposed here.

#include <cstdint>
#include <utility>
#include <vector>

#include "solver/equity.hpp"  // reuses EquityCounts

namespace poker_solver {

// Naive, unoptimized CUDA port of simulate_equity, deliberately scoped to
// ONE hand pair -- this is a correctness-first starting point, not the
// batched-across-all-hand-pairs version discussed in the design notes.
// One thread per trial, one kernel launch, curand per-thread state,
// straight atomicAdd into global counters (no block-level shared-memory
// reduction yet). Optimize only after this is verified correct against
// the CPU simulate_equity on the same inputs.
//
// Also simplifies conflict handling vs. the CPU version's resample-until-
// valid loop: each thread gets a small fixed number of attempts (see
// kMaxAttemptsPerTrial in the .cu file) and just doesn't count itself if
// none succeed, rather than looping indefinitely (loop divergence is the
// exact problem discussed for why GPU threads shouldn't spin in an
// unbounded resample loop). n_trials in the returned EquityCounts may
// therefore be somewhat below n_samples -- same "can be < requested"
// contract as the CPU version, just a different reason.
EquityCounts simulate_equity_cuda(
    const std::vector<std::pair<int, int>>& combos_a,
    const std::vector<std::pair<int, int>>& combos_b,
    const std::vector<int>& board,
    int n_samples,
    uint64_t seed
);

}  // namespace poker_solver
