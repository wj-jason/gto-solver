#include "solver/equity_cuda.cuh"
#include "solver/eval_device.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <stdexcept>
#include <string>

namespace poker_solver {

namespace {

__host__ __device__ inline uint64_t card_bit(int card_index) { return 1ull << card_index; }

// Minimal starting point: bounded retries instead of the CPU version's
// unbounded resample-while-conflicting loop. Keeps every thread's control
// flow shaped the same way regardless of whether it eventually finds a
// valid deal -- see equity_cuda.cuh for why (warp divergence).
constexpr int kMaxAttemptsPerTrial = 8;
constexpr int kThreadsPerBlock = 256;

struct Combo {
    int c1;
    int c2;
};

// One thread == one trial. `board_mask` and `remaining_needed` are the
// same board every thread completes; `combos_a`/`combos_b` are the full
// combo lists for this one hand pair, read redundantly from global memory
// by every thread -- exactly the kind of read the shared-memory version
// (next optimization pass, not this one) would hoist once per block.
__global__ void equity_kernel(
    const Combo* combos_a, int n_combos_a,
    const Combo* combos_b, int n_combos_b,
    uint64_t board_mask, int board_size, int remaining_needed,
    int n_samples, uint64_t seed,
    int* d_wins, int* d_ties, int* d_losses, int* d_valid
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples) return;

    curandStatePhilox4_32_10_t rng;
    // subsequence = tid guarantees independent streams per thread (see
    // curand docs on subsequence vs. offset) -- analogous to how the CPU
    // version's single std::mt19937_64 is replaced by one stream per
    // trial instead of one shared generator.
    curand_init(seed, /*subsequence=*/static_cast<uint64_t>(tid), /*offset=*/0, &rng);

    for (int attempt = 0; attempt < kMaxAttemptsPerTrial; ++attempt) {
        int ia = curand(&rng) % n_combos_a;
        int ib = curand(&rng) % n_combos_b;
        Combo a = combos_a[ia];
        Combo b = combos_b[ib];

        uint64_t mask_a = card_bit(a.c1) | card_bit(a.c2);
        uint64_t mask_b = card_bit(b.c1) | card_bit(b.c2);
        uint64_t used = mask_a | mask_b | board_mask;

        if (__popcll(used) != 4 + board_size) {
            continue;  // hole-vs-hole or hole-vs-board conflict -- retry
        }

        // Complete the board WITHOUT materializing a 52-card deck array
        // (the register/local-memory spill problem flagged in the design
        // notes) -- draw remaining_needed distinct cards by rejection
        // sampling directly against the 52-bit `used` mask, which fits in
        // a register.
        uint64_t extra_mask = 0;
        bool ok = true;
        for (int k = 0; k < remaining_needed; ++k) {
            int found = -1;
            for (int try_card = 0; try_card < 52; ++try_card) {
                // deterministic scan with a random starting offset, so we
                // don't need an unbounded "keep drawing until free" loop
                int idx = (curand(&rng) + try_card) % 52;
                uint64_t bit = card_bit(idx);
                if (!(used & bit) && !(extra_mask & bit)) {
                    found = idx;
                    break;
                }
            }
            if (found < 0) {
                ok = false;
                break;
            }
            extra_mask |= card_bit(found);
        }
        if (!ok) continue;

        uint64_t full_board_mask = board_mask | extra_mask;
        uint64_t seven_a = mask_a | full_board_mask;
        uint64_t seven_b = mask_b | full_board_mask;

        int score_a = device::best_hand_score(seven_a);
        int score_b = device::best_hand_score(seven_b);

        atomicAdd(d_valid, 1);
        if (score_a > score_b) {
            atomicAdd(d_wins, 1);
        } else if (score_a < score_b) {
            atomicAdd(d_losses, 1);
        } else {
            atomicAdd(d_ties, 1);
        }
        return;
    }
    // exhausted attempts without a valid deal -- this trial just doesn't
    // count (n_trials in the result will be < n_samples), same contract
    // as the CPU version's max_attempts guard.
}

#define CUDA_CHECK(expr)                                                            \
    do {                                                                            \
        cudaError_t _err = (expr);                                                  \
        if (_err != cudaSuccess) {                                                  \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err)); \
        }                                                                            \
    } while (0)

}  // namespace

EquityCounts simulate_equity_cuda(
    const std::vector<std::pair<int, int>>& combos_a,
    const std::vector<std::pair<int, int>>& combos_b,
    const std::vector<int>& board,
    int n_samples,
    uint64_t seed
) {
    if (combos_a.empty() || combos_b.empty()) {
        throw std::invalid_argument("combos_a/combos_b must be non-empty");
    }

    uint64_t board_mask = 0;
    for (int c : board) board_mask |= card_bit(c);
    int board_size = static_cast<int>(board.size());
    int remaining_needed = 5 - board_size;

    std::vector<Combo> host_a(combos_a.size());
    for (size_t i = 0; i < combos_a.size(); ++i) host_a[i] = {combos_a[i].first, combos_a[i].second};
    std::vector<Combo> host_b(combos_b.size());
    for (size_t i = 0; i < combos_b.size(); ++i) host_b[i] = {combos_b[i].first, combos_b[i].second};

    Combo* d_combos_a = nullptr;
    Combo* d_combos_b = nullptr;
    int* d_wins = nullptr;
    int* d_ties = nullptr;
    int* d_losses = nullptr;
    int* d_valid = nullptr;

    CUDA_CHECK(cudaMalloc(&d_combos_a, host_a.size() * sizeof(Combo)));
    CUDA_CHECK(cudaMalloc(&d_combos_b, host_b.size() * sizeof(Combo)));
    CUDA_CHECK(cudaMalloc(&d_wins, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ties, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_losses, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valid, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_combos_a, host_a.data(), host_a.size() * sizeof(Combo), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_combos_b, host_b.data(), host_b.size() * sizeof(Combo), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_wins, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ties, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_losses, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_valid, 0, sizeof(int)));

    int num_blocks = (n_samples + kThreadsPerBlock - 1) / kThreadsPerBlock;
    equity_kernel<<<num_blocks, kThreadsPerBlock>>>(
        d_combos_a, static_cast<int>(host_a.size()),
        d_combos_b, static_cast<int>(host_b.size()),
        board_mask, board_size, remaining_needed,
        n_samples, seed,
        d_wins, d_ties, d_losses, d_valid
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int wins = 0, ties = 0, losses = 0, valid = 0;
    CUDA_CHECK(cudaMemcpy(&wins, d_wins, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&ties, d_ties, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&losses, d_losses, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_combos_a);
    cudaFree(d_combos_b);
    cudaFree(d_wins);
    cudaFree(d_ties);
    cudaFree(d_losses);
    cudaFree(d_valid);

    return EquityCounts{wins, ties, losses, valid};
}

}  // namespace poker_solver
