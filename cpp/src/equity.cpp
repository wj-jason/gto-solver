#include "solver/equity.hpp"
#include "solver/eval.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <iterator>
#include <random>

namespace poker_solver {

namespace {
inline uint64_t card_bit(int card_index) { return 1ull << card_index; }
}  // namespace

EquityCounts simulate_equity(
    const std::vector<std::pair<int, int>>& combos_a,
    const std::vector<std::pair<int, int>>& combos_b,
    const std::vector<int>& board,
    int n_samples,
    uint64_t seed
) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist_a(0, combos_a.size() - 1);
    std::uniform_int_distribution<size_t> dist_b(0, combos_b.size() - 1);

    uint64_t board_mask = 0;
    for (int c : board) board_mask |= card_bit(c);
    int remaining_needed = 5 - static_cast<int>(board.size());
    int board_size = static_cast<int>(board.size());

    int wins = 0, ties = 0, losses = 0, collected = 0;
    int attempts = 0;
    int max_attempts = n_samples * 30;  // guard against pathological conflict rates

    while (collected < n_samples && attempts < max_attempts) {
        ++attempts;
        auto [a1, a2] = combos_a[dist_a(rng)];
        auto [b1, b2] = combos_b[dist_b(rng)];
        uint64_t mask_a = card_bit(a1) | card_bit(a2);
        uint64_t mask_b = card_bit(b1) | card_bit(b2);
        uint64_t used = mask_a | mask_b | board_mask;

        if (std::popcount(used) != 4 + board_size) {
            continue;  // card-removal conflict (hole vs hole, or hole vs board) -- resample
        }

        std::array<int, 52> deck{};
        int deck_size = 0;
        for (int i = 0; i < 52; ++i) {
            if (!((used >> i) & 1)) deck[deck_size++] = i;
        }

        uint64_t extra_board_mask = 0;
        if (remaining_needed > 0) {
            std::array<int, 5> chosen{};
            std::sample(deck.begin(), deck.begin() + deck_size, chosen.begin(),
                        remaining_needed, rng);
            for (int i = 0; i < remaining_needed; ++i) {
                extra_board_mask |= card_bit(chosen[i]);
            }
        }

        uint64_t full_board_mask = board_mask | extra_board_mask;
        uint64_t seven_a = mask_a | full_board_mask;
        uint64_t seven_b = mask_b | full_board_mask;

        int score_a = best_hand_score(seven_a);
        int score_b = best_hand_score(seven_b);
        if (score_a > score_b) {
            ++wins;
        } else if (score_a < score_b) {
            ++losses;
        } else {
            ++ties;
        }
        ++collected;
    }

    return EquityCounts{wins, ties, losses, collected};
}

}  // namespace poker_solver