#include "solver/eval.hpp"

#include <algorithm>
#include <array>
#include <bit>

namespace poker_solver {

namespace {

constexpr int kSuitBits = 13;
constexpr uint64_t kSuitMask = (1u << kSuitBits) - 1;  // 0b1111111111111

enum Category {
    kHighCard = 0,
    kPair = 1,
    kTwoPair = 2,
    kTrips = 3,
    kStraight = 4,
    kFlush = 5,
    kFullHouse = 6,
    kQuads = 7,
    kStraightFlush = 8,
};

std::array<uint32_t, kSuitBits> extract_suit_masks(uint64_t hand_mask) {
    return {
        static_cast<uint32_t>((hand_mask >> (0 * kSuitBits)) & kSuitMask),
        static_cast<uint32_t>((hand_mask >> (1 * kSuitBits)) & kSuitMask),
        static_cast<uint32_t>((hand_mask >> (2 * kSuitBits)) & kSuitMask),
        static_cast<uint32_t>((hand_mask >> (3 * kSuitBits)) & kSuitMask),
    };
}

// Returns the high rank (0-12) of a straight found in rank_mask, or -1.
// Checks all 10 possible 5-consecutive-rank windows, including the wheel
// (A-2-3-4-5, where rank 12 = Ace plays low alongside ranks 0-3).
int find_straight_high(uint32_t rank_mask) {
    // wheel: ranks {12 (A), 0, 1, 2, 3}
    constexpr uint32_t kWheelMask = (1u << 12) | 0b1111;
    if ((rank_mask & kWheelMask) == kWheelMask) {
        // report as high=3 (a "5-high" straight) so it correctly loses to
        // a 6-high straight etc. -- matches the Python wheel convention
        return 3;
    }
    for (int high = 4; high <= 12; ++high) {
        uint32_t window = 0b11111u << (high - 4);
        if ((rank_mask & window) == window) {
            return high;
        }
    }
    return -1;
}

// Encode (category, tiebreakers...) into a single comparable int. Base-15
// packing is enough since ranks fit in 0-14 (13 ranks used here, 0-12).
int encode(Category category, std::array<int, 5> tiebreak, int n_tiebreak) {
    int score = static_cast<int>(category);
    for (int i = 0; i < n_tiebreak; ++i) {
        score = score * 15 + (tiebreak[i] + 1);  // +1 so "no value" (0) sorts lowest
    }
    // pad remaining slots so all categories produce comparable-width scores
    for (int i = n_tiebreak; i < 5; ++i) {
        score = score * 15;
    }
    return score;
}

}  // namespace

int score_5card(uint64_t hand_mask) {
    auto suits = extract_suit_masks(hand_mask);
    uint32_t rank_mask = suits[0] | suits[1] | suits[2] | suits[3];

    bool is_flush = false;
    uint32_t flush_suit_mask = 0;
    for (uint32_t s : suits) {
        if (std::popcount(s) == 5) {
            is_flush = true;
            flush_suit_mask = s;
        }
    }

    int straight_high = find_straight_high(is_flush ? flush_suit_mask : rank_mask);
    // NOTE: straight-flush needs the straight to exist WITHIN the flush
    // suit specifically; a non-flush straight check uses the combined
    // rank_mask instead (computed below if not a flush).
    if (is_flush && straight_high >= 0) {
        return encode(kStraightFlush, {straight_high, 0, 0, 0, 0}, 1);
    }

    std::array<int, kSuitBits> counts{};
    for (int r = 0; r < kSuitBits; ++r) {
        counts[r] = ((suits[0] >> r) & 1) + ((suits[1] >> r) & 1) +
                    ((suits[2] >> r) & 1) + ((suits[3] >> r) & 1);
    }

    // ranks sorted descending by (count, rank) -- mirrors Counter.most_common
    std::array<int, kSuitBits> ranks_desc;
    for (int r = 0; r < kSuitBits; ++r) ranks_desc[r] = kSuitBits - 1 - r;
    std::stable_sort(ranks_desc.begin(), ranks_desc.end(), [&](int a, int b) {
        if (counts[a] != counts[b]) return counts[a] > counts[b];
        return a > b;
    });

    int top_count = counts[ranks_desc[0]];
    int second_count = counts[ranks_desc[1]];

    if (top_count == 4) {
        int quad_rank = ranks_desc[0];
        int kicker = ranks_desc[1];
        return encode(kQuads, {quad_rank, kicker, 0, 0, 0}, 2);
    }
    if (top_count == 3 && second_count == 2) {
        return encode(kFullHouse, {ranks_desc[0], ranks_desc[1], 0, 0, 0}, 2);
    }
    if (is_flush) {
        std::array<int, kSuitBits> flush_ranks_desc;
        for (int r = 0; r < kSuitBits; ++r) flush_ranks_desc[r] = kSuitBits - 1 - r;
        // ranks present in the flush suit, descending
        std::array<int, 5> top5{};
        int idx = 0;
        for (int r = kSuitBits - 1; r >= 0 && idx < 5; --r) {
            if ((flush_suit_mask >> r) & 1) top5[idx++] = r;
        }
        return encode(kFlush, top5, 5);
    }

    // non-flush straight check (combined across suits)
    int plain_straight_high = find_straight_high(rank_mask);
    if (plain_straight_high >= 0) {
        return encode(kStraight, {plain_straight_high, 0, 0, 0, 0}, 1);
    }

    if (top_count == 3) {
        int trips_rank = ranks_desc[0];
        std::array<int, 5> tiebreak = {trips_rank, ranks_desc[1], ranks_desc[2], 0, 0};
        return encode(kTrips, tiebreak, 3);
    }
    if (top_count == 2 && second_count == 2) {
        int hi_pair = std::max(ranks_desc[0], ranks_desc[1]);
        int lo_pair = std::min(ranks_desc[0], ranks_desc[1]);
        int kicker = ranks_desc[2];
        return encode(kTwoPair, {hi_pair, lo_pair, kicker, 0, 0}, 3);
    }
    if (top_count == 2) {
        int pair_rank = ranks_desc[0];
        std::array<int, 5> tiebreak = {pair_rank, ranks_desc[1], ranks_desc[2], ranks_desc[3], 0};
        return encode(kPair, tiebreak, 4);
    }

    std::array<int, 5> tiebreak = {ranks_desc[0], ranks_desc[1], ranks_desc[2],
                                    ranks_desc[3], ranks_desc[4]};
    return encode(kHighCard, tiebreak, 5);
}

int best_hand_score(uint64_t seven_card_mask) {
    // extract positions of the 7 set bits
    std::array<int, 7> positions{};
    int n = 0;
    for (int i = 0; i < 52 && n < 7; ++i) {
        if ((seven_card_mask >> i) & 1) positions[n++] = i;
    }

    int best = -1;
    // enumerate all C(7,5) = 21 subsets via a 5-bit-set-of-7 counter
    for (int subset = 0; subset < (1 << 7); ++subset) {
        if (std::popcount(static_cast<unsigned>(subset)) != 5) continue;
        uint64_t five_card_mask = 0;
        for (int i = 0; i < 7; ++i) {
            if ((subset >> i) & 1) five_card_mask |= (1ull << positions[i]);
        }
        best = std::max(best, score_5card(five_card_mask));
    }
    return best;
}

}  // namespace poker_solver
