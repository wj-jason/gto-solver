#pragma once
// __device__ port of hand_evaluator.cpp's score_5card / best_hand_score.
//
// This is a DELIBERATE duplicate of the CPU logic in hand_evaluator.cpp,
// not a shared header -- device code can't use std::stable_sort or
// std::popcount (C++20 <bit>, host-only), so the handful of spots that
// relied on those are hand-rolled here (__popc intrinsic, a manual
// insertion sort over the 13 ranks). Keep the two in sync by hand if the
// scoring logic ever changes; there are only ~180 lines of it.
//
// Same bit layout as hand_evaluator.hpp: 52-bit mask, 4 x 13-bit suit
// chunks (spades, hearts, diamonds, clubs), bit 0 = rank 2, bit 12 = Ace.

#include <cstdint>

namespace poker_solver {
namespace device {

namespace detail {

constexpr int kSuitBits = 13;
constexpr uint32_t kSuitMask = (1u << kSuitBits) - 1;

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

__device__ __forceinline__ void extract_suit_masks(uint64_t hand_mask, uint32_t suits[4]) {
    suits[0] = static_cast<uint32_t>((hand_mask >> (0 * kSuitBits)) & kSuitMask);
    suits[1] = static_cast<uint32_t>((hand_mask >> (1 * kSuitBits)) & kSuitMask);
    suits[2] = static_cast<uint32_t>((hand_mask >> (2 * kSuitBits)) & kSuitMask);
    suits[3] = static_cast<uint32_t>((hand_mask >> (3 * kSuitBits)) & kSuitMask);
}

// Same window-scan approach as the CPU version, including the wheel
// special case (A-2-3-4-5 reported as high=3, i.e. "5-high").
__device__ __forceinline__ int find_straight_high(uint32_t rank_mask) {
    constexpr uint32_t kWheelMask = (1u << 12) | 0b1111u;
    if ((rank_mask & kWheelMask) == kWheelMask) {
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

__device__ __forceinline__ int encode(int category, const int* tiebreak, int n_tiebreak) {
    int score = category;
    for (int i = 0; i < n_tiebreak; ++i) {
        score = score * 15 + (tiebreak[i] + 1);
    }
    for (int i = n_tiebreak; i < 5; ++i) {
        score = score * 15;
    }
    return score;
}

}  // namespace detail

__device__ inline int score_5card(uint64_t hand_mask) {
    using namespace detail;

    uint32_t suits[4];
    extract_suit_masks(hand_mask, suits);
    uint32_t rank_mask = suits[0] | suits[1] | suits[2] | suits[3];

    bool is_flush = false;
    uint32_t flush_suit_mask = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        if (__popc(suits[s]) == 5) {
            is_flush = true;
            flush_suit_mask = suits[s];
        }
    }

    int straight_high = find_straight_high(is_flush ? flush_suit_mask : rank_mask);
    if (is_flush && straight_high >= 0) {
        int tb[5] = {straight_high, 0, 0, 0, 0};
        return encode(kStraightFlush, tb, 1);
    }

    int counts[kSuitBits];
#pragma unroll
    for (int r = 0; r < kSuitBits; ++r) {
        counts[r] = ((suits[0] >> r) & 1) + ((suits[1] >> r) & 1) +
                    ((suits[2] >> r) & 1) + ((suits[3] >> r) & 1);
    }

    // ranks sorted descending by (count, rank) -- manual insertion sort,
    // stand-in for the CPU version's std::stable_sort (device code can't
    // use <algorithm>'s host-only sort). 13 elements, fine unrolled.
    int ranks_desc[kSuitBits];
#pragma unroll
    for (int r = 0; r < kSuitBits; ++r) ranks_desc[r] = kSuitBits - 1 - r;
#pragma unroll
    for (int i = 1; i < kSuitBits; ++i) {
        int key = ranks_desc[i];
        int key_count = counts[key];
        int j = i - 1;
        while (j >= 0) {
            int cand = ranks_desc[j];
            bool goes_after = (counts[cand] != key_count) ? (counts[cand] < key_count)
                                                            : (cand < key);
            if (!goes_after) break;
            ranks_desc[j + 1] = ranks_desc[j];
            --j;
        }
        ranks_desc[j + 1] = key;
    }

    int top_count = counts[ranks_desc[0]];
    int second_count = counts[ranks_desc[1]];

    if (top_count == 4) {
        int tb[5] = {ranks_desc[0], ranks_desc[1], 0, 0, 0};
        return encode(kQuads, tb, 2);
    }
    if (top_count == 3 && second_count == 2) {
        int tb[5] = {ranks_desc[0], ranks_desc[1], 0, 0, 0};
        return encode(kFullHouse, tb, 2);
    }
    if (is_flush) {
        int top5[5] = {0, 0, 0, 0, 0};
        int idx = 0;
#pragma unroll
        for (int r = kSuitBits - 1; r >= 0 && idx < 5; --r) {
            if ((flush_suit_mask >> r) & 1) top5[idx++] = r;
        }
        return encode(kFlush, top5, 5);
    }

    int plain_straight_high = find_straight_high(rank_mask);
    if (plain_straight_high >= 0) {
        int tb[5] = {plain_straight_high, 0, 0, 0, 0};
        return encode(kStraight, tb, 1);
    }

    if (top_count == 3) {
        int tb[5] = {ranks_desc[0], ranks_desc[1], ranks_desc[2], 0, 0};
        return encode(kTrips, tb, 3);
    }
    if (top_count == 2 && second_count == 2) {
        int hi_pair = ranks_desc[0] > ranks_desc[1] ? ranks_desc[0] : ranks_desc[1];
        int lo_pair = ranks_desc[0] > ranks_desc[1] ? ranks_desc[1] : ranks_desc[0];
        int tb[5] = {hi_pair, lo_pair, ranks_desc[2], 0, 0};
        return encode(kTwoPair, tb, 3);
    }
    if (top_count == 2) {
        int tb[5] = {ranks_desc[0], ranks_desc[1], ranks_desc[2], ranks_desc[3], 0};
        return encode(kPair, tb, 4);
    }

    int tb[5] = {ranks_desc[0], ranks_desc[1], ranks_desc[2], ranks_desc[3], ranks_desc[4]};
    return encode(kHighCard, tb, 5);
}

__device__ inline int best_hand_score(uint64_t seven_card_mask) {
    int positions[7];
    int n = 0;
#pragma unroll 1
    for (int i = 0; i < 52 && n < 7; ++i) {
        if ((seven_card_mask >> i) & 1) positions[n++] = i;
    }

    int best = -1;
#pragma unroll 1
    for (int subset = 0; subset < (1 << 7); ++subset) {
        if (__popc(static_cast<unsigned>(subset)) != 5) continue;
        uint64_t five_card_mask = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i) {
            if ((subset >> i) & 1) five_card_mask |= (1ull << positions[i]);
        }
        int s = score_5card(five_card_mask);
        if (s > best) best = s;
    }
    return best;
}

}  // namespace device
}  // namespace poker_solver
