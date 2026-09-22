#pragma once
#include <cstdint>

namespace poker_solver {

// A hand is represented as a 52-bit mask packed as 4 x 13-bit suit chunks:
// bits 0-12 = spades, 13-25 = hearts, 26-38 = diamonds, 39-51 = clubs.
// Within a suit chunk, bit 0 = rank 2, bit 12 = rank Ace. This layout is
// what makes every check (flush, straight, rank counts) a handful of
// bitwise ops instead of Python-level loops over card objects -- see the
// project notes on "bitwise hand representation" for the full reasoning.

// Score a single 5-card hand. Higher return value = a better hand. The
// exact encoding of the returned int isn't meant to be human-readable --
// only its ordering relative to other scores matters.
int score_5card(uint64_t hand_mask);

// Best 5-of-7 score, for a 7-card mask (hole cards + board).
int best_hand_score(uint64_t seven_card_mask);

}  // namespace poker_solver
