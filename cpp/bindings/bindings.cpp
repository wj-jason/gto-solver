#include <pybind11/pybind11.h>

#include "solver/eval.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_poker_solver_cpp, m) {
    m.doc() = "Fast C++ hand evaluator bindings for poker_solver";

    m.def(
        "score_5card", &poker_solver::score_5card,
        py::arg("hand_mask"),
        "Score a 5-card hand given as a 52-bit mask (see eval.hpp "
        "for the bit layout). Higher return value = better hand."
    );

    m.def(
        "best_hand_score", &poker_solver::best_hand_score,
        py::arg("seven_card_mask"),
        "Best 5-of-7 card score, for a 7-card bitmask (hole cards + board)."
    );
}
