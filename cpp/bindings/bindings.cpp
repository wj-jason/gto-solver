#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // auto-converts Python list<tuple> <-> std::vector<std::pair>

#include "solver/equity.hpp"
#include "solver/eval.hpp"

#ifdef POKER_SOLVER_HAS_CUDA
#include "solver/equity_cuda.cuh"
#endif

namespace py = pybind11;

// The module name here ("_poker_solver_cpp") is what Python imports --
// see src/poker_solver/eval/fast_hand_evaluator.py for the thin Python
// wrapper that imports this and converts card strings <-> bitmasks, so
// the rest of the codebase never has to touch raw bitmasks directly.
PYBIND11_MODULE(_poker_solver_cpp, m) {
    m.doc() = "Fast C++ hand evaluator bindings for poker_solver";

    m.def(
        "score_5card", &poker_solver::score_5card,
        py::arg("hand_mask"),
        "Score a 5-card hand given as a 52-bit mask (see hand_evaluator.hpp "
        "for the bit layout). Higher return value = better hand."
    );

    m.def(
        "best_hand_score", &poker_solver::best_hand_score,
        py::arg("seven_card_mask"),
        "Best 5-of-7 card score, for a 7-card bitmask (hole cards + board)."
    );

    py::class_<poker_solver::EquityCounts>(m, "EquityCounts")
        .def_readonly("wins", &poker_solver::EquityCounts::wins)
        .def_readonly("ties", &poker_solver::EquityCounts::ties)
        .def_readonly("losses", &poker_solver::EquityCounts::losses)
        .def_readonly("n_trials", &poker_solver::EquityCounts::n_trials);

    m.def(
        "simulate_equity", &poker_solver::simulate_equity,
        py::arg("combos_a"), py::arg("combos_b"), py::arg("board"),
        py::arg("n_samples"), py::arg("seed"),
        "Full Monte Carlo equity simulation (sampling + evaluation), entirely "
        "in C++ -- one call per hand pair, not one per trial."
    );

#ifdef POKER_SOLVER_HAS_CUDA
    m.def(
        "simulate_equity_cuda", &poker_solver::simulate_equity_cuda,
        py::arg("combos_a"), py::arg("combos_b"), py::arg("board"),
        py::arg("n_samples"), py::arg("seed"),
        "Experimental single-hand-pair CUDA port of simulate_equity. Naive "
        "correctness-first version: one thread per trial, one kernel "
        "launch, global atomicAdd reduction, no shared-memory batching "
        "across hand pairs yet. Only built when POKER_SOLVER_ENABLE_CUDA "
        "is turned on at configure time."
    );
#endif
}
