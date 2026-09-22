cmake_minimum_required(VERSION 3.18)
project(poker_solver_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)  # needed for std::popcount
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_poker_solver_cpp
    bindings/bindings.cpp
    src/eval.cpp
    src/equity.cpp
)

target_include_directories(_poker_solver_cpp PRIVATE include)
target_compile_options(_poker_solver_cpp PRIVATE -O3)

option(POKER_SOLVER_ENABLE_CUDA "Build the experimental CUDA equity kernel" OFF)

if(POKER_SOLVER_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()

        target_sources(_poker_solver_cpp PRIVATE src/equity_mc_cuda.cu)
        target_compile_definitions(_poker_solver_cpp PRIVATE POKER_SOLVER_HAS_CUDA)
        target_compile_options(_poker_solver_cpp PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        )
        set_target_properties(_poker_solver_cpp PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    else()
        message(WARNING
            "POKER_SOLVER_ENABLE_CUDA=ON but no CUDA compiler was found -- "
            "building CPU-only. Check that the CUDA toolkit is installed "
            "and nvcc is on PATH."
        )
    endif()
endif()
# ------------------------------------------------------------------------

install(TARGETS _poker_solver_cpp DESTINATION solver)
