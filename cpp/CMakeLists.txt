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

install(TARGETS _poker_solver_cpp DESTINATION solver)
