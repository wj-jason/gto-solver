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

# --- Experimental CUDA equity kernel (opt-in) -------------------------
# OFF by default so the ordinary CPU-only build (this container included --
# no nvcc/GPU here) is completely unaffected. Turn on explicitly with:
#   pip install -e . --config-settings=cmake.define.POKER_SOLVER_ENABLE_CUDA=ON
# or: cmake -DPOKER_SOLVER_ENABLE_CUDA=ON ... (if invoking CMake directly)
option(POKER_SOLVER_ENABLE_CUDA "Build the experimental CUDA equity kernel" OFF)

if(POKER_SOLVER_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # "native" (auto-detect the building machine's GPU) needs CMake
        # 3.24+ -- on anything older, CMake silently fails to resolve it
        # and CUDA_ARCHITECTURES ends up empty at generate time (that's
        # the "CUDA_ARCHITECTURES is empty" error, if you hit it). Fall
        # back to an explicit list covering Pascal through Ada/Hopper
        # (GTX 10-series through RTX 40-series / A100 / H100 / T4 / L4)
        # so the build still works on an older CMake; pip install -U cmake
        # to get "native" back if you'd rather not maintain this list.
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
                set(CMAKE_CUDA_ARCHITECTURES native)
            else()
                message(STATUS
                    "CMake ${CMAKE_VERSION} < 3.24: CMAKE_CUDA_ARCHITECTURES="
                    "native isn't supported, falling back to an explicit "
                    "architecture list (60;70;75;80;86;89;90). Pass "
                    "-DCMAKE_CUDA_ARCHITECTURES=<yours> to override, or "
                    "upgrade cmake to 3.24+ to auto-detect instead."
                )
                set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
            endif()
        endif()

        target_sources(_poker_solver_cpp PRIVATE src/equity_cuda.cu)
        target_compile_definitions(_poker_solver_cpp PRIVATE POKER_SOLVER_HAS_CUDA)
        target_compile_options(_poker_solver_cpp PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        )
        # Deliberately NOT setting CUDA_SEPARABLE_COMPILATION: that's only
        # needed when device code in one .cu file calls __device__
        # functions compiled in a DIFFERENT .cu file (relocatable device
        # code + a separate nvlink device-link step). We don't have that --
        # hand_evaluator_device.cuh's __device__ functions are #included
        # directly into equity_mc_cuda.cu, so they're already in the same
        # translation unit. Turning it on anyway adds an extra device-link
        # step (the cmake_device_link.o you'd see in the build log) whose
        # output doesn't always embed correctly into a pybind11 MODULE
        # target (vs. a plain SHARED library) -- that mismatch is what
        # produces "undefined symbol: fatbinData" at import time. If a
        # second .cu file is ever added that needs to call into this one's
        # device functions, turn this back on AND verify the import still
        # works (or move the shared device code into a header instead).
    else()
        message(WARNING
            "POKER_SOLVER_ENABLE_CUDA=ON but no CUDA compiler was found -- "
            "building CPU-only. Check that the CUDA toolkit is installed "
            "and nvcc is on PATH."
        )
    endif()
endif()
# ------------------------------------------------------------------------

# scikit-build-core installs the built extension module into the Python
# package directory so `from poker_solver import _poker_solver_cpp` (or
# however it's imported) just works after `pip install -e .`
install(TARGETS _poker_solver_cpp DESTINATION solver)
