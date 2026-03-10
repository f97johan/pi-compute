cmake_minimum_required(VERSION 3.14)
project(pi_compute VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Aggressive optimization: -O3 + CPU-specific tuning (AVX2, AVX-512, etc.)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG")

# Option to enable CUDA (off by default for macOS dev)
option(ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

if(ENABLE_CUDA)
    if(CMAKE_VERSION VERSION_LESS "3.18")
        message(FATAL_ERROR "CUDA support requires CMake 3.18+. You have ${CMAKE_VERSION}. Build without -DENABLE_CUDA=ON for CPU-only.")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(PI_CUDA_ENABLED)
endif()

# Find GMP
find_library(GMP_LIBRARY gmp REQUIRED)
find_path(GMP_INCLUDE_DIR gmp.h REQUIRED)

# Find pthreads (needed for std::async/std::thread on Linux)
find_package(Threads REQUIRED)

# Google Test via FetchContent
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Source subdirectories
add_subdirectory(src)
add_subdirectory(tests)
