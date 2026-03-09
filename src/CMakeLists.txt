# Pi Compute - Source library

# Core CPU sources (always built)
set(PI_CPU_SOURCES
    engine/pi_engine.cpp
    engine/binary_splitting.cpp
    arithmetic/gmp_multiplier.cpp
    arithmetic/newton_divider.cpp
    io/base_converter.cpp
    io/chunked_writer.cpp
)

# GPU sources (only built with CUDA)
set(PI_GPU_SOURCES "")
if(ENABLE_CUDA)
    set(PI_GPU_SOURCES
        gpu/ntt_engine.cu
        gpu/pointwise_multiply.cu
        gpu/carry_propagation.cu
        gpu/int_ntt.cu
        arithmetic/gpu_ntt_multiplier.cpp
        arithmetic/int_ntt_multiplier.cpp
    )
endif()

add_library(pi_engine_lib
    ${PI_CPU_SOURCES}
    ${PI_GPU_SOURCES}
)

target_include_directories(pi_engine_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GMP_INCLUDE_DIR}
)

target_link_libraries(pi_engine_lib PUBLIC
    ${GMP_LIBRARY}
)

if(ENABLE_CUDA)
    target_link_libraries(pi_engine_lib PUBLIC
        CUDA::cufft
        CUDA::cudart
    )
    # Auto-detect GPU architecture at build time.
    # "native" queries the installed GPU; falls back to a safe default if no GPU present.
    # CMake 3.24+ supports "native". For older CMake, we use a broad range.
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set_target_properties(pi_engine_lib PROPERTIES
            CUDA_ARCHITECTURES "native"
        )
    else()
        # Fallback: support Ampere through Blackwell (skip retired Volta/Turing)
        set_target_properties(pi_engine_lib PROPERTIES
            CUDA_ARCHITECTURES "80;86;89;90;100"
        )
    endif()
endif()

# Main executable
add_executable(pi_compute main.cpp)
target_link_libraries(pi_compute PRIVATE pi_engine_lib)
