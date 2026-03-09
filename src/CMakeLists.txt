# Pi Compute - Source library
add_library(pi_engine_lib
    engine/pi_engine.cpp
    engine/binary_splitting.cpp
    arithmetic/gmp_multiplier.cpp
    arithmetic/newton_divider.cpp
    io/base_converter.cpp
    io/chunked_writer.cpp
)

target_include_directories(pi_engine_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GMP_INCLUDE_DIR}
)

target_link_libraries(pi_engine_lib PUBLIC
    ${GMP_LIBRARY}
)

# Main executable
add_executable(pi_compute main.cpp)
target_link_libraries(pi_compute PRIVATE pi_engine_lib)
