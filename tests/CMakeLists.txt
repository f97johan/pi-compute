# Tests

# CPU tests (always built)
set(CPU_TEST_SOURCES
    test_gmp_multiplier.cpp
    test_binary_splitting.cpp
    test_newton_divider.cpp
    test_base_converter.cpp
    test_chunked_writer.cpp
    test_pi_engine.cpp
    test_validation.cpp
)

# GPU tests (only built with CUDA)
set(GPU_TEST_SOURCES "")
if(ENABLE_CUDA)
    set(GPU_TEST_SOURCES
        test_gpu_ntt_multiplier.cpp
    )
endif()

add_executable(pi_tests
    ${CPU_TEST_SOURCES}
    ${GPU_TEST_SOURCES}
)

target_link_libraries(pi_tests PRIVATE
    pi_engine_lib
    GTest::gtest_main
)

target_include_directories(pi_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Define path to test data directory
target_compile_definitions(pi_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

include(GoogleTest)
gtest_discover_tests(pi_tests)
