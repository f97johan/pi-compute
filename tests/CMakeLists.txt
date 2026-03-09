# Tests
add_executable(pi_tests
    test_gmp_multiplier.cpp
    test_binary_splitting.cpp
    test_newton_divider.cpp
    test_base_converter.cpp
    test_chunked_writer.cpp
    test_pi_engine.cpp
    test_validation.cpp
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
