/**
 * @file test_pi_engine.cpp
 * @brief Integration tests for PiEngine — compute pi and verify against known digits.
 */

#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include "engine/pi_engine.h"
#include "arithmetic/gmp_multiplier.h"

class PiEngineTest : public ::testing::Test {
protected:
    pi::GmpMultiplier multiplier;

    std::string load_reference(const std::string& filename) {
        std::string path = std::string(TEST_DATA_DIR) + "/" + filename;
        std::ifstream f(path);
        if (!f.is_open()) {
            return "";
        }
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    }
};

TEST_F(PiEngineTest, Compute50Digits) {
    pi::PiEngine engine(multiplier);
    std::string result = engine.compute_digits(50);

    // First 50 digits of pi
    std::string expected = "3.14159265358979323846264338327950288419716939937510";
    EXPECT_EQ(result.substr(0, expected.size()), expected)
        << "Got: " << result.substr(0, 60);
}

TEST_F(PiEngineTest, Compute100Digits) {
    pi::PiEngine engine(multiplier);
    std::string result = engine.compute_digits(100);

    std::string expected_start = "3.14159265358979323846264338327950288419716939937510"
                                  "58209749445923078164062862089986280348253421170679";
    EXPECT_EQ(result.substr(0, expected_start.size()), expected_start)
        << "Got: " << result.substr(0, 110);
}

TEST_F(PiEngineTest, Compute1000DigitsMatchesReference) {
    pi::PiEngine engine(multiplier);
    std::string result = engine.compute_digits(1000);

    std::string reference = load_reference("pi_1000.txt");
    ASSERT_FALSE(reference.empty()) << "Could not load pi_1000.txt reference file";

    // Compare digit by digit
    size_t min_len = std::min(result.size(), reference.size());
    for (size_t i = 0; i < min_len; ++i) {
        EXPECT_EQ(result[i], reference[i])
            << "Mismatch at position " << i
            << ": got '" << result[i] << "', expected '" << reference[i] << "'"
            << "\nContext: ..." << result.substr(std::max(0, (int)i - 5), 10) << "...";
    }
}

TEST_F(PiEngineTest, ComputeReturnsTimingInfo) {
    pi::PiEngine engine(multiplier);
    pi::PiConfig config;
    config.digits = 100;

    pi::PiResult result = engine.compute(config);

    EXPECT_GT(result.elapsed_seconds, 0.0);
    EXPECT_GT(result.terms_used, 0u);
    EXPECT_FALSE(result.digits.empty());
}

TEST_F(PiEngineTest, Compute10000Digits) {
    pi::PiEngine engine(multiplier);
    std::string result = engine.compute_digits(10000);

    // Verify length: "3." + 10000 digits = 10002 characters
    EXPECT_GE(result.size(), 10001u);

    // Verify starts correctly
    EXPECT_EQ(result.substr(0, 7), "3.14159");
}
