/**
 * @file test_validation.cpp
 * @brief Validation tests that compare computed pi against public reference data.
 *
 * These tests are slower (computing many digits) and are tagged for separate execution.
 */

#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include "engine/pi_engine.h"
#include "arithmetic/gmp_multiplier.h"

class ValidationTest : public ::testing::Test {
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

TEST_F(ValidationTest, First50DigitsHardcoded) {
    pi::PiEngine engine(multiplier);
    std::string result = engine.compute_digits(50);

    // Hardcoded known value — no external dependency
    EXPECT_EQ(result,
        "3.14159265358979323846264338327950288419716939937510");
}

TEST_F(ValidationTest, First100DigitsHardcoded) {
    pi::PiEngine engine(multiplier);
    std::string result = engine.compute_digits(100);

    std::string expected =
        "3.14159265358979323846264338327950288419716939937510"
        "58209749445923078164062862089986280348253421170679";
    EXPECT_EQ(result, expected);
}

TEST_F(ValidationTest, First1000DigitsMatchReference) {
    pi::PiEngine engine(multiplier);
    std::string result = engine.compute_digits(1000);

    std::string reference = load_reference("pi_1000.txt");
    ASSERT_FALSE(reference.empty()) << "Could not load pi_1000.txt reference file";

    // Trim any trailing whitespace/newlines from reference
    while (!reference.empty() && (reference.back() == '\n' || reference.back() == '\r' || reference.back() == ' ')) {
        reference.pop_back();
    }

    EXPECT_EQ(result.size(), reference.size())
        << "Length mismatch: computed " << result.size() << " vs reference " << reference.size();

    // Find first mismatch for better error reporting
    for (size_t i = 0; i < std::min(result.size(), reference.size()); ++i) {
        if (result[i] != reference[i]) {
            FAIL() << "First mismatch at position " << i
                   << ": computed '" << result[i] << "' vs reference '" << reference[i] << "'"
                   << "\nComputed context: ..." << result.substr(std::max(size_t(0), i - 10), 20) << "..."
                   << "\nReference context: ..." << reference.substr(std::max(size_t(0), i - 10), 20) << "...";
        }
    }
}

TEST_F(ValidationTest, DigitConsistencyAcrossScales) {
    // Compute at different scales and verify they agree on overlapping digits
    pi::PiEngine engine(multiplier);

    std::string pi_100 = engine.compute_digits(100);
    std::string pi_500 = engine.compute_digits(500);
    std::string pi_1000 = engine.compute_digits(1000);

    // First 100 digits should match across all computations
    EXPECT_EQ(pi_100, pi_500.substr(0, pi_100.size()))
        << "100-digit and 500-digit computations disagree";
    EXPECT_EQ(pi_100, pi_1000.substr(0, pi_100.size()))
        << "100-digit and 1000-digit computations disagree";
    EXPECT_EQ(pi_500, pi_1000.substr(0, pi_500.size()))
        << "500-digit and 1000-digit computations disagree";
}
