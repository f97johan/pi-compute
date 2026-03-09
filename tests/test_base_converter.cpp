/**
 * @file test_base_converter.cpp
 * @brief TDD tests for BaseConverter (mpf to decimal string).
 */

#include <gtest/gtest.h>
#include <gmp.h>
#include <string>
#include "io/base_converter.h"

class BaseConverterTest : public ::testing::Test {
protected:
    mpf_t value;

    void SetUp() override {
        mpf_init2(value, 512);
    }

    void TearDown() override {
        mpf_clear(value);
    }
};

TEST_F(BaseConverterTest, ConvertPiApproximation) {
    // Set value to 355/113 ≈ 3.14159292...
    mpf_set_ui(value, 355);
    mpf_t den;
    mpf_init_set_ui(den, 113);
    mpf_div(value, value, den);
    mpf_clear(den);

    std::string result = pi::BaseConverter::to_decimal_string(value, 8);
    EXPECT_TRUE(result.substr(0, 6) == "3.1415") << "Got: " << result;
}

TEST_F(BaseConverterTest, ConvertExactInteger) {
    mpf_set_ui(value, 42);
    std::string result = pi::BaseConverter::to_decimal_string(value, 5);
    // Should start with "42" or "42.000"
    EXPECT_TRUE(result.substr(0, 2) == "42") << "Got: " << result;
}

TEST_F(BaseConverterTest, ConvertSmallFraction) {
    // 1/3 = 0.33333...
    mpf_set_ui(value, 1);
    mpf_t den;
    mpf_init_set_ui(den, 3);
    mpf_div(value, value, den);
    mpf_clear(den);

    std::string result = pi::BaseConverter::to_decimal_string(value, 10);
    EXPECT_TRUE(result.find("0.3333") == 0) << "Got: " << result;
}

TEST_F(BaseConverterTest, ZeroDigitsThrows) {
    mpf_set_ui(value, 3);
    EXPECT_THROW(pi::BaseConverter::to_decimal_string(value, 0), std::invalid_argument);
}

TEST_F(BaseConverterTest, ConvertHighPrecision) {
    // Use GMP to compute a known value with high precision
    mpf_set_prec(value, 1024);
    mpf_set_ui(value, 2);
    mpf_sqrt(value, value);

    std::string result = pi::BaseConverter::to_decimal_string(value, 50);
    // sqrt(2) = 1.41421356237309504880168872420969807856967187537694...
    EXPECT_TRUE(result.substr(0, 16) == "1.41421356237309") << "Got: " << result;
}
