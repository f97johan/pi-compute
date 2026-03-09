/**
 * @file test_gmp_multiplier.cpp
 * @brief TDD tests for GmpMultiplier.
 */

#include <gtest/gtest.h>
#include <gmp.h>
#include "arithmetic/gmp_multiplier.h"

class GmpMultiplierTest : public ::testing::Test {
protected:
    pi::GmpMultiplier multiplier;
    mpz_t a, b, result, expected;

    void SetUp() override {
        mpz_init(a);
        mpz_init(b);
        mpz_init(result);
        mpz_init(expected);
    }

    void TearDown() override {
        mpz_clear(a);
        mpz_clear(b);
        mpz_clear(result);
        mpz_clear(expected);
    }
};

TEST_F(GmpMultiplierTest, MultiplySmallNumbers) {
    mpz_set_ui(a, 123);
    mpz_set_ui(b, 456);
    mpz_set_ui(expected, 56088);

    multiplier.multiply(result, a, b);
    EXPECT_EQ(mpz_cmp(result, expected), 0);
}

TEST_F(GmpMultiplierTest, MultiplyByZero) {
    mpz_set_ui(a, 999999);
    mpz_set_ui(b, 0);
    mpz_set_ui(expected, 0);

    multiplier.multiply(result, a, b);
    EXPECT_EQ(mpz_cmp(result, expected), 0);
}

TEST_F(GmpMultiplierTest, MultiplyByOne) {
    mpz_set_ui(a, 42);
    mpz_set_ui(b, 1);
    mpz_set_ui(expected, 42);

    multiplier.multiply(result, a, b);
    EXPECT_EQ(mpz_cmp(result, expected), 0);
}

TEST_F(GmpMultiplierTest, MultiplyNegativeNumbers) {
    mpz_set_si(a, -7);
    mpz_set_si(b, 13);
    mpz_set_si(expected, -91);

    multiplier.multiply(result, a, b);
    EXPECT_EQ(mpz_cmp(result, expected), 0);
}

TEST_F(GmpMultiplierTest, MultiplyLargeNumbers) {
    // 10^50 * 10^50 = 10^100
    mpz_ui_pow_ui(a, 10, 50);
    mpz_ui_pow_ui(b, 10, 50);
    mpz_ui_pow_ui(expected, 10, 100);

    multiplier.multiply(result, a, b);
    EXPECT_EQ(mpz_cmp(result, expected), 0);
}

TEST_F(GmpMultiplierTest, MultiplyVeryLargeNumbers) {
    // Multiply two 1000-digit numbers and verify via GMP directly
    mpz_ui_pow_ui(a, 2, 3321);  // ~1000 decimal digits
    mpz_ui_pow_ui(b, 3, 2095);  // ~1000 decimal digits

    // Compute expected with GMP directly
    mpz_mul(expected, a, b);

    multiplier.multiply(result, a, b);
    EXPECT_EQ(mpz_cmp(result, expected), 0);
}

TEST_F(GmpMultiplierTest, SquareSmallNumber) {
    mpz_set_ui(a, 12345);
    mpz_set_ui(expected, 152399025UL);

    multiplier.square(result, a);
    EXPECT_EQ(mpz_cmp(result, expected), 0);
}

TEST_F(GmpMultiplierTest, SquareLargeNumber) {
    mpz_ui_pow_ui(a, 10, 500);
    mpz_ui_pow_ui(expected, 10, 1000);

    multiplier.square(result, a);
    EXPECT_EQ(mpz_cmp(result, expected), 0);
}

TEST_F(GmpMultiplierTest, SquareNegativeNumber) {
    mpz_set_si(a, -42);
    mpz_set_ui(expected, 1764);

    multiplier.square(result, a);
    EXPECT_EQ(mpz_cmp(result, expected), 0);
}

TEST_F(GmpMultiplierTest, MultiplyCommutative) {
    mpz_set_str(a, "123456789012345678901234567890", 10);
    mpz_set_str(b, "987654321098765432109876543210", 10);

    mpz_t result2;
    mpz_init(result2);

    multiplier.multiply(result, a, b);
    multiplier.multiply(result2, b, a);

    EXPECT_EQ(mpz_cmp(result, result2), 0);

    mpz_clear(result2);
}
