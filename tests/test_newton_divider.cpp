/**
 * @file test_newton_divider.cpp
 * @brief TDD tests for NewtonDivider (division and square root).
 */

#include <gtest/gtest.h>
#include <gmp.h>
#include <string>
#include "arithmetic/newton_divider.h"

class NewtonDividerTest : public ::testing::Test {
protected:
    mpf_t result;
    mpz_t num, den;

    void SetUp() override {
        mpf_init(result);
        mpz_init(num);
        mpz_init(den);
    }

    void TearDown() override {
        mpf_clear(result);
        mpz_clear(num);
        mpz_clear(den);
    }
};

TEST_F(NewtonDividerTest, DivideSimple) {
    // 22 / 7 ≈ 3.142857142857...
    mpz_set_ui(num, 22);
    mpz_set_ui(den, 7);

    pi::NewtonDivider::divide(result, num, den, 20);

    // Check first few digits
    mp_exp_t exp;
    char* str = mpf_get_str(nullptr, &exp, 10, 20, result);
    std::string s(str);
    free(str);

    // Should start with "31428571428571428571" (22/7 repeating)
    EXPECT_EQ(exp, 1);  // Decimal point after first digit
    EXPECT_TRUE(s.substr(0, 7) == "3142857") << "Got: " << s;
}

TEST_F(NewtonDividerTest, DivideExact) {
    // 100 / 4 = 25.0
    mpz_set_ui(num, 100);
    mpz_set_ui(den, 4);

    pi::NewtonDivider::divide(result, num, den, 10);

    mpf_t expected;
    mpf_init_set_ui(expected, 25);
    EXPECT_EQ(mpf_cmp(result, expected), 0);
    mpf_clear(expected);
}

TEST_F(NewtonDividerTest, DivideByZeroThrows) {
    mpz_set_ui(num, 42);
    mpz_set_ui(den, 0);

    EXPECT_THROW(pi::NewtonDivider::divide(result, num, den, 10), std::invalid_argument);
}

TEST_F(NewtonDividerTest, DivideLargeNumbers) {
    // Large numerator / large denominator
    mpz_ui_pow_ui(num, 10, 100);
    mpz_set_ui(den, 3);

    pi::NewtonDivider::divide(result, num, den, 50);

    // Result should be 10^100 / 3 ≈ 3.333... × 10^99
    mp_exp_t exp;
    char* str = mpf_get_str(nullptr, &exp, 10, 10, result);
    std::string s(str);
    free(str);

    EXPECT_EQ(exp, 100);  // 10^99 magnitude
    EXPECT_TRUE(s.substr(0, 5) == "33333") << "Got: " << s;
}

TEST_F(NewtonDividerTest, SqrtOf4) {
    pi::NewtonDivider::sqrt_to_precision(result, 4, 20);

    mpf_t expected;
    mpf_init_set_ui(expected, 2);
    EXPECT_EQ(mpf_cmp(result, expected), 0);
    mpf_clear(expected);
}

TEST_F(NewtonDividerTest, SqrtOf2) {
    pi::NewtonDivider::sqrt_to_precision(result, 2, 50);

    // sqrt(2) ≈ 1.41421356237...
    mp_exp_t exp;
    char* str = mpf_get_str(nullptr, &exp, 10, 15, result);
    std::string s(str);
    free(str);

    EXPECT_EQ(exp, 1);
    EXPECT_TRUE(s.substr(0, 12) == "141421356237") << "Got: " << s;
}

TEST_F(NewtonDividerTest, SqrtOf10005) {
    // sqrt(10005) ≈ 100.02499687578...
    pi::NewtonDivider::sqrt_to_precision(result, 10005, 30);

    mp_exp_t exp;
    char* str = mpf_get_str(nullptr, &exp, 10, 15, result);
    std::string s(str);
    free(str);

    EXPECT_EQ(exp, 3);  // 100.xxx
    EXPECT_TRUE(s.substr(0, 8) == "10002499") << "Got: " << s;
}

TEST_F(NewtonDividerTest, DivisionPrecision100Digits) {
    // Verify we get at least 100 correct digits of 1/3
    mpz_set_ui(num, 1);
    mpz_set_ui(den, 3);

    pi::NewtonDivider::divide(result, num, den, 100);

    mp_exp_t exp;
    char* str = mpf_get_str(nullptr, &exp, 10, 100, result);
    std::string s(str);
    free(str);

    // 1/3 = 0.333333...
    EXPECT_EQ(exp, 0);
    for (size_t i = 0; i < std::min(s.size(), size_t(100)); ++i) {
        EXPECT_EQ(s[i], '3') << "Digit " << i << " is wrong";
    }
}
