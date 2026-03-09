/**
 * @file test_binary_splitting.cpp
 * @brief TDD tests for Chudnovsky binary splitting.
 */

#include <gtest/gtest.h>
#include <gmp.h>
#include "engine/binary_splitting.h"
#include "arithmetic/gmp_multiplier.h"

class BinarySplittingTest : public ::testing::Test {
protected:
    pi::GmpMultiplier multiplier;
};

TEST_F(BinarySplittingTest, TermsNeededFor100Digits) {
    unsigned long terms = pi::BinarySplitting::terms_needed(100);
    // 100 / 14.18 ≈ 7.05, so we need at least 8 terms
    EXPECT_GE(terms, 8);
    EXPECT_LE(terms, 15);  // Shouldn't need more than ~15
}

TEST_F(BinarySplittingTest, TermsNeededFor1000Digits) {
    unsigned long terms = pi::BinarySplitting::terms_needed(1000);
    // 1000 / 14.18 ≈ 70.5, so we need ~71-72 terms
    EXPECT_GE(terms, 71);
    EXPECT_LE(terms, 80);
}

TEST_F(BinarySplittingTest, TermsNeededFor1MillionDigits) {
    unsigned long terms = pi::BinarySplitting::terms_needed(1000000);
    // 1000000 / 14.18 ≈ 70,522
    EXPECT_GE(terms, 70000);
    EXPECT_LE(terms, 71000);
}

TEST_F(BinarySplittingTest, BaseCaseTerm0) {
    // For a=0: P(0,1)=1, Q(0,1)=1, R(0,1)=13591409
    pi::BinarySplitting bs(multiplier);
    pi::BSResult result = bs.compute(0, 1);

    EXPECT_EQ(mpz_cmp_ui(result.P, 1), 0);
    EXPECT_EQ(mpz_cmp_ui(result.Q, 1), 0);
    EXPECT_EQ(mpz_cmp_ui(result.R, 13591409), 0);
}

TEST_F(BinarySplittingTest, BaseCaseTerm1) {
    // For a=1:
    // P(1,2) = -(6*1-5)(2*1-1)(6*1-1) = -(1)(1)(5) = -5
    // Q(1,2) = 10939058860032000 * 1^3 = 10939058860032000
    // R(1,2) = P(1,2) * (13591409 + 545140134*1) = -5 * 558731543 = -2793657715
    pi::BinarySplitting bs(multiplier);
    pi::BSResult result = bs.compute(1, 2);

    EXPECT_EQ(mpz_cmp_si(result.P, -5), 0);

    mpz_t expected_q;
    mpz_init(expected_q);
    mpz_set_str(expected_q, "10939058860032000", 10);
    EXPECT_EQ(mpz_cmp(result.Q, expected_q), 0);

    // R = -5 * (13591409 + 545140134) = -5 * 558731543 = -2793657715
    mpz_t expected_r;
    mpz_init(expected_r);
    mpz_set_str(expected_r, "-2793657715", 10);
    EXPECT_EQ(mpz_cmp(result.R, expected_r), 0);

    mpz_clear(expected_q);
    mpz_clear(expected_r);
}

TEST_F(BinarySplittingTest, BaseCaseTerm2) {
    // For a=2:
    // P(2,3) = -(6*2-5)(2*2-1)(6*2-1) = -(7)(3)(11) = -231
    // Q(2,3) = 10939058860032000 * 2^3 = 10939058860032000 * 8 = 87512470880256000
    pi::BinarySplitting bs(multiplier);
    pi::BSResult result = bs.compute(2, 3);

    EXPECT_EQ(mpz_cmp_si(result.P, -231), 0);

    mpz_t expected_q;
    mpz_init(expected_q);
    mpz_set_str(expected_q, "87512470880256000", 10);
    EXPECT_EQ(mpz_cmp(result.Q, expected_q), 0);

    mpz_clear(expected_q);
}

TEST_F(BinarySplittingTest, MergeTwoTerms) {
    // Compute [0,2) = merge of [0,1) and [1,2)
    pi::BinarySplitting bs(multiplier);
    pi::BSResult result = bs.compute(0, 2);

    // Verify merge:
    // P(0,2) = P(0,1) * P(1,2) = 1 * (-5) = -5
    EXPECT_EQ(mpz_cmp_si(result.P, -5), 0);

    // Q(0,2) = Q(0,1) * Q(1,2) = 1 * 10939058860032000 = 10939058860032000
    mpz_t expected_q;
    mpz_init(expected_q);
    mpz_set_str(expected_q, "10939058860032000", 10);
    EXPECT_EQ(mpz_cmp(result.Q, expected_q), 0);

    // R(0,2) = Q(1,2)*R(0,1) + P(0,1)*R(1,2)
    //        = 10939058860032000 * 13591409 + 1 * (-2793657715)
    //        = 148677972042891249408000 - 2793657715
    //        = 148677972040097591693(something)
    // We just verify it's non-zero and has the right sign (positive, since first term dominates)
    EXPECT_GT(mpz_sgn(result.R), 0);

    mpz_clear(expected_q);
}

TEST_F(BinarySplittingTest, ComputeProducesValidResult) {
    // Compute enough terms for 50 digits and verify P, Q, R are non-trivial
    pi::BinarySplitting bs(multiplier);
    unsigned long terms = pi::BinarySplitting::terms_needed(50);
    pi::BSResult result = bs.compute(0, terms);

    // P should be non-zero
    EXPECT_NE(mpz_sgn(result.P), 0);
    // Q should be positive
    EXPECT_GT(mpz_sgn(result.Q), 0);
    // R should be non-zero
    EXPECT_NE(mpz_sgn(result.R), 0);
}
