/**
 * @file binary_splitting.cpp
 * @brief Chudnovsky binary splitting implementation.
 */

#include "binary_splitting.h"
#include <cmath>
#include <cassert>

namespace pi {

// --- BSResult ---

BSResult::BSResult() {
    mpz_init(P);
    mpz_init(Q);
    mpz_init(R);
}

BSResult::~BSResult() {
    mpz_clear(P);
    mpz_clear(Q);
    mpz_clear(R);
}

BSResult::BSResult(BSResult&& other) noexcept {
    mpz_init(P); mpz_init(Q); mpz_init(R);
    mpz_swap(P, other.P);
    mpz_swap(Q, other.Q);
    mpz_swap(R, other.R);
}

BSResult& BSResult::operator=(BSResult&& other) noexcept {
    if (this != &other) {
        mpz_swap(P, other.P);
        mpz_swap(Q, other.Q);
        mpz_swap(R, other.R);
    }
    return *this;
}

// --- BinarySplitting ---

// Chudnovsky constant: 640320^3 / 24 = 10939058860032000
static const char* C3_OVER_24 = "10939058860032000";

BinarySplitting::BinarySplitting(Multiplier& multiplier)
    : multiplier_(multiplier) {}

unsigned long BinarySplitting::terms_needed(size_t digits) {
    // Each term of the Chudnovsky series gives ~14.1816474627... digits
    // digits_per_term = log10(640320^3 / (6*2*6)) ≈ 14.1816...
    // We add a small margin to ensure we have enough terms.
    return static_cast<unsigned long>(std::ceil(static_cast<double>(digits) / 14.1816474627)) + 1;
}

BSResult BinarySplitting::base_case(unsigned long a) {
    BSResult result;

    if (a == 0) {
        // Special case: first term
        // P(0,1) = 1
        // Q(0,1) = 1
        // R(0,1) = P(0,1) * (13591409 + 545140134*0) = 13591409
        mpz_set_ui(result.P, 1);
        mpz_set_ui(result.Q, 1);
        mpz_set_ui(result.R, 13591409);
    } else {
        // General case:
        // P(a, a+1) = -(6a-5)(2a-1)(6a-1)
        // Q(a, a+1) = 10939058860032000 * a^3
        // R(a, a+1) = P(a, a+1) * (13591409 + 545140134*a)

        // Compute P = -(6a-5)(2a-1)(6a-1)
        mpz_t t1, t2, t3;
        mpz_init(t1); mpz_init(t2); mpz_init(t3);

        mpz_set_ui(t1, 6 * a - 5);
        mpz_set_ui(t2, 2 * a - 1);
        mpz_set_ui(t3, 6 * a - 1);

        mpz_mul(result.P, t1, t2);
        mpz_mul(result.P, result.P, t3);
        mpz_neg(result.P, result.P);

        // Compute Q = 10939058860032000 * a^3
        mpz_set_str(result.Q, C3_OVER_24, 10);
        mpz_t a_cubed;
        mpz_init(a_cubed);
        mpz_set_ui(a_cubed, a);
        mpz_pow_ui(a_cubed, a_cubed, 3);
        mpz_mul(result.Q, result.Q, a_cubed);

        // Compute R = P * (13591409 + 545140134 * a)
        mpz_t linear;
        mpz_init(linear);
        mpz_set_ui(linear, 545140134);
        mpz_mul_ui(linear, linear, a);
        mpz_add_ui(linear, linear, 13591409);

        mpz_mul(result.R, result.P, linear);

        mpz_clear(t1); mpz_clear(t2); mpz_clear(t3);
        mpz_clear(a_cubed);
        mpz_clear(linear);
    }

    return result;
}

BSResult BinarySplitting::merge(BSResult& left, BSResult& right) {
    BSResult result;

    // R(a,b) = Q(m,b) * R(a,m) + P(a,m) * R(m,b)
    mpz_t temp1, temp2;
    mpz_init(temp1);
    mpz_init(temp2);

    multiplier_.multiply(temp1, right.Q, left.R);   // Q(m,b) * R(a,m)
    multiplier_.multiply(temp2, left.P, right.R);    // P(a,m) * R(m,b)
    mpz_add(result.R, temp1, temp2);

    // P(a,b) = P(a,m) * P(m,b)
    multiplier_.multiply(result.P, left.P, right.P);

    // Q(a,b) = Q(a,m) * Q(m,b)
    multiplier_.multiply(result.Q, left.Q, right.Q);

    mpz_clear(temp1);
    mpz_clear(temp2);

    return result;
}

BSResult BinarySplitting::compute(unsigned long a, unsigned long b) {
    assert(b > a);

    if (b - a == 1) {
        return base_case(a);
    }

    unsigned long m = a + (b - a) / 2;

    BSResult left = compute(a, m);
    BSResult right = compute(m, b);

    return merge(left, right);
}

} // namespace pi
