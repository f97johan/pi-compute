/**
 * @file newton_divider.cpp
 * @brief Division and square root with optional parallel Newton iteration.
 */

#include "newton_divider.h"
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace pi {

void NewtonDivider::divide(mpf_t result, const mpz_t numerator, const mpz_t denominator,
                           size_t precision_digits) {
    if (mpz_sgn(denominator) == 0) {
        throw std::invalid_argument("Division by zero");
    }

    mp_bitcnt_t precision_bits = static_cast<mp_bitcnt_t>(precision_digits * 3.3219281) + 64;

    mpf_t num_f, den_f;
    mpf_init2(num_f, precision_bits);
    mpf_init2(den_f, precision_bits);
    mpf_set_prec(result, precision_bits);

    mpf_set_z(num_f, numerator);
    mpf_set_z(den_f, denominator);
    mpf_div(result, num_f, den_f);

    mpf_clear(num_f);
    mpf_clear(den_f);
}

void NewtonDivider::sqrt_to_precision(mpf_t result, unsigned long n, size_t precision_digits) {
    mp_bitcnt_t precision_bits = static_cast<mp_bitcnt_t>(precision_digits * 3.3219281) + 64;
    mpf_set_prec(result, precision_bits);
    mpf_set_ui(result, n);
    mpf_sqrt(result, result);
}

void NewtonDivider::parallel_sqrt(mpf_t result, unsigned long n, size_t precision_digits,
                                   Multiplier& multiplier) {
    // For small precision, just use GMP's built-in sqrt
    if (precision_digits < 1000000) {
        sqrt_to_precision(result, n, precision_digits);
        return;
    }

    mp_bitcnt_t precision_bits = static_cast<mp_bitcnt_t>(precision_digits * 3.3219281) + 64;

    // Newton iteration for 1/sqrt(n):
    //   x_{n+1} = x_n * (3 - n * x_n^2) / 2
    //
    // We work in integer arithmetic scaled by 10^precision:
    //   X = x * 10^P (where P = precision_digits)
    //   X_{n+1} = X_n * (3 * 10^(2P) - n * X_n^2) / (2 * 10^P)
    //
    // Actually, it's simpler to use GMP's mpf for Newton iteration
    // but use our multiplier for the large mpz multiplications inside.
    //
    // For now, use GMP's mpf_sqrt which is already quite fast,
    // and the multiplier is used for the binary splitting phase.
    //
    // The parallel benefit of Newton sqrt is limited because:
    // 1. Only ~30 iterations needed (each doubles precision)
    // 2. Each iteration is dominated by one large multiplication
    // 3. GMP's mpf_sqrt already uses an efficient algorithm
    //
    // The real parallel benefit comes from the binary splitting and
    // string conversion, which we've already parallelized.

    // Use GMP's built-in sqrt (already well-optimized)
    sqrt_to_precision(result, n, precision_digits);
}

} // namespace pi
