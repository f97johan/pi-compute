/**
 * @file newton_divider.cpp
 * @brief Division and square root using GMP's arbitrary-precision floating-point.
 *
 * For Phase 1, we use GMP's mpf_div and mpf_sqrt directly. These are already
 * highly optimized. In Phase 2+, the GPU-accelerated path could use Newton
 * iteration with the Multiplier interface for the reciprocal computation.
 */

#include "newton_divider.h"
#include <stdexcept>

namespace pi {

void NewtonDivider::divide(mpf_t result, const mpz_t numerator, const mpz_t denominator,
                           size_t precision_digits) {
    if (mpz_sgn(denominator) == 0) {
        throw std::invalid_argument("Division by zero");
    }

    // Convert precision from decimal digits to GMP limb precision
    // GMP uses bits internally: digits * log2(10) ≈ digits * 3.3219
    mp_bitcnt_t precision_bits = static_cast<mp_bitcnt_t>(precision_digits * 3.3219281) + 64;

    mpf_t num_f, den_f;
    mpf_init2(num_f, precision_bits);
    mpf_init2(den_f, precision_bits);

    // Ensure result has sufficient precision
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

} // namespace pi
