/**
 * @file gmp_multiplier.cpp
 * @brief GMP-based arbitrary-precision multiplication implementation.
 */

#include "gmp_multiplier.h"

namespace pi {

void GmpMultiplier::multiply(mpz_t result, const mpz_t a, const mpz_t b) {
    mpz_mul(result, a, b);
}

void GmpMultiplier::square(mpz_t result, const mpz_t a) {
    // GMP's mpz_mul is already optimized for squaring when both operands
    // point to the same data, but mpz_pow_ui(result, a, 2) or explicit
    // squaring could also be used. mpz_mul handles the a==b case internally.
    mpz_mul(result, a, a);
}

} // namespace pi
