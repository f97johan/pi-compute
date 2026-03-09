#pragma once

/**
 * @file newton_divider.h
 * @brief High-precision division using Newton's method (iterative reciprocal).
 *
 * For the Chudnovsky algorithm, we need to compute:
 *   pi = (Q * C) / (12 * (13591409*Q + R))
 * where C = 426880 * sqrt(10005)
 *
 * This class provides division to arbitrary precision using GMP's mpf (floating-point)
 * or via Newton iteration with the Multiplier interface.
 */

#include <gmp.h>
#include <cstddef>

namespace pi {

class NewtonDivider {
public:
    /**
     * @brief Divide numerator by denominator to the specified number of decimal digits.
     * @param result Output: the quotient as an mpf (GMP floating-point)
     * @param numerator The numerator (mpz integer)
     * @param denominator The denominator (mpz integer)
     * @param precision_digits Number of decimal digits of precision required
     */
    static void divide(mpf_t result, const mpz_t numerator, const mpz_t denominator,
                       size_t precision_digits);

    /**
     * @brief Compute the integer square root: result = floor(sqrt(n) * 10^precision_digits)
     * @param result Output: scaled square root as mpz integer
     * @param n The number to take the square root of
     * @param precision_digits Number of decimal digits of precision
     */
    static void sqrt_to_precision(mpf_t result, unsigned long n, size_t precision_digits);
};

} // namespace pi
