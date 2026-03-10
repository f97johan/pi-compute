#pragma once

/**
 * @file newton_divider.h
 * @brief High-precision division and square root.
 *
 * Provides both GMP's built-in sqrt (single-threaded) and a parallel
 * Newton iteration sqrt that uses the Multiplier interface.
 */

#include <gmp.h>
#include <cstddef>
#include "../arithmetic/multiplier.h"

namespace pi {

class NewtonDivider {
public:
    /**
     * @brief Divide numerator by denominator using GMP's mpf_div.
     */
    static void divide(mpf_t result, const mpz_t numerator, const mpz_t denominator,
                       size_t precision_digits);

    /**
     * @brief Compute sqrt using GMP's mpf_sqrt (single-threaded).
     */
    static void sqrt_to_precision(mpf_t result, unsigned long n, size_t precision_digits);

    /**
     * @brief Compute sqrt using parallel Newton iteration.
     *
     * Uses the Multiplier interface for the large multiplications,
     * which enables multi-threaded GMP or GPU acceleration.
     *
     * Newton iteration for 1/sqrt(a):
     *   x_{n+1} = x_n * (3 - a * x_n^2) / 2
     *
     * Then sqrt(a) = a * (1/sqrt(a))
     *
     * Each iteration doubles the number of correct digits.
     *
     * @param result Output
     * @param n The number to take sqrt of
     * @param precision_digits Required precision
     * @param multiplier The multiplication backend (for parallel multiply)
     */
    static void parallel_sqrt(mpf_t result, unsigned long n, size_t precision_digits,
                               Multiplier& multiplier);
};

} // namespace pi
