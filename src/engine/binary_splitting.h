#pragma once

/**
 * @file binary_splitting.h
 * @brief Chudnovsky algorithm with binary splitting for computing pi.
 *
 * The Chudnovsky formula:
 *   1/pi = 12 * sum_{k=0}^{inf} (-1)^k * (6k)! * (13591409 + 545140134*k)
 *          / ((3k)! * (k!)^3 * 640320^(3k+3/2))
 *
 * Binary splitting converts this into computing three sequences P(a,b), Q(a,b), R(a,b):
 *   Base case (b = a+1):
 *     P(a,b) = -(6a-5)(2a-1)(6a-1)
 *     Q(a,b) = 10939058860032000 * a^3    [= 640320^3 / 24]
 *     R(a,b) = P(a,b) * (13591409 + 545140134*a)
 *   Special case a=0: Q(0,1) = 1, P(0,1) = 1
 *
 *   Merge step:
 *     P(a,b) = P(a,m) * P(m,b)
 *     Q(a,b) = Q(a,m) * Q(m,b)
 *     R(a,b) = Q(m,b) * R(a,m) + P(a,m) * R(m,b)
 *
 * Final computation:
 *   pi = (Q(0,N) * 426880 * sqrt(10005)) / (13591409 * Q(0,N) + R(0,N))
 */

#include <gmp.h>
#include <cstddef>
#include "../arithmetic/multiplier.h"

namespace pi {

/**
 * @brief Result of binary splitting: the three accumulated sequences P, Q, R.
 */
struct BSResult {
    mpz_t P;  ///< Product of numerator terms
    mpz_t Q;  ///< Product of denominator terms
    mpz_t R;  ///< Accumulated series sum (weighted)

    BSResult();
    ~BSResult();

    // Non-copyable, movable
    BSResult(const BSResult&) = delete;
    BSResult& operator=(const BSResult&) = delete;
    BSResult(BSResult&& other) noexcept;
    BSResult& operator=(BSResult&& other) noexcept;
};

class BinarySplitting {
public:
    /**
     * @brief Construct a binary splitting engine.
     * @param multiplier The multiplication strategy to use (CPU or GPU)
     */
    explicit BinarySplitting(Multiplier& multiplier);

    /**
     * @brief Compute the binary splitting for the range [a, b).
     * @param a Start index (inclusive)
     * @param b End index (exclusive)
     * @return BSResult containing P(a,b), Q(a,b), R(a,b)
     */
    BSResult compute(unsigned long a, unsigned long b);

    /**
     * @brief Determine how many series terms are needed for N decimal digits.
     * @param digits Number of decimal digits desired
     * @return Number of terms needed (each term gives ~14.18 digits)
     */
    static unsigned long terms_needed(size_t digits);

private:
    Multiplier& multiplier_;

    /**
     * @brief Compute the base case for a single term at index a.
     */
    BSResult base_case(unsigned long a);

    /**
     * @brief Merge two BSResults: left=[a,m) and right=[m,b).
     */
    BSResult merge(BSResult& left, BSResult& right);
};

} // namespace pi
