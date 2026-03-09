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
 *   pi = (Q(0,N) * 426880 * sqrt(10005)) / R(0,N)
 */

#include <gmp.h>
#include <cstddef>
#include <thread>
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
     * @param num_threads Number of threads for parallel computation (0 = auto-detect)
     */
    explicit BinarySplitting(Multiplier& multiplier, unsigned int num_threads = 0);

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

    /**
     * @brief Get the number of threads being used.
     */
    unsigned int thread_count() const { return num_threads_; }

private:
    Multiplier& multiplier_;
    unsigned int num_threads_;

    /**
     * @brief Minimum range size to consider parallelizing.
     * Below this, sequential is faster (avoids thread overhead).
     */
    static constexpr unsigned long PARALLEL_THRESHOLD = 64;

    /**
     * @brief Compute the base case for a single term at index a.
     */
    BSResult base_case(unsigned long a);

    /**
     * @brief Merge two BSResults: left=[a,m) and right=[m,b).
     */
    BSResult merge(BSResult& left, BSResult& right);

    /**
     * @brief Sequential compute (no threading).
     */
    BSResult compute_sequential(unsigned long a, unsigned long b);

    /**
     * @brief Parallel compute using thread pool.
     * @param a Start index
     * @param b End index
     * @param depth Current recursion depth (limits thread spawning)
     */
    BSResult compute_parallel(unsigned long a, unsigned long b, int depth);
};

} // namespace pi
