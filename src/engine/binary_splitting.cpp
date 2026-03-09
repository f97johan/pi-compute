/**
 * @file binary_splitting.cpp
 * @brief Chudnovsky binary splitting implementation with multi-threading.
 *
 * Threading strategy:
 * - At the top levels of the recursion tree, left and right halves are
 *   computed in parallel using std::thread.
 * - Below a certain depth (log2(num_threads)), computation is sequential
 *   to avoid excessive thread creation overhead.
 * - Each thread works on independent GMP integers, so no locking is needed.
 * - The merge step (which involves multiplications) is always sequential
 *   within a single thread.
 */

#include "binary_splitting.h"
#include <cmath>
#include <cassert>
#include <future>
#include <algorithm>

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

BinarySplitting::BinarySplitting(Multiplier& multiplier, unsigned int num_threads)
    : multiplier_(multiplier),
      num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) {
    // Ensure at least 1 thread
    if (num_threads_ == 0) num_threads_ = 1;
}

unsigned long BinarySplitting::terms_needed(size_t digits) {
    // Each term of the Chudnovsky series gives ~14.1816474627... digits
    return static_cast<unsigned long>(std::ceil(static_cast<double>(digits) / 14.1816474627)) + 1;
}

BSResult BinarySplitting::base_case(unsigned long a) {
    BSResult result;

    if (a == 0) {
        // Special case: first term
        mpz_set_ui(result.P, 1);
        mpz_set_ui(result.Q, 1);
        mpz_set_ui(result.R, 13591409);
    } else {
        // General case:
        // P(a, a+1) = -(6a-5)(2a-1)(6a-1)
        // Q(a, a+1) = 10939058860032000 * a^3
        // R(a, a+1) = P(a, a+1) * (13591409 + 545140134*a)

        mpz_t t1, t2, t3;
        mpz_init(t1); mpz_init(t2); mpz_init(t3);

        mpz_set_ui(t1, 6 * a - 5);
        mpz_set_ui(t2, 2 * a - 1);
        mpz_set_ui(t3, 6 * a - 1);

        mpz_mul(result.P, t1, t2);
        mpz_mul(result.P, result.P, t3);
        mpz_neg(result.P, result.P);

        // Q = 10939058860032000 * a^3
        mpz_set_str(result.Q, C3_OVER_24, 10);
        mpz_t a_cubed;
        mpz_init(a_cubed);
        mpz_set_ui(a_cubed, a);
        mpz_pow_ui(a_cubed, a_cubed, 3);
        mpz_mul(result.Q, result.Q, a_cubed);

        // R = P * (13591409 + 545140134 * a)
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

BSResult BinarySplitting::compute_sequential(unsigned long a, unsigned long b) {
    assert(b > a);

    if (b - a == 1) {
        return base_case(a);
    }

    unsigned long m = a + (b - a) / 2;

    BSResult left = compute_sequential(a, m);
    BSResult right = compute_sequential(m, b);

    return merge(left, right);
}

BSResult BinarySplitting::compute_parallel(unsigned long a, unsigned long b, int depth) {
    assert(b > a);

    if (b - a == 1) {
        return base_case(a);
    }

    unsigned long m = a + (b - a) / 2;

    // Only parallelize if:
    // 1. We haven't exceeded the thread depth (2^depth <= num_threads)
    // 2. The range is large enough to justify thread overhead
    bool should_parallelize = (depth > 0) && (b - a >= PARALLEL_THRESHOLD);

    if (should_parallelize) {
        // Compute right half in a separate thread
        // Left half computed in current thread
        std::future<BSResult> right_future = std::async(
            std::launch::async,
            [this, m, b, depth]() {
                return compute_parallel(m, b, depth - 1);
            }
        );

        BSResult left = compute_parallel(a, m, depth - 1);
        BSResult right = right_future.get();

        return merge(left, right);
    } else {
        // Sequential: range too small or enough threads spawned
        return compute_sequential(a, b);
    }
}

BSResult BinarySplitting::compute(unsigned long a, unsigned long b) {
    assert(b > a);

    if (num_threads_ <= 1) {
        return compute_sequential(a, b);
    }

    // Calculate max parallel depth: log2(num_threads)
    // At depth D, we spawn up to 2^D threads
    int max_depth = 0;
    unsigned int t = num_threads_;
    while (t > 1) {
        max_depth++;
        t >>= 1;
    }

    return compute_parallel(a, b, max_depth);
}

} // namespace pi
