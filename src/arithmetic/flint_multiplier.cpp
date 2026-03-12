/**
 * @file flint_multiplier.cpp
 * @brief Multi-threaded multiplication using FLINT library.
 *
 * Converts GMP mpz_t to FLINT fmpz_t, performs multi-threaded multiplication,
 * and converts back. Falls back to GMP for small operands.
 */

#include "flint_multiplier.h"
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <thread>
#include <algorithm>

namespace pi {

FlintMultiplier::FlintMultiplier(unsigned int num_threads, size_t threshold_limbs)
    : threshold_limbs_(threshold_limbs),
      num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) {
    if (num_threads_ == 0) num_threads_ = 1;
    flint_set_num_threads(static_cast<int>(num_threads_));
}

void FlintMultiplier::multiply(mpz_t result, const mpz_t a, const mpz_t b) {
    size_t max_size = std::max(mpz_size(a), mpz_size(b));

    // For small operands, GMP is faster (no conversion overhead)
    if (max_size < threshold_limbs_) {
        mpz_mul(result, a, b);
        return;
    }

    // Convert to FLINT fmpz_t
    fmpz_t fa, fb, fr;
    fmpz_init(fa);
    fmpz_init(fb);
    fmpz_init(fr);

    fmpz_set_mpz(fa, a);
    fmpz_set_mpz(fb, b);

    // Multi-threaded multiplication
    fmpz_mul(fr, fa, fb);

    // Convert back to GMP
    fmpz_get_mpz(result, fr);

    fmpz_clear(fa);
    fmpz_clear(fb);
    fmpz_clear(fr);
}

void FlintMultiplier::square(mpz_t result, const mpz_t a) {
    size_t size = mpz_size(a);

    // For small operands, GMP is faster
    if (size < threshold_limbs_) {
        mpz_mul(result, a, a);
        return;
    }

    // Convert to FLINT fmpz_t
    fmpz_t fa, fr;
    fmpz_init(fa);
    fmpz_init(fr);

    fmpz_set_mpz(fa, a);

    // Multi-threaded squaring (FLINT optimizes a*a)
    fmpz_mul(fr, fa, fa);

    // Convert back to GMP
    fmpz_get_mpz(result, fr);

    fmpz_clear(fa);
    fmpz_clear(fr);
}

} // namespace pi
