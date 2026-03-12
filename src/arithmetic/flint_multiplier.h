#pragma once

/**
 * @file flint_multiplier.h
 * @brief Multi-threaded multiplication using FLINT library.
 *
 * FLINT's fmpz_mul uses a multi-threaded FFT for large operands,
 * utilizing all available cores for a single multiplication.
 * This gives much better CPU utilization than GMP's single-threaded mpz_mul.
 *
 * For small operands (below a configurable threshold), falls back to
 * GMP to avoid the fmpz conversion overhead.
 */

#include "multiplier.h"
#include <gmp.h>

namespace pi {

class FlintMultiplier : public Multiplier {
public:
    /**
     * @param num_threads Number of threads for FLINT (0 = auto-detect)
     * @param threshold_limbs Min operand size (in GMP limbs) to use FLINT.
     *                        Below this, GMP is used directly. Default: 10000
     *                        (~80K digits). Set to 0 to always use FLINT.
     */
    explicit FlintMultiplier(unsigned int num_threads = 0, size_t threshold_limbs = 10000);
    ~FlintMultiplier() override = default;

    void multiply(mpz_t result, const mpz_t a, const mpz_t b) override;
    void square(mpz_t result, const mpz_t a) override;

private:
    size_t threshold_limbs_;
    unsigned int num_threads_;
};

} // namespace pi
