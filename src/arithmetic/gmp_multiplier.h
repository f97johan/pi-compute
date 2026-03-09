#pragma once

/**
 * @file gmp_multiplier.h
 * @brief CPU-based multiplication using GMP's optimized routines.
 *
 * GMP automatically selects the best algorithm based on operand size:
 * schoolbook -> Karatsuba -> Toom-Cook -> FFT.
 */

#include "multiplier.h"

namespace pi {

class GmpMultiplier : public Multiplier {
public:
    GmpMultiplier() = default;
    ~GmpMultiplier() override = default;

    void multiply(mpz_t result, const mpz_t a, const mpz_t b) override;
    void square(mpz_t result, const mpz_t a) override;
};

} // namespace pi
