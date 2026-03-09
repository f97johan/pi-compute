#pragma once

/**
 * @file multiplier.h
 * @brief Abstract interface for arbitrary-precision integer multiplication.
 *
 * Strategy pattern: allows swapping between CPU (GMP) and GPU (NTT) multiplication
 * without changing the binary splitting algorithm code.
 */

#include <gmp.h>
#include <memory>

namespace pi {

class Multiplier {
public:
    virtual ~Multiplier() = default;

    /**
     * @brief Multiply two arbitrary-precision integers: result = a * b
     * @param result Output: product of a and b (must be initialized)
     * @param a First operand
     * @param b Second operand
     */
    virtual void multiply(mpz_t result, const mpz_t a, const mpz_t b) = 0;

    /**
     * @brief Square an arbitrary-precision integer: result = a * a
     * @param result Output: square of a (must be initialized)
     * @param a Operand to square
     *
     * Default implementation calls multiply(result, a, a).
     * Subclasses may override for optimized squaring.
     */
    virtual void square(mpz_t result, const mpz_t a) {
        multiply(result, a, a);
    }
};

using MultiplierPtr = std::unique_ptr<Multiplier>;

} // namespace pi
