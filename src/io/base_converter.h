#pragma once

/**
 * @file base_converter.h
 * @brief Convert GMP numbers to decimal digit strings.
 */

#include <gmp.h>
#include <string>
#include <cstddef>

namespace pi {

class BaseConverter {
public:
    /**
     * @brief Convert an mpf floating-point value to a decimal string.
     * @param value The GMP floating-point value
     * @param digits Number of decimal digits to extract
     * @return String of decimal digits (e.g., "3.14159265...")
     */
    static std::string to_decimal_string(const mpf_t value, size_t digits);

    /**
     * @brief Convert an mpz integer to a decimal string.
     * Uses GMP's mpz_get_str which is already subquadratic in GMP 6.x.
     * @param n The integer to convert
     * @param min_digits Minimum number of digits (left-padded with zeros)
     * @return Decimal string representation
     */
    static std::string fast_integer_to_decimal(const mpz_t n, size_t min_digits = 0);
};

} // namespace pi
