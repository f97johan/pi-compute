#pragma once

/**
 * @file base_converter.h
 * @brief Convert GMP floating-point numbers to decimal digit strings.
 *
 * Supports two modes:
 * - GMP's mpf_get_str (simple, O(n²) for n digits)
 * - Divide-and-conquer (fast, O(n·log(n)²) using GMP's fast multiplication)
 *
 * The divide-and-conquer method is dramatically faster for large digit counts
 * (100M+ digits) where mpf_get_str becomes the bottleneck.
 */

#include <gmp.h>
#include <string>
#include <cstddef>

namespace pi {

class BaseConverter {
public:
    /**
     * @brief Convert an mpf floating-point value to a decimal string.
     * @param value The GMP floating-point value (should be pi)
     * @param digits Number of decimal digits to extract (after the "3.")
     * @return String of decimal digits (e.g., "3.14159265...")
     */
    static std::string to_decimal_string(const mpf_t value, size_t digits);

    /**
     * @brief Fast divide-and-conquer conversion of mpz integer to decimal string.
     *
     * Converts an integer to its decimal representation in O(n·log(n)²) time
     * instead of GMP's default O(n²) mpz_get_str.
     *
     * @param n The integer to convert
     * @param min_digits Minimum number of digits (left-padded with zeros)
     * @return Decimal string representation
     */
    static std::string fast_integer_to_decimal(const mpz_t n, size_t min_digits = 0);

private:
    // Implementation uses free functions with PowerTree in the .cpp file
};

} // namespace pi
