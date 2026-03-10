#pragma once

/**
 * @file base_converter.h
 * @brief Fast parallel decimal conversion with precomputed power-of-10 tree.
 */

#include <gmp.h>
#include <string>
#include <cstddef>
#include <functional>

namespace pi {

class BaseConverter {
public:
    /**
     * @brief Convert mpf to decimal string (for backward compatibility).
     */
    static std::string to_decimal_string(const mpf_t value, size_t digits);

    /**
     * @brief Convert mpz integer to decimal string using GMP's mpz_get_str.
     */
    static std::string fast_integer_to_decimal(const mpz_t n, size_t min_digits = 0);

    /**
     * @brief Parallel divide-and-conquer decimal conversion with precomputed powers.
     *
     * Significantly faster than mpz_get_str for 100M+ digit numbers:
     * - Precomputes power-of-10 tree once (10^1, 10^2, 10^4, ...)
     * - Splits number at precomputed powers, converts halves in parallel
     * - Optionally streams output chunks via callback
     *
     * @param n The integer to convert
     * @param num_threads Number of threads (0 = auto)
     * @param chunk_callback Called with (offset, chunk_string) for streaming output.
     *                       If null, returns full string.
     * @return Full decimal string (or empty if using chunk_callback)
     */
    static std::string parallel_to_decimal(
        const mpz_t n,
        unsigned int num_threads = 0,
        std::function<void(size_t offset, const std::string& chunk)> chunk_callback = nullptr
    );
};

} // namespace pi
