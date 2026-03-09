#pragma once

/**
 * @file base_converter.h
 * @brief Convert GMP floating-point numbers to decimal digit strings.
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
};

} // namespace pi
