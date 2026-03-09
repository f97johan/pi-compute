/**
 * @file base_converter.cpp
 * @brief Decimal string conversion using GMP's mpf_get_str.
 *
 * After profiling, GMP's mpf_get_str is actually well-optimized for
 * extracting decimal digits from mpf_t. Our divide-and-conquer approach
 * was slower because computing 10^N and multiplying is expensive.
 *
 * The real bottleneck at 100M digits is the mpf_get_str call itself,
 * which is O(n·M(n)) where M(n) is the multiplication time. GMP uses
 * its own divide-and-conquer internally for this.
 */

#include "base_converter.h"
#include <cstdlib>
#include <stdexcept>

namespace pi {

std::string BaseConverter::to_decimal_string(const mpf_t value, size_t digits) {
    if (digits == 0) {
        throw std::invalid_argument("digits must be > 0");
    }

    // mpf_get_str returns digits without decimal point
    // exp receives the exponent (position of decimal point)
    mp_exp_t exp;
    char* raw = mpf_get_str(nullptr, &exp, 10, digits + 1, value);

    if (!raw) {
        throw std::runtime_error("mpf_get_str failed");
    }

    std::string result(raw);
    free(raw);  // mpf_get_str allocates with malloc

    // For pi, exp should be 1 (the "3" before the decimal point)
    // raw will be like "314159265..." and exp=1
    // We want to insert the decimal point after the first digit
    if (result.empty()) {
        return "0";
    }

    // Handle negative sign
    size_t start = 0;
    std::string prefix;
    if (result[0] == '-') {
        prefix = "-";
        start = 1;
    }

    if (exp <= 0) {
        // Number is like 0.000...digits
        std::string decimal = prefix + "0.";
        for (mp_exp_t i = 0; i < -exp; ++i) {
            decimal += '0';
        }
        decimal += result.substr(start);
        return decimal;
    }

    // Insert decimal point at position exp
    size_t decimal_pos = static_cast<size_t>(exp) + start;
    if (decimal_pos >= result.size()) {
        // No decimal point needed (integer)
        return result;
    }

    std::string formatted = prefix + result.substr(start, static_cast<size_t>(exp))
                          + "." + result.substr(decimal_pos);

    // Trim to requested number of digits after decimal point
    size_t dot_pos = formatted.find('.');
    if (dot_pos != std::string::npos && formatted.size() > dot_pos + 1 + digits) {
        formatted = formatted.substr(0, dot_pos + 1 + digits);
    }

    return formatted;
}

std::string BaseConverter::fast_integer_to_decimal(const mpz_t n, size_t min_digits) {
    if (mpz_sgn(n) == 0) {
        if (min_digits > 0) return std::string(min_digits, '0');
        return "0";
    }

    char* str = mpz_get_str(nullptr, 10, n);
    std::string result(str);
    free(str);

    if (result.size() < min_digits) {
        result = std::string(min_digits - result.size(), '0') + result;
    }

    return result;
}

} // namespace pi
