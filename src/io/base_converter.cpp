/**
 * @file base_converter.cpp
 * @brief Decimal string conversion using GMP's optimized routines.
 *
 * GMP 6.x already implements subquadratic base conversion internally
 * for large numbers. Our divide-and-conquer attempt was slower due to
 * overhead. Using GMP's mpz_get_str / mpf_get_str directly.
 */

#include "base_converter.h"
#include <cstdlib>
#include <stdexcept>
#include <cstring>

namespace pi {

std::string BaseConverter::to_decimal_string(const mpf_t value, size_t digits) {
    if (digits == 0) {
        throw std::invalid_argument("digits must be > 0");
    }

    mp_exp_t exp;
    char* raw = mpf_get_str(nullptr, &exp, 10, digits + 1, value);
    if (!raw) throw std::runtime_error("mpf_get_str failed");

    std::string result(raw);
    free(raw);

    if (result.empty()) return "0";

    size_t start = 0;
    std::string prefix;
    if (result[0] == '-') { prefix = "-"; start = 1; }

    if (exp <= 0) {
        std::string decimal = prefix + "0.";
        for (mp_exp_t i = 0; i < -exp; ++i) decimal += '0';
        decimal += result.substr(start);
        return decimal;
    }

    size_t decimal_pos = static_cast<size_t>(exp) + start;
    if (decimal_pos >= result.size()) return result;

    std::string formatted = prefix + result.substr(start, static_cast<size_t>(exp))
                          + "." + result.substr(decimal_pos);

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
