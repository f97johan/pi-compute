/**
 * @file base_converter.cpp
 * @brief Subquadratic decimal conversion with precomputed power-of-10 tree.
 *
 * Standard mpz_get_str is O(n²) for n-digit numbers. This implementation
 * uses divide-and-conquer with a precomputed tree of powers of 10,
 * achieving O(n·log(n)²) complexity for the fast_integer_to_decimal path.
 */

#include "base_converter.h"
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <memory>

namespace pi {

// Precomputed power-of-10 tree for fast base conversion
// powers[i] = 10^(2^i), digit_counts[i] = 2^i
struct PowerTree {
    struct Entry {
        mpz_t value;
        size_t digit_count;

        Entry(size_t dc) : digit_count(dc) { mpz_init(value); }
        ~Entry() { mpz_clear(value); }

        // Non-copyable, non-movable (mpz_t is tricky)
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;
    };

    std::vector<std::unique_ptr<Entry>> entries;

    void build(size_t max_digits) {
        size_t count = 1;
        while (count < max_digits) {
            auto entry = std::make_unique<Entry>(count);

            if (entries.empty()) {
                mpz_set_ui(entry->value, 10);  // 10^1
            } else {
                // 10^(2^i) = (10^(2^(i-1)))^2
                mpz_mul(entry->value, entries.back()->value, entries.back()->value);
            }

            entries.push_back(std::move(entry));
            count *= 2;
        }
    }

    size_t size() const { return entries.size(); }
    const mpz_t& power(int i) const { return entries[i]->value; }
    size_t digits(int i) const { return entries[i]->digit_count; }
};

// Recursive divide-and-conquer conversion
static void dc_convert(const mpz_t n, size_t num_digits,
                        const PowerTree& tree, int level,
                        std::string& result) {
    // Base case: small enough for GMP's mpz_get_str
    if (num_digits <= 64 || level < 0) {
        char* str = mpz_get_str(nullptr, 10, n);
        size_t len = strlen(str);
        if (len < num_digits) {
            result.append(num_digits - len, '0');
        }
        result.append(str);
        free(str);
        return;
    }

    // Find the right level
    while (level >= 0 && tree.digits(level) >= num_digits) {
        level--;
    }

    if (level < 0) {
        char* str = mpz_get_str(nullptr, 10, n);
        size_t len = strlen(str);
        if (len < num_digits) {
            result.append(num_digits - len, '0');
        }
        result.append(str);
        free(str);
        return;
    }

    size_t low_digits = tree.digits(level);
    size_t high_digits = num_digits - low_digits;

    mpz_t q, r;
    mpz_init(q);
    mpz_init(r);
    mpz_tdiv_qr(q, r, n, tree.power(level));

    dc_convert(q, high_digits, tree, level - 1, result);
    dc_convert(r, low_digits, tree, level - 1, result);

    mpz_clear(q);
    mpz_clear(r);
}

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

    size_t num_digits = mpz_sizeinbase(n, 10);

    // For small numbers, use GMP directly
    if (num_digits <= 10000) {
        char* str = mpz_get_str(nullptr, 10, n);
        std::string result(str);
        free(str);
        if (result.size() < min_digits) {
            result = std::string(min_digits - result.size(), '0') + result;
        }
        return result;
    }

    // Build precomputed power-of-10 tree
    PowerTree tree;
    tree.build(num_digits);

    // Convert using divide-and-conquer
    std::string result;
    result.reserve(num_digits + 1);

    int top_level = static_cast<int>(tree.size()) - 1;
    dc_convert(n, num_digits, tree, top_level, result);

    if (result.size() < min_digits) {
        result = std::string(min_digits - result.size(), '0') + result;
    }

    return result;
}

} // namespace pi
