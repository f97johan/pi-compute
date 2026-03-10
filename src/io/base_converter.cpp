/**
 * @file base_converter.cpp
 * @brief Parallel decimal conversion with precomputed power-of-10 tree.
 *
 * Algorithm:
 * 1. Precompute: 10^1, 10^2, 10^4, ..., 10^(2^k) by repeated squaring
 * 2. Split: divmod(n, 10^(2^k)) → high half, low half
 * 3. Recurse: convert each half (in parallel for large numbers)
 * 4. Concatenate with zero-padding
 *
 * The precomputation is O(n·log(n)) and done once.
 * Each recursion level does one O(n·log(n)) division.
 * Total: O(n·log²(n)) — same as GMP, but parallelized.
 */

#include "base_converter.h"
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <memory>
#include <future>
#include <thread>
#include <algorithm>
#include <iostream>
#include <mutex>

namespace pi {

// ============================================================
// Power-of-10 tree (precomputed once, reused across recursion)
// ============================================================

struct PowerEntry {
    mpz_t value;       // 10^(2^level)
    size_t num_digits; // 2^level

    PowerEntry() { mpz_init(value); }
    ~PowerEntry() { mpz_clear(value); }
    PowerEntry(const PowerEntry&) = delete;
    PowerEntry& operator=(const PowerEntry&) = delete;
};

struct PowerTree {
    std::vector<std::unique_ptr<PowerEntry>> entries;

    // Build powers: 10^1, 10^2, 10^4, ..., 10^(2^k) until >= max_digits
    void build(size_t max_digits) {
        size_t d = 1;
        while (d < max_digits) {
            auto e = std::make_unique<PowerEntry>();
            e->num_digits = d;

            if (entries.empty()) {
                mpz_set_ui(e->value, 10);  // 10^1
            } else {
                mpz_mul(e->value, entries.back()->value, entries.back()->value);  // square
            }

            entries.push_back(std::move(e));
            d *= 2;
        }
    }

    int size() const { return static_cast<int>(entries.size()); }
    const mpz_t& power(int i) const { return entries[i]->value; }
    size_t digits(int i) const { return entries[i]->num_digits; }
};

// ============================================================
// Parallel recursive conversion
// ============================================================

// Thread-safe string assembly: each chunk writes to its position
struct StringAssembler {
    std::string result;
    std::mutex mutex;

    void reserve(size_t n) { result.resize(n, '0'); }

    void write_at(size_t offset, const char* data, size_t len) {
        // Direct write — no lock needed if ranges don't overlap
        memcpy(&result[offset], data, len);
    }
};

static void parallel_convert(const mpz_t n, size_t num_digits, size_t offset,
                              const PowerTree& tree, int level,
                              StringAssembler& output, int depth_remaining) {
    // Base case: small enough for GMP
    if (num_digits <= 256 || level < 0) {
        char* str = mpz_get_str(nullptr, 10, n);
        size_t len = strlen(str);

        // Write with zero-padding at the correct offset
        size_t pad = (len < num_digits) ? num_digits - len : 0;
        // Zeros are already in place (StringAssembler initialized with '0')
        output.write_at(offset + pad, str, len);
        free(str);
        return;
    }

    // Find the right split level
    while (level >= 0 && tree.digits(level) >= num_digits) {
        level--;
    }
    if (level < 0) {
        // Fallback to base case
        char* str = mpz_get_str(nullptr, 10, n);
        size_t len = strlen(str);
        size_t pad = (len < num_digits) ? num_digits - len : 0;
        output.write_at(offset + pad, str, len);
        free(str);
        return;
    }

    size_t low_digits = tree.digits(level);
    size_t high_digits = num_digits - low_digits;

    // Split: q = n / 10^(2^level), r = n % 10^(2^level)
    mpz_t q, r;
    mpz_init(q);
    mpz_init(r);
    mpz_tdiv_qr(q, r, n, tree.power(level));

    // Parallelize if we have depth remaining
    if (depth_remaining > 0 && num_digits > 10000) {
        auto future = std::async(std::launch::async, [&]() {
            parallel_convert(r, low_digits, offset + high_digits,
                           tree, level - 1, output, depth_remaining - 1);
        });

        parallel_convert(q, high_digits, offset,
                        tree, level - 1, output, depth_remaining - 1);

        future.get();
    } else {
        // Sequential
        parallel_convert(q, high_digits, offset,
                        tree, level - 1, output, 0);
        parallel_convert(r, low_digits, offset + high_digits,
                        tree, level - 1, output, 0);
    }

    mpz_clear(q);
    mpz_clear(r);
}

// ============================================================
// Public API
// ============================================================

std::string BaseConverter::to_decimal_string(const mpf_t value, size_t digits) {
    if (digits == 0) throw std::invalid_argument("digits must be > 0");

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
    if (dot_pos != std::string::npos && formatted.size() > dot_pos + 1 + digits)
        formatted = formatted.substr(0, dot_pos + 1 + digits);

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
    if (result.size() < min_digits)
        result = std::string(min_digits - result.size(), '0') + result;
    return result;
}

std::string BaseConverter::parallel_to_decimal(
    const mpz_t n, unsigned int num_threads,
    std::function<void(size_t, const std::string&)> chunk_callback) {

    if (mpz_sgn(n) == 0) return "0";

    size_t num_digits = mpz_sizeinbase(n, 10);

    // For small numbers, use GMP directly
    if (num_digits <= 100000) {
        return fast_integer_to_decimal(n);
    }

    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }

    // Step 1: Build power-of-10 tree (one-time cost)
    PowerTree tree;
    tree.build(num_digits);

    // Step 2: Prepare output buffer (pre-filled with '0' for zero-padding)
    StringAssembler output;
    output.reserve(num_digits);

    // Step 3: Parallel divide-and-conquer conversion
    int max_depth = 0;
    unsigned int t = num_threads;
    while (t > 1) { max_depth++; t >>= 1; }

    int top_level = tree.size() - 1;
    parallel_convert(n, num_digits, 0, tree, top_level, output, max_depth);

    // Step 4: If streaming callback provided, call it with chunks
    if (chunk_callback) {
        const size_t chunk_size = 1000000;  // 1M digits per chunk
        for (size_t i = 0; i < num_digits; i += chunk_size) {
            size_t len = std::min(chunk_size, num_digits - i);
            chunk_callback(i, output.result.substr(i, len));
        }
        return "";
    }

    return output.result;
}

} // namespace pi
