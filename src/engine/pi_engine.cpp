/**
 * @file pi_engine.cpp
 * @brief Pi computation orchestrator — pure integer arithmetic.
 *
 * Computes floor(pi * 10^N) as a pure integer, eliminating the expensive
 * mpf_get_str decimal conversion (which was 41% of runtime at 100M digits).
 *
 * Formula:
 *   pi = 426880 * sqrt(10005) * Q(0,N) / R(0,N)
 *
 * To get N decimal digits as integer:
 *   pi_digits = floor(426880 * isqrt(10005 * 10^(2*N+guard)) * Q / R / 10^guard)
 *
 * where isqrt(X) = floor(sqrt(X)) via GMP's mpz_sqrt (exact integer operation).
 *
 * The final result is an mpz_t integer that we convert to string with mpz_get_str,
 * which is O(n*log(n)) — much faster than mpf_get_str's O(n^2).
 */

#include "pi_engine.h"
#include "binary_splitting.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <gmp.h>

namespace pi {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

PiEngine::PiEngine(Multiplier& multiplier)
    : multiplier_(multiplier) {}

PiResult PiEngine::compute(const PiConfig& config) {
    auto total_start = Clock::now();

    if (config.verbose) {
        std::cout << "Computing " << config.digits << " digits of pi (integer mode)..." << std::endl;
    }

    // Guard digits to ensure the last requested digit is correct
    size_t guard_digits = 100;
    size_t precision = config.digits + guard_digits;

    // Step 1: Binary splitting
    unsigned long terms = BinarySplitting::terms_needed(precision);
    if (config.verbose) {
        std::cout << "  Terms needed: " << terms << std::endl;
    }

    auto bs_start = Clock::now();
    BinarySplitting bs(multiplier_);
    if (config.verbose) {
        std::cout << "  Threads: " << bs.thread_count() << std::endl;
    }
    BSResult bsr = bs.compute(0, terms);
    auto bs_end = Clock::now();
    double bs_time = Duration(bs_end - bs_start).count();

    if (config.verbose) {
        std::cout << "  Binary splitting: " << std::fixed << std::setprecision(3)
                  << bs_time << "s" << std::endl;
    }

    // Step 2: Compute pi as integer
    // pi_int = floor(426880 * isqrt(10005 * 10^(2*precision)) * Q / R)
    auto final_start = Clock::now();

    // Compute 10^(2*precision) — this is the scaling factor for sqrt
    auto pow_start = Clock::now();
    mpz_t scale;
    mpz_init(scale);
    mpz_ui_pow_ui(scale, 10, 2 * precision);
    auto pow_end = Clock::now();

    // Compute 10005 * 10^(2*precision)
    mpz_t sqrt_arg;
    mpz_init(sqrt_arg);
    mpz_mul_ui(sqrt_arg, scale, 10005);

    // isqrt(10005 * 10^(2*precision)) — this gives sqrt(10005) * 10^precision
    auto sqrt_start = Clock::now();
    mpz_t sqrt_val;
    mpz_init(sqrt_val);
    mpz_sqrt(sqrt_val, sqrt_arg);
    auto sqrt_end = Clock::now();
    double sqrt_time = Duration(sqrt_end - sqrt_start).count();

    // Compute numerator: 426880 * sqrt_val * Q
    auto mul_start = Clock::now();
    mpz_t numerator;
    mpz_init(numerator);
    mpz_mul_ui(numerator, sqrt_val, 426880);
    multiplier_.multiply(numerator, numerator, bsr.Q);

    // Compute pi_scaled = numerator / R
    // This gives pi * 10^precision (approximately)
    mpz_t pi_scaled;
    mpz_init(pi_scaled);
    mpz_tdiv_q(pi_scaled, numerator, bsr.R);

    // Remove guard digits: pi_int = pi_scaled / 10^guard_digits
    mpz_t guard_power;
    mpz_init(guard_power);
    mpz_ui_pow_ui(guard_power, 10, guard_digits);
    mpz_t pi_int;
    mpz_init(pi_int);
    mpz_tdiv_q(pi_int, pi_scaled, guard_power);
    auto mul_end = Clock::now();
    double mul_time = Duration(mul_end - mul_start).count();

    double final_time = Duration(mul_end - final_start).count();
    double pow_time = Duration(pow_end - pow_start).count();

    if (config.verbose) {
        std::cout << "  Final computation: " << std::fixed << std::setprecision(3)
                  << final_time << "s"
                  << " (10^N: " << pow_time << "s, sqrt: " << sqrt_time
                  << "s, multiply+divide: " << mul_time << "s)"
                  << std::endl;
    }

    // Step 3: Convert integer to string
    // mpz_get_str for integers is O(n*log(n)) — much faster than mpf_get_str
    auto conv_start = Clock::now();
    char* raw = mpz_get_str(nullptr, 10, pi_int);
    std::string digits_str(raw);
    free(raw);

    // Format: insert "." after first digit
    // pi_int should be like 314159265... (config.digits+1 digits total)
    std::string result_str;
    if (digits_str.size() > 1) {
        result_str = digits_str.substr(0, 1) + "." + digits_str.substr(1);
    } else {
        result_str = digits_str;
    }

    // Trim to requested number of digits after decimal point
    size_t dot_pos = result_str.find('.');
    if (dot_pos != std::string::npos) {
        size_t target_len = dot_pos + 1 + config.digits;
        if (result_str.size() > target_len) {
            result_str = result_str.substr(0, target_len);
        }
    }

    auto conv_end = Clock::now();
    double conv_time = Duration(conv_end - conv_start).count();

    if (config.verbose) {
        std::cout << "  Integer to string: " << std::fixed << std::setprecision(3)
                  << conv_time << "s" << std::endl;
    }

    auto total_end = Clock::now();
    double total_time = Duration(total_end - total_start).count();

    if (config.verbose) {
        std::cout << "  ----------------------------------------" << std::endl;
        std::cout << "  Total: " << std::fixed << std::setprecision(3)
                  << total_time << "s" << std::endl;
        std::cout << "  Breakdown:" << std::endl;
        std::cout << "    Binary splitting:   " << std::setw(8) << bs_time << "s ("
                  << std::setw(5) << std::setprecision(1) << (bs_time / total_time * 100) << "%)" << std::endl;
        std::cout << "    Final computation:  " << std::setw(8) << std::setprecision(3) << final_time << "s ("
                  << std::setw(5) << std::setprecision(1) << (final_time / total_time * 100) << "%)" << std::endl;
        std::cout << "    Integer to string:  " << std::setw(8) << std::setprecision(3) << conv_time << "s ("
                  << std::setw(5) << std::setprecision(1) << (conv_time / total_time * 100) << "%)" << std::endl;
    }

    // Cleanup
    mpz_clear(scale);
    mpz_clear(sqrt_arg);
    mpz_clear(sqrt_val);
    mpz_clear(numerator);
    mpz_clear(pi_scaled);
    mpz_clear(guard_power);
    mpz_clear(pi_int);

    return PiResult{result_str, total_time, terms};
}

std::string PiEngine::compute_digits(size_t digits) {
    PiConfig config;
    config.digits = digits;
    config.verbose = false;
    return compute(config).digits;
}

} // namespace pi
