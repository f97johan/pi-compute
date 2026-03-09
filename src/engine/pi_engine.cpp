/**
 * @file pi_engine.cpp
 * @brief Pi computation orchestrator implementation with detailed timing.
 */

#include "pi_engine.h"
#include "binary_splitting.h"
#include "../arithmetic/newton_divider.h"
#include "../io/base_converter.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <gmp.h>

namespace pi {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

PiEngine::PiEngine(Multiplier& multiplier)
    : multiplier_(multiplier) {}

PiResult PiEngine::compute(const PiConfig& config) {
    auto total_start = Clock::now();

    if (config.verbose) {
        std::cout << "Computing " << config.digits << " digits of pi..." << std::endl;
    }

    size_t guard_digits = 100;
    size_t precision = config.digits + guard_digits;

    // Step 1: Determine number of terms
    unsigned long terms = BinarySplitting::terms_needed(precision);
    if (config.verbose) {
        std::cout << "  Terms needed: " << terms << std::endl;
    }

    // Step 2: Binary splitting
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

    // Step 3: Final computation (sqrt, multiply, divide)
    auto final_start = Clock::now();

    mpz_t numerator_int;
    mpz_init(numerator_int);
    mpz_mul_ui(numerator_int, bsr.Q, 426880);

    mpz_t denominator_int;
    mpz_init(denominator_int);
    mpz_t temp;
    mpz_init(temp);
    mpz_set(denominator_int, bsr.R);

    // sqrt(10005)
    auto sqrt_start = Clock::now();
    mpf_t sqrt_10005;
    mpf_init(sqrt_10005);
    NewtonDivider::sqrt_to_precision(sqrt_10005, 10005, precision);
    auto sqrt_end = Clock::now();
    double sqrt_time = Duration(sqrt_end - sqrt_start).count();

    // Final multiply and divide
    auto div_start = Clock::now();
    mp_bitcnt_t precision_bits = static_cast<mp_bitcnt_t>(precision * 3.3219281) + 64;

    mpf_t pi_value;
    mpf_init2(pi_value, precision_bits);

    mpf_t num_f, den_f;
    mpf_init2(num_f, precision_bits);
    mpf_init2(den_f, precision_bits);

    mpf_set_z(num_f, numerator_int);
    mpf_set_z(den_f, denominator_int);

    mpf_mul(pi_value, num_f, sqrt_10005);
    mpf_div(pi_value, pi_value, den_f);
    auto div_end = Clock::now();
    double div_time = Duration(div_end - div_start).count();

    double final_time = Duration(div_end - final_start).count();

    if (config.verbose) {
        std::cout << "  Final computation: " << std::fixed << std::setprecision(3)
                  << final_time << "s"
                  << " (sqrt: " << sqrt_time << "s, multiply+divide: " << div_time << "s)"
                  << std::endl;
    }

    // Step 4: Decimal conversion
    auto conv_start = Clock::now();
    std::string digit_string = BaseConverter::to_decimal_string(pi_value, precision);

    size_t target_len = config.digits + 2;
    if (digit_string.size() > target_len) {
        digit_string = digit_string.substr(0, target_len);
    }
    auto conv_end = Clock::now();
    double conv_time = Duration(conv_end - conv_start).count();

    if (config.verbose) {
        std::cout << "  Decimal conversion: " << std::fixed << std::setprecision(3)
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
        std::cout << "    Decimal conversion: " << std::setw(8) << std::setprecision(3) << conv_time << "s ("
                  << std::setw(5) << std::setprecision(1) << (conv_time / total_time * 100) << "%)" << std::endl;
    }

    // Cleanup
    mpz_clear(numerator_int);
    mpz_clear(denominator_int);
    mpz_clear(temp);
    mpf_clear(sqrt_10005);
    mpf_clear(pi_value);
    mpf_clear(num_f);
    mpf_clear(den_f);

    return PiResult{digit_string, total_time, terms};
}

std::string PiEngine::compute_digits(size_t digits) {
    PiConfig config;
    config.digits = digits;
    config.verbose = false;
    return compute(config).digits;
}

} // namespace pi
