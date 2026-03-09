/**
 * @file pi_engine.cpp
 * @brief Pi computation orchestrator implementation.
 */

#include "pi_engine.h"
#include "binary_splitting.h"
#include "../arithmetic/newton_divider.h"
#include "../io/base_converter.h"
#include <chrono>
#include <iostream>
#include <gmp.h>

namespace pi {

PiEngine::PiEngine(Multiplier& multiplier)
    : multiplier_(multiplier) {}

PiResult PiEngine::compute(const PiConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (config.verbose) {
        std::cout << "Computing " << config.digits << " digits of pi..." << std::endl;
    }

    // We need extra precision for intermediate calculations.
    // Guard digits ensure the last requested digit is correct after rounding.
    // We compute with extra digits and then truncate to the requested count.
    size_t guard_digits = 100;  // Extra digits beyond what's requested
    size_t precision = config.digits + guard_digits;

    // Step 1: Determine number of terms needed (for full precision including guard)
    unsigned long terms = BinarySplitting::terms_needed(precision);
    if (config.verbose) {
        std::cout << "  Terms needed: " << terms << std::endl;
    }

    // Step 2: Run binary splitting to get P, Q, R
    BinarySplitting bs(multiplier_);
    if (config.verbose) {
        std::cout << "  Threads: " << bs.thread_count() << std::endl;
    }
    BSResult bsr = bs.compute(0, terms);

    if (config.verbose) {
        std::cout << "  Binary splitting complete." << std::endl;
    }

    // Step 3: Compute pi = (Q * 426880 * sqrt(10005)) / R
    //
    // The binary splitting R(0,N) already includes the a(k) = A + B*k factor,
    // so the final formula is simply:
    //   pi = C^(3/2)/12 * Q(0,N) / R(0,N)
    //      = 426880 * sqrt(10005) * Q(0,N) / R(0,N)

    // precision and guard_digits already computed above (before terms_needed)

    // Compute numerator: Q * 426880 (as integer, then multiply by sqrt(10005) in float)
    mpz_t numerator_int;
    mpz_init(numerator_int);
    mpz_mul_ui(numerator_int, bsr.Q, 426880);

    // Denominator is simply R(0,N) — it already contains the series sum
    // including the a(k) = 13591409 + 545140134*k linear terms
    mpz_t denominator_int;
    mpz_init(denominator_int);
    mpz_t temp;
    mpz_init(temp);
    mpz_set(denominator_int, bsr.R);

    if (config.verbose) {
        std::cout << "  Computing sqrt(10005)..." << std::endl;
    }

    // Compute sqrt(10005) to required precision
    mpf_t sqrt_10005;
    mpf_init(sqrt_10005);
    NewtonDivider::sqrt_to_precision(sqrt_10005, 10005, precision);

    // Compute the final result as floating-point:
    // pi = (numerator_int * sqrt_10005) / denominator_int
    mp_bitcnt_t precision_bits = static_cast<mp_bitcnt_t>(precision * 3.3219281) + 64;

    mpf_t pi_value;
    mpf_init2(pi_value, precision_bits);

    // Convert integers to floats
    mpf_t num_f, den_f;
    mpf_init2(num_f, precision_bits);
    mpf_init2(den_f, precision_bits);

    mpf_set_z(num_f, numerator_int);
    mpf_set_z(den_f, denominator_int);

    // pi = (num_f * sqrt_10005) / den_f
    mpf_mul(pi_value, num_f, sqrt_10005);
    mpf_div(pi_value, pi_value, den_f);

    if (config.verbose) {
        std::cout << "  Converting to decimal string..." << std::endl;
    }

    // Step 4: Convert to decimal string with extra precision, then truncate
    std::string digit_string = BaseConverter::to_decimal_string(pi_value, precision);

    // Truncate to requested digits: "3." + config.digits decimal digits
    size_t target_len = config.digits + 2;  // "3." prefix + digits
    if (digit_string.size() > target_len) {
        digit_string = digit_string.substr(0, target_len);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    if (config.verbose) {
        std::cout << "  Done in " << elapsed << " seconds." << std::endl;
    }

    // Cleanup
    mpz_clear(numerator_int);
    mpz_clear(denominator_int);
    mpz_clear(temp);
    mpf_clear(sqrt_10005);
    mpf_clear(pi_value);
    mpf_clear(num_f);
    mpf_clear(den_f);

    return PiResult{digit_string, elapsed, terms};
}

std::string PiEngine::compute_digits(size_t digits) {
    PiConfig config;
    config.digits = digits;
    config.verbose = false;
    return compute(config).digits;
}

} // namespace pi
