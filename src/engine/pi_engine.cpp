/**
 * @file pi_engine.cpp
 * @brief Pi computation orchestrator — hybrid mpf/mpz approach.
 *
 * Uses mpf for the final computation (sqrt + divide) because it's faster
 * than the pure integer approach (avoids computing 10^(2N)).
 * Then extracts the result as an integer and uses mpz_get_str for
 * string conversion (faster than mpf_get_str at large scales).
 */

#include "pi_engine.h"
#include "binary_splitting.h"
#include "../arithmetic/newton_divider.h"
#include "../io/base_converter.h"
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
        std::cout << "Computing " << config.digits << " digits of pi..." << std::endl;
    }

    size_t guard_digits = 100;
    size_t precision = config.digits + guard_digits;

    // Step 1: Binary splitting
    unsigned long terms = BinarySplitting::terms_needed(precision);
    if (config.verbose) {
        std::cout << "  Terms needed: " << terms << std::endl;
    }

    auto bs_start = Clock::now();
    BinarySplitting bs(multiplier_);
    if (!config.checkpoint_dir.empty()) {
        bs.enable_checkpointing(config.checkpoint_dir);
        if (config.verbose) {
            std::cout << "  Checkpointing: " << config.checkpoint_dir << std::endl;
        }
    }
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

    // Step 2: Final computation using mpf (faster than pure integer for sqrt+divide)
    // pi = 426880 * sqrt(10005) * Q / R
    auto final_start = Clock::now();

    mp_bitcnt_t precision_bits = static_cast<mp_bitcnt_t>(precision * 3.3219281) + 64;

    // Compute sqrt(10005)
    auto sqrt_start = Clock::now();
    mpf_t sqrt_10005;
    mpf_init2(sqrt_10005, precision_bits);
    NewtonDivider::sqrt_to_precision(sqrt_10005, 10005, precision);
    auto sqrt_end = Clock::now();
    double sqrt_time = Duration(sqrt_end - sqrt_start).count();

    // Compute pi as mpf: (426880 * sqrt(10005) * Q) / R
    auto div_start = Clock::now();
    mpz_t numerator_int;
    mpz_init(numerator_int);
    mpz_mul_ui(numerator_int, bsr.Q, 426880);

    mpf_t pi_value, num_f, den_f;
    mpf_init2(pi_value, precision_bits);
    mpf_init2(num_f, precision_bits);
    mpf_init2(den_f, precision_bits);

    mpf_set_z(num_f, numerator_int);
    mpf_set_z(den_f, bsr.R);

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

    // Step 3: Extract as integer and convert to string
    auto conv_start = Clock::now();

    mpz_t pi_int;
    mpz_init(pi_int);

    // Check for pi_int checkpoint (skips both binary splitting AND final computation)
    std::string pi_int_ckpt = config.checkpoint_dir.empty() ? "" :
                               config.checkpoint_dir + "/pi_int_" + std::to_string(config.digits) + ".ckpt";
    bool loaded_pi_int = false;

    if (!pi_int_ckpt.empty()) {
        FILE* f = fopen(pi_int_ckpt.c_str(), "rb");
        if (f) {
            size_t count;
            if (fread(&count, sizeof(size_t), 1, f) == 1 && count > 0) {
                std::vector<uint8_t> data(count);
                if (fread(data.data(), 1, count, f) == count) {
                    mpz_import(pi_int, count, -1, 1, -1, 0, data.data());
                    loaded_pi_int = true;
                    if (config.verbose) {
                        std::cout << "  Loaded pi_int checkpoint (" << count << " bytes)" << std::endl;
                    }
                }
            }
            fclose(f);
        }
    }

    if (!loaded_pi_int) {
        // Scale pi by 10^digits to get an integer
        mpf_t scale_f;
        mpf_init2(scale_f, precision_bits);
        mpz_t scale_z;
        mpz_init(scale_z);
        mpz_ui_pow_ui(scale_z, 10, config.digits);
        mpf_set_z(scale_f, scale_z);
        mpf_mul(pi_value, pi_value, scale_f);

        // Extract integer part
        mpz_set_f(pi_int, pi_value);

        mpf_clear(scale_f);
        mpz_clear(scale_z);

        // Save pi_int checkpoint
        if (!pi_int_ckpt.empty()) {
            FILE* f = fopen(pi_int_ckpt.c_str(), "wb");
            if (f) {
                size_t count = 0;
                void* data = mpz_export(nullptr, &count, -1, 1, -1, 0, pi_int);
                fwrite(&count, sizeof(size_t), 1, f);
                if (count > 0 && data) { fwrite(data, 1, count, f); free(data); }
                fclose(f);
                if (config.verbose) {
                    std::cout << "  Saved pi_int checkpoint (" << count << " bytes)" << std::endl;
                }
            }
        }
    }

    // Convert to string using parallel divide-and-conquer
    // For large numbers (>100M digits), stream directly to file to save RAM
    std::string result_str;
    size_t num_digits = mpz_sizeinbase(pi_int, 10);
    bool use_streaming = (num_digits > 100000000);  // Stream for >100M digits

    if (use_streaming && !config.output_file.empty()) {
        // Streaming mode: write digits directly to file
        if (config.verbose) {
            std::cout << "  Streaming " << num_digits << " digits to file..." << std::endl;
        }

        FILE* out = fopen(config.output_file.c_str(), "w");
        if (!out) throw std::runtime_error("Cannot open output file: " + config.output_file);

        // Write "3." prefix
        fputc('3', out);
        fputc('.', out);

        // Stream digits via callback
        size_t digits_written = 0;
        BaseConverter::parallel_to_decimal(pi_int, 0,
            [&](size_t offset, const std::string& chunk) {
                // Skip the leading "3" (first digit), write the rest
                if (offset == 0 && !chunk.empty()) {
                    // First chunk starts with "3", skip it
                    fwrite(chunk.data() + 1, 1, chunk.size() - 1, out);
                    digits_written += chunk.size() - 1;
                } else {
                    size_t to_write = std::min(chunk.size(), config.digits - digits_written);
                    fwrite(chunk.data(), 1, to_write, out);
                    digits_written += to_write;
                }
            }
        );

        fclose(out);
        result_str = "3.14159...  (streamed to " + config.output_file + ")";
    } else {
        // In-memory mode: build full string
        std::string digits_str = BaseConverter::parallel_to_decimal(pi_int);

        if (digits_str.size() > 1) {
            result_str = digits_str.substr(0, 1) + "." + digits_str.substr(1);
        } else {
            result_str = digits_str;
        }

        size_t dot_pos = result_str.find('.');
        if (dot_pos != std::string::npos) {
            size_t target_len = dot_pos + 1 + config.digits;
            if (result_str.size() > target_len) {
                result_str = result_str.substr(0, target_len);
            }
        }
    }

    auto conv_end = Clock::now();
    double conv_time = Duration(conv_end - conv_start).count();

    if (config.verbose) {
        std::cout << "  String conversion: " << std::fixed << std::setprecision(3)
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
        std::cout << "    String conversion:  " << std::setw(8) << std::setprecision(3) << conv_time << "s ("
                  << std::setw(5) << std::setprecision(1) << (conv_time / total_time * 100) << "%)" << std::endl;
    }

    // Cleanup
    mpz_clear(numerator_int);
    mpz_clear(pi_int);
    mpf_clear(sqrt_10005);
    mpf_clear(pi_value);
    mpf_clear(num_f);
    mpf_clear(den_f);

    return PiResult{result_str, total_time, terms};
}

std::string PiEngine::compute_digits(size_t digits) {
    PiConfig config;
    config.digits = digits;
    config.verbose = false;
    return compute(config).digits;
}

} // namespace pi
