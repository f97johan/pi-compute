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
#include <fstream>

#ifdef PI_FLINT_ENABLED
#include <flint/flint.h>
#include <flint/fmpz.h>
#endif

#ifdef __linux__
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#endif
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <gmp.h>

namespace pi {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

/// Multiply two mpz_t values, using FLINT if available and operands are large.
/// GMP 6.2.1 has silent corruption in mpz_mul when both operands exceed ~5B digits.
/// FLINT's fmpz_mul uses a different FFT implementation that doesn't have this bug.
static void safe_mul(mpz_t result, const mpz_t a, const mpz_t b) {
    size_t max_size = std::max(mpz_size(a), mpz_size(b));

#ifdef PI_FLINT_ENABLED
    // Use FLINT for large multiplications to avoid GMP corruption
    if (max_size > 500000000UL) {  // > ~4GB per operand
        fmpz_t fa, fb, fr;
        fmpz_init(fa); fmpz_init(fb); fmpz_init(fr);
        fmpz_set_mpz(fa, a);
        fmpz_set_mpz(fb, b);
        fmpz_mul(fr, fa, fb);
        fmpz_get_mpz(result, fr);
        fmpz_clear(fa); fmpz_clear(fb); fmpz_clear(fr);
        return;
    }
#endif

    // GMP for small-to-medium operands (fast, no corruption)
    mpz_mul(result, a, b);
}

/// Compute result = 10^exp efficiently.
/// For large exponents, splits into manageable chunks and uses binary
/// exponentiation with safe_mul (FLINT for large operands, GMP for small).
///
/// Strategy: 10^exp = (10^chunk)^(exp/chunk) * 10^(exp%chunk)
/// where chunk <= 1B (safe for mpz_ui_pow_ui).
static void mpz_pow10(mpz_t result, size_t exp, bool verbose = false) {
    if (exp == 0) { mpz_set_ui(result, 1); return; }
    if (exp <= 999999999UL) {
        mpz_ui_pow_ui(result, 10, static_cast<unsigned long>(exp));
        return;
    }

    // Split: 10^exp = (10^chunk)^q * 10^r
    const size_t chunk = 500000000UL;  // 500M — base is ~2GB
    size_t q = exp / chunk;
    size_t r = exp % chunk;

    if (verbose) {
        std::cout << "    Computing 10^" << exp << " = (10^" << chunk << ")^" << q;
        if (r > 0) std::cout << " * 10^" << r;
        std::cout << std::endl;
    }

    // Compute base = 10^chunk
    mpz_t base;
    mpz_init(base);
    mpz_ui_pow_ui(base, 10, static_cast<unsigned long>(chunk));

    if (verbose) {
        std::cout << "    Base 10^" << chunk << ": "
                  << mpz_sizeinbase(base, 10) << " digits, "
                  << (mpz_size(base) * 8 / (1024*1024)) << " MB"
                  << " | RSS: " << PiEngine::get_rss_mb() << " MB" << std::endl;
    }

    // Binary exponentiation using safe_mul (FLINT for large operands)
    mpz_set_ui(result, 1);
    mpz_t b, tmp;
    mpz_init_set(b, base);
    mpz_init(tmp);

    size_t e = q;
    int step = 0;
    int total_steps = 0;
    { size_t t = q; while (t > 0) { total_steps++; t >>= 1; } }

    if (verbose) {
        std::cout << "    q=" << q << " (binary:";
        for (int i = total_steps - 1; i >= 0; --i)
            std::cout << ((q >> i) & 1);
        std::cout << "), " << total_steps << " steps" << std::endl;
    }

    while (e > 0) {
        bool bit_set = (e & 1);
        if (bit_set) {
            safe_mul(tmp, result, b);
            mpz_swap(result, tmp);
        }
        e >>= 1;
        step++;

        if (verbose) {
            std::cout << "    step " << step << "/" << total_steps
                      << " bit=" << bit_set
                      << " result=" << mpz_sizeinbase(result, 10) << "d"
                      << " base=" << mpz_sizeinbase(b, 10) << "d"
                      << " e_remaining=" << e
                      << " | RSS: " << PiEngine::get_rss_mb() << " MB"
                      << std::endl;
        }

        if (e > 0) {
            safe_mul(tmp, b, b);
            mpz_swap(b, tmp);
        }
    }
    mpz_clear(b);
    mpz_clear(tmp);

    // Multiply by 10^r for the remainder
    if (r > 0) {
        mpz_t tail;
        mpz_init(tail);
        mpz_ui_pow_ui(tail, 10, static_cast<unsigned long>(r));
        mpz_mul(result, result, tail);
        mpz_clear(tail);
    }

    mpz_clear(base);

    if (verbose) {
        size_t result_digits = mpz_sizeinbase(result, 10);
        std::cout << "    Result: " << result_digits << " digits, "
                  << (mpz_size(result) * 8 / (1024*1024)) << " MB"
                  << " | RSS: " << PiEngine::get_rss_mb() << " MB" << std::endl;
        if (result_digits < exp * 9 / 10 || result_digits > exp + 10) {
            std::cerr << "    ERROR: Expected ~" << exp << " digits, got " << result_digits
                      << ". Power computation may have failed!" << std::endl;
        }
    }
}

PiEngine::PiEngine(Multiplier& multiplier)
    : multiplier_(multiplier) {}

size_t PiEngine::get_rss_mb() {
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            // Format: "VmRSS:    12345 kB"
            size_t kb = 0;
            sscanf(line.c_str(), "VmRSS: %zu", &kb);
            return kb / 1024;
        }
    }
    return 0;
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size / (1024 * 1024);
    }
    return 0;
#else
    return 0;
#endif
}

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
    BinarySplitting bs(multiplier_, config.num_threads);
    if (!config.checkpoint_dir.empty()) {
        bs.enable_checkpointing(config.checkpoint_dir);
        if (config.verbose) {
            std::cout << "  Checkpointing: " << config.checkpoint_dir << std::endl;
        }
    }
    if (config.out_of_core) {
        if (config.checkpoint_dir.empty()) {
            std::cerr << "  WARNING: --out-of-core requires --checkpoint, ignoring" << std::endl;
        } else {
            bs.enable_out_of_core();
            if (config.verbose) {
                std::cout << "  Out-of-core: enabled (compute wide, merge narrow)" << std::endl;
            }
        }
    }
    if (config.verbose) {
        std::cout << "  Threads: " << bs.thread_count()
                  << " | RSS: " << get_rss_mb() << " MB" << std::endl;
    }
    BSResult bsr = bs.compute(0, terms);
    auto bs_end = Clock::now();
    double bs_time = Duration(bs_end - bs_start).count();

    if (config.verbose) {
        std::cout << "  Binary splitting: " << std::fixed << std::setprecision(3)
                  << bs_time << "s | RSS: " << get_rss_mb() << " MB" << std::endl;
    }

    // Step 2+3: Final computation → pi_int (integer with N digits after "3.")
    auto final_start = Clock::now();

    // Free P immediately — it's not needed for the final formula.
    mpz_realloc2(bsr.P, 0);

    mpz_t pi_int;
    mpz_init(pi_int);

    // Check for pi_int checkpoint (skips final computation entirely)
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

    double sqrt_time = 0, div_time = 0, final_time = 0;

    if (!loaded_pi_int && config.integer_math) {
        // ============================================================
        // INTEGER-ONLY PATH: all computation using mpz (no mpf)
        //
        // Restructured formula to avoid computing 10^(2*precision):
        //   pi_int = 426880 * Q * sqrt_small * 10^digits / (R * 10^guard)
        //
        // where sqrt_small = isqrt(10005 * 10^(2*guard)) is tiny (~100 digits)
        // and 10^digits is the only large power of 10 needed.
        //
        // This avoids the 100B-digit intermediate (10^(2*50B)) that caused
        // GMP segfaults due to internal limits on very large mpz_mul.
        // ============================================================

        if (config.verbose) {
            std::cout << "  Using integer-only math (no mpf)" << std::endl;
        }

        // Step 2a: Compute sqrt(10005) at guard_digits precision (tiny)
        auto sqrt_start = Clock::now();

        mpz_t sqrt_val;
        mpz_init(sqrt_val);
        {
            mpz_t S;
            mpz_init(S);
            mpz_ui_pow_ui(S, 10, 2 * guard_digits);  // 10^200 — tiny
            mpz_mul_ui(S, S, 10005);
            mpz_sqrt(sqrt_val, S);  // sqrt(10005) * 10^guard_digits (~100 digits)
            mpz_clear(S);
        }

        auto sqrt_end = Clock::now();
        sqrt_time = Duration(sqrt_end - sqrt_start).count();

        if (config.verbose) {
            std::cout << "  Integer sqrt(10005) at " << guard_digits << " extra digits: "
                      << std::fixed << std::setprecision(3) << sqrt_time << "s"
                      << " (" << mpz_sizeinbase(sqrt_val, 10) << " digits)"
                      << " | RSS: " << get_rss_mb() << " MB" << std::endl;
        }

        // Step 2b: Compute pi_int
        // pi_int = 426880 * Q * sqrt_val * 10^digits / (R * 10^guard_digits)
        auto div_start_t = Clock::now();

        // numerator = 426880 * Q * sqrt_val
        mpz_t numerator;
        mpz_init(numerator);

        if (config.verbose) {
            std::cout << "  Q size: " << mpz_sizeinbase(bsr.Q, 10) << " digits, "
                      << (mpz_size(bsr.Q) * 8 / (1024*1024)) << " MB, sign="
                      << mpz_sgn(bsr.Q) << std::endl;
            std::cout << "  R size: " << mpz_sizeinbase(bsr.R, 10) << " digits, "
                      << (mpz_size(bsr.R) * 8 / (1024*1024)) << " MB, sign="
                      << mpz_sgn(bsr.R) << std::endl;
        }

        mpz_mul_ui(numerator, bsr.Q, 426880);
        mpz_realloc2(bsr.Q, 0);  // Free Q

        // sqrt_val is small (~100 digits), so plain mpz_mul is fine
        mpz_mul(numerator, numerator, sqrt_val);
        mpz_clear(sqrt_val);

        if (config.verbose) {
            std::cout << "  Numerator (426880*Q*sqrt): "
                      << mpz_sizeinbase(numerator, 10) << " digits"
                      << " | RSS: " << get_rss_mb() << " MB" << std::endl;
        }

        // Scale numerator by 10^digits
        if (config.verbose) {
            std::cout << "  Computing 10^" << config.digits << " for scaling..." << std::endl;
        }
        {
            mpz_t scale;
            mpz_init(scale);
            mpz_pow10(scale, config.digits, config.verbose);

            if (config.verbose) {
                std::cout << "  Multiplying numerator by 10^" << config.digits
                          << " (" << mpz_sizeinbase(scale, 10) << " digits)..."
                          << " | RSS: " << get_rss_mb() << " MB" << std::endl;
            }
            safe_mul(numerator, numerator, scale);
            mpz_clear(scale);
        }

        if (config.verbose) {
            std::cout << "  Scaled numerator: "
                      << mpz_sizeinbase(numerator, 10) << " digits"
                      << " | RSS: " << get_rss_mb() << " MB" << std::endl;
        }

        // denominator = R * 10^guard_digits (guard_digits is small, ~100)
        mpz_t denominator;
        mpz_init(denominator);
        {
            mpz_t guard_scale;
            mpz_init(guard_scale);
            mpz_ui_pow_ui(guard_scale, 10, guard_digits);  // 10^100 — tiny
            mpz_mul(denominator, bsr.R, guard_scale);
            mpz_clear(guard_scale);
        }
        mpz_realloc2(bsr.R, 0);  // Free R

        // pi_int = numerator / denominator
        if (config.verbose) {
            std::cout << "  Final division: " << mpz_sizeinbase(numerator, 10)
                      << " / " << mpz_sizeinbase(denominator, 10) << " digits..."
                      << " | RSS: " << get_rss_mb() << " MB" << std::endl;
        }
        mpz_tdiv_q(pi_int, numerator, denominator);
        mpz_clear(numerator);
        mpz_clear(denominator);

        auto div_end_t = Clock::now();
        div_time = Duration(div_end_t - div_start_t).count();
        final_time = Duration(div_end_t - final_start).count();

        if (config.verbose) {
            std::cout << "  Final computation: " << std::fixed << std::setprecision(3)
                      << final_time << "s"
                      << " (sqrt: " << sqrt_time << "s, multiply+divide: " << div_time << "s)"
                      << " | RSS: " << get_rss_mb() << " MB" << std::endl;
        }

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

    } else if (!loaded_pi_int) {
        // ============================================================
        // FLOATING-POINT PATH (original): uses mpf for sqrt + divide
        // ============================================================

        mp_bitcnt_t precision_bits = static_cast<mp_bitcnt_t>(precision * 3.3219281) + 64;

        auto sqrt_start = Clock::now();
        mpf_t sqrt_10005;
        mpf_init2(sqrt_10005, precision_bits);
        NewtonDivider::sqrt_to_precision(sqrt_10005, 10005, precision);
        auto sqrt_end = Clock::now();
        sqrt_time = Duration(sqrt_end - sqrt_start).count();

        auto div_start_t = Clock::now();
        mpz_t numerator_int;
        mpz_init(numerator_int);
        mpz_mul_ui(numerator_int, bsr.Q, 426880);
        mpz_realloc2(bsr.Q, 0);

        mpf_t pi_value, num_f, den_f;
        mpf_init2(pi_value, precision_bits);
        mpf_init2(num_f, precision_bits);
        mpf_init2(den_f, precision_bits);

        mpf_set_z(num_f, numerator_int);
        mpz_clear(numerator_int);

        mpf_set_z(den_f, bsr.R);
        mpz_realloc2(bsr.R, 0);

        mpf_mul(pi_value, num_f, sqrt_10005);
        mpf_clear(sqrt_10005);
        mpf_clear(num_f);

        mpf_div(pi_value, pi_value, den_f);
        mpf_clear(den_f);
        auto div_end_t = Clock::now();
        div_time = Duration(div_end_t - div_start_t).count();
        final_time = Duration(div_end_t - final_start).count();

        if (config.verbose) {
            std::cout << "  Final computation: " << std::fixed << std::setprecision(3)
                      << final_time << "s"
                      << " (sqrt: " << sqrt_time << "s, multiply+divide: " << div_time << "s)"
                      << " | RSS: " << get_rss_mb() << " MB" << std::endl;
        }

        // Scale and extract integer
        mpf_t scale_f;
        mpf_init2(scale_f, precision_bits);
        mpz_t scale_z;
        mpz_init(scale_z);
        mpz_ui_pow_ui(scale_z, 10, config.digits);
        mpf_set_z(scale_f, scale_z);
        mpz_clear(scale_z);
        mpf_mul(pi_value, pi_value, scale_f);
        mpf_clear(scale_f);

        mpz_set_f(pi_int, pi_value);
        mpf_clear(pi_value);

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
    } else {
        // Loaded from checkpoint — free BSResult
        mpz_realloc2(bsr.Q, 0);
        mpz_realloc2(bsr.R, 0);
        final_time = Duration(Clock::now() - final_start).count();
    }

    // Step 4: Convert to string
    auto conv_start = Clock::now();

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
                  << conv_time << "s | RSS: " << get_rss_mb() << " MB" << std::endl;
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

    // Cleanup (most variables already freed early to reduce peak RSS)
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
