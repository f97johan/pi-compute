/**
 * @file gpu_ntt_multiplier.cpp
 * @brief GPU NTT multiplier implementation — bridges GMP and CUDA.
 */

#ifdef PI_CUDA_ENABLED

#include "gpu_ntt_multiplier.h"
#include <gmp.h>
#include <vector>
#include <cstring>
#include <algorithm>

namespace pi {

// Base for GPU representation: 2^15 = 32768
// Chosen so that B^2 * N < 2^53 for FFT sizes up to 2^20 (~1M elements).
// With B=2^15: (2^15)^2 * 2^20 = 2^50 < 2^53 (safe for double precision)
// With B=2^24: (2^24)^2 * 2^14 = 2^62 > 2^53 (OVERFLOW — causes wrong results)
static constexpr uint32_t GPU_BASE_BITS = 15;
static constexpr uint32_t GPU_BASE = 1u << GPU_BASE_BITS;  // 32768

GpuNttMultiplier::GpuNttMultiplier(size_t threshold)
    : threshold_(threshold) {}

std::string GpuNttMultiplier::device_name() const {
    return gpu::NttEngine::get_device_name();
}

bool GpuNttMultiplier::should_use_gpu(const mpz_t a, const mpz_t b) const {
    // Use GPU only if BOTH operands are large enough to benefit.
    // Small × large is still fast on CPU; the GPU overhead isn't worth it.
    size_t a_limbs = mpz_size(a);
    size_t b_limbs = mpz_size(b);
    size_t min_limbs = std::min(a_limbs, b_limbs);
    return (min_limbs >= threshold_);
}

void GpuNttMultiplier::mpz_to_base24(const mpz_t n, std::vector<uint32_t>& digits) {
    // Get the number of bits in n
    size_t bits = mpz_sizeinbase(n, 2);
    if (bits == 0 || mpz_sgn(n) == 0) {
        digits.assign(1, 0);
        return;
    }

    // Number of base-2^24 digits needed
    size_t num_digits = (bits + GPU_BASE_BITS - 1) / GPU_BASE_BITS;
    digits.resize(num_digits);

    // Extract base-2^24 digits by repeated division
    // More efficient: directly extract from GMP's internal limb representation
    mpz_t temp, remainder;
    mpz_init(temp);
    mpz_init(remainder);
    mpz_abs(temp, n);

    for (size_t i = 0; i < num_digits; ++i) {
        // digits[i] = temp % 2^24
        digits[i] = static_cast<uint32_t>(mpz_fdiv_ui(temp, GPU_BASE));
        // temp = temp / 2^24
        mpz_fdiv_q_2exp(temp, temp, GPU_BASE_BITS);
    }

    mpz_clear(temp);
    mpz_clear(remainder);

    // Remove trailing zeros
    while (digits.size() > 1 && digits.back() == 0) {
        digits.pop_back();
    }
}

void GpuNttMultiplier::base24_to_mpz(const uint32_t* digits, size_t len, mpz_t result) {
    mpz_set_ui(result, 0);

    // Build from most significant digit down: result = result * base + digit
    mpz_t base_power;
    mpz_init(base_power);

    for (size_t i = len; i > 0; --i) {
        mpz_mul_2exp(result, result, GPU_BASE_BITS);  // result <<= 24
        mpz_add_ui(result, result, digits[i - 1]);
    }

    mpz_clear(base_power);
}

void GpuNttMultiplier::multiply(mpz_t result, const mpz_t a, const mpz_t b) {
    // Fall back to GMP for small operands
    if (!should_use_gpu(a, b)) {
        mpz_mul(result, a, b);
        return;
    }

    // Determine sign of result
    int sign_a = mpz_sgn(a);
    int sign_b = mpz_sgn(b);
    int result_sign = sign_a * sign_b;

    if (result_sign == 0) {
        mpz_set_ui(result, 0);
        return;
    }

    // Convert to base-2^24
    std::vector<uint32_t> digits_a, digits_b;
    mpz_to_base24(a, digits_a);
    mpz_to_base24(b, digits_b);

    // Allocate result array
    size_t max_result_len = digits_a.size() + digits_b.size();
    std::vector<uint32_t> result_digits(max_result_len, 0);

    // GPU multiply via NTT
    size_t actual_len = ntt_engine_.multiply(
        digits_a.data(), digits_a.size(),
        digits_b.data(), digits_b.size(),
        result_digits.data(), max_result_len
    );

    // Convert back to GMP
    base24_to_mpz(result_digits.data(), actual_len, result);

    // Apply sign
    if (result_sign < 0) {
        mpz_neg(result, result);
    }
}

void GpuNttMultiplier::square(mpz_t result, const mpz_t a) {
    // For squaring, we can optimize by only doing one FFT transform
    // (since both operands are the same). For now, just call multiply.
    multiply(result, a, a);
}

} // namespace pi

#endif // PI_CUDA_ENABLED
