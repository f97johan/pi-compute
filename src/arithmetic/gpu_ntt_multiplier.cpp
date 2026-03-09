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
    // O(n) conversion: extract base-2^15 digits directly from GMP's limb array.
    // GMP stores numbers as arrays of mp_limb_t (64-bit on most platforms).
    // We extract GPU_BASE_BITS-bit chunks by bit-shifting through the limbs.

    if (mpz_sgn(n) == 0) {
        digits.assign(1, 0);
        return;
    }

    size_t bits = mpz_sizeinbase(n, 2);
    size_t num_digits = (bits + GPU_BASE_BITS - 1) / GPU_BASE_BITS;
    digits.resize(num_digits);

    // Export raw bytes from GMP (little-endian word order, native byte order)
    size_t count = 0;
    // Use mpz_export to get raw limb data as bytes
    // We'll work directly with the limb pointer for maximum efficiency
    const mp_limb_t* limbs = mpz_limbs_read(n);
    size_t num_limbs = mpz_size(n);

    // Extract GPU_BASE_BITS-bit digits from the limb array
    // Each limb is GMP_NUMB_BITS bits (typically 64)
    uint64_t accumulator = 0;
    int acc_bits = 0;
    size_t limb_idx = 0;
    size_t digit_idx = 0;
    const uint32_t mask = GPU_BASE - 1;  // 0x7FFF for base 2^15

    while (digit_idx < num_digits) {
        // Fill accumulator with more bits from limbs
        while (acc_bits < GPU_BASE_BITS && limb_idx < num_limbs) {
            accumulator |= (static_cast<uint64_t>(limbs[limb_idx]) << acc_bits);
            acc_bits += GMP_NUMB_BITS;
            limb_idx++;
        }

        // Extract one digit
        digits[digit_idx] = static_cast<uint32_t>(accumulator & mask);
        accumulator >>= GPU_BASE_BITS;
        acc_bits -= GPU_BASE_BITS;
        digit_idx++;
    }

    // Remove trailing zeros
    while (digits.size() > 1 && digits.back() == 0) {
        digits.pop_back();
    }
}

void GpuNttMultiplier::base24_to_mpz(const uint32_t* digits, size_t len, mpz_t result) {
    // O(n) conversion: pack base-2^15 digits directly into GMP limbs.

    if (len == 0) {
        mpz_set_ui(result, 0);
        return;
    }

    // Calculate how many limbs we need
    size_t total_bits = static_cast<size_t>(len) * GPU_BASE_BITS;
    size_t num_limbs = (total_bits + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;

    // Get writable limb pointer from GMP
    mp_limb_t* limbs = mpz_limbs_write(result, num_limbs);

    // Pack digits into limbs
    uint64_t accumulator = 0;
    int acc_bits = 0;
    size_t limb_idx = 0;
    size_t digit_idx = 0;

    while (digit_idx < len) {
        accumulator |= (static_cast<uint64_t>(digits[digit_idx]) << acc_bits);
        acc_bits += GPU_BASE_BITS;
        digit_idx++;

        // Flush full limbs
        while (acc_bits >= GMP_NUMB_BITS && limb_idx < num_limbs) {
            limbs[limb_idx] = static_cast<mp_limb_t>(accumulator);
            accumulator >>= GMP_NUMB_BITS;
            acc_bits -= GMP_NUMB_BITS;
            limb_idx++;
        }
    }

    // Flush remaining bits
    if (acc_bits > 0 && limb_idx < num_limbs) {
        limbs[limb_idx] = static_cast<mp_limb_t>(accumulator);
        limb_idx++;
    }

    // Zero any remaining limbs
    while (limb_idx < num_limbs) {
        limbs[limb_idx] = 0;
        limb_idx++;
    }

    // Tell GMP how many limbs are actually used
    mpz_limbs_finish(result, num_limbs);
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
