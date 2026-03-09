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
    // O(n) conversion using mpz_export to get raw bytes, then extract base-2^15 digits.

    if (mpz_sgn(n) == 0) {
        digits.assign(1, 0);
        return;
    }

    // Export absolute value as little-endian bytes
    size_t bits = mpz_sizeinbase(n, 2);
    size_t num_bytes = (bits + 7) / 8;
    std::vector<uint8_t> bytes(num_bytes + 1, 0);  // +1 for safety

    size_t count = 0;
    mpz_export(bytes.data(), &count, -1 /* least significant first */,
               1 /* 1 byte per word */, -1 /* little-endian */,
               0 /* no nail bits */, n);

    // Now extract GPU_BASE_BITS-bit digits from the byte array
    size_t num_digits = (bits + GPU_BASE_BITS - 1) / GPU_BASE_BITS;
    digits.resize(num_digits);

    // Walk through bytes, accumulating bits and extracting digits
    uint32_t accumulator = 0;
    int acc_bits = 0;
    size_t byte_idx = 0;
    const uint32_t mask = GPU_BASE - 1;

    for (size_t d = 0; d < num_digits; ++d) {
        // Fill accumulator until we have enough bits
        while (acc_bits < GPU_BASE_BITS && byte_idx < count) {
            accumulator |= (static_cast<uint32_t>(bytes[byte_idx]) << acc_bits);
            acc_bits += 8;
            byte_idx++;
        }

        digits[d] = accumulator & mask;
        accumulator >>= GPU_BASE_BITS;
        acc_bits -= GPU_BASE_BITS;
        if (acc_bits < 0) acc_bits = 0;
    }

    // Remove trailing zeros
    while (digits.size() > 1 && digits.back() == 0) {
        digits.pop_back();
    }
}

void GpuNttMultiplier::base24_to_mpz(const uint32_t* digits, size_t len, mpz_t result) {
    // O(n) conversion: pack base-2^15 digits into bytes, then mpz_import.

    if (len == 0) {
        mpz_set_ui(result, 0);
        return;
    }

    // Calculate total bytes needed
    size_t total_bits = static_cast<size_t>(len) * GPU_BASE_BITS;
    size_t num_bytes = (total_bits + 7) / 8;
    std::vector<uint8_t> bytes(num_bytes + 4, 0);  // +4 for safety

    // Pack digits into byte array (little-endian)
    uint32_t accumulator = 0;
    int acc_bits = 0;
    size_t byte_idx = 0;

    for (size_t d = 0; d < len; ++d) {
        accumulator |= (static_cast<uint32_t>(digits[d]) << acc_bits);
        acc_bits += GPU_BASE_BITS;

        // Flush complete bytes
        while (acc_bits >= 8) {
            bytes[byte_idx] = static_cast<uint8_t>(accumulator & 0xFF);
            accumulator >>= 8;
            acc_bits -= 8;
            byte_idx++;
        }
    }

    // Flush remaining bits
    if (acc_bits > 0) {
        bytes[byte_idx] = static_cast<uint8_t>(accumulator & 0xFF);
        byte_idx++;
    }

    // Import bytes into GMP
    mpz_import(result, byte_idx, -1 /* least significant first */,
               1 /* 1 byte per word */, -1 /* little-endian */,
               0 /* no nail bits */, bytes.data());
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

    // GPU multiply via NTT — serialized because cuFFT is not thread-safe
    size_t actual_len;
    {
        std::lock_guard<std::mutex> lock(gpu_mutex_);
        actual_len = ntt_engine_.multiply(
            digits_a.data(), digits_a.size(),
            digits_b.data(), digits_b.size(),
            result_digits.data(), max_result_len
        );
    }

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
