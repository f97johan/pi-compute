/**
 * @file gpu_ntt_multiplier.cpp
 * @brief Multi-GPU NTT multiplier implementation — bridges GMP and CUDA.
 *
 * Creates one NTT engine per GPU, dispatches multiplications round-robin.
 * On single-GPU systems, behaves identically to a serialized multiplier.
 */

#ifdef PI_CUDA_ENABLED

#include "gpu_ntt_multiplier.h"
#include <gmp.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

namespace pi {

// Base for GPU representation: 2^15 = 32768
static constexpr uint32_t GPU_BASE_BITS = 15;
static constexpr uint32_t GPU_BASE = 1u << GPU_BASE_BITS;  // 32768

GpuNttMultiplier::GpuNttMultiplier(size_t threshold, int num_gpus)
    : threshold_(threshold) {

    // Auto-detect GPU count
    int available_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&available_gpus);
    if (err != cudaSuccess || available_gpus == 0) {
        throw std::runtime_error("No CUDA GPUs available");
    }

    // Use requested count or all available
    int gpus_to_use = (num_gpus <= 0) ? available_gpus : std::min(num_gpus, available_gpus);

    // Create one NTT engine per GPU
    for (int i = 0; i < gpus_to_use; ++i) {
        auto ctx = std::make_unique<GpuContext>(i);
        ctx->engine = std::make_unique<gpu::NttEngine>(i);
        gpu_contexts_.push_back(std::move(ctx));
    }
}

std::string GpuNttMultiplier::device_name() const {
    std::string names;
    for (size_t i = 0; i < gpu_contexts_.size(); ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpu_contexts_[i]->device_id);
        if (i > 0) names += ", ";
        names += prop.name;
    }
    return names;
}

GpuContext& GpuNttMultiplier::select_gpu() {
    uint64_t idx = next_gpu_.fetch_add(1) % gpu_contexts_.size();
    return *gpu_contexts_[idx];
}

bool GpuNttMultiplier::should_use_gpu(const mpz_t a, const mpz_t b) const {
    size_t a_limbs = mpz_size(a);
    size_t b_limbs = mpz_size(b);
    size_t min_limbs = std::min(a_limbs, b_limbs);
    return (min_limbs >= threshold_);
}

void GpuNttMultiplier::mpz_to_base15(const mpz_t n, std::vector<uint32_t>& digits) {
    if (mpz_sgn(n) == 0) {
        digits.assign(1, 0);
        return;
    }

    size_t bits = mpz_sizeinbase(n, 2);
    size_t num_bytes = (bits + 7) / 8;
    std::vector<uint8_t> bytes(num_bytes + 1, 0);

    size_t count = 0;
    mpz_export(bytes.data(), &count, -1, 1, -1, 0, n);

    size_t num_digits = (bits + GPU_BASE_BITS - 1) / GPU_BASE_BITS;
    digits.resize(num_digits);

    uint32_t accumulator = 0;
    int acc_bits = 0;
    size_t byte_idx = 0;
    const uint32_t mask = GPU_BASE - 1;

    for (size_t d = 0; d < num_digits; ++d) {
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

    while (digits.size() > 1 && digits.back() == 0) {
        digits.pop_back();
    }
}

void GpuNttMultiplier::base15_to_mpz(const uint32_t* digits, size_t len, mpz_t result) {
    if (len == 0) {
        mpz_set_ui(result, 0);
        return;
    }

    size_t total_bits = static_cast<size_t>(len) * GPU_BASE_BITS;
    size_t num_bytes = (total_bits + 7) / 8;
    std::vector<uint8_t> bytes(num_bytes + 4, 0);

    uint32_t accumulator = 0;
    int acc_bits = 0;
    size_t byte_idx = 0;

    for (size_t d = 0; d < len; ++d) {
        accumulator |= (static_cast<uint32_t>(digits[d]) << acc_bits);
        acc_bits += GPU_BASE_BITS;
        while (acc_bits >= 8) {
            bytes[byte_idx] = static_cast<uint8_t>(accumulator & 0xFF);
            accumulator >>= 8;
            acc_bits -= 8;
            byte_idx++;
        }
    }
    if (acc_bits > 0) {
        bytes[byte_idx] = static_cast<uint8_t>(accumulator & 0xFF);
        byte_idx++;
    }

    mpz_import(result, byte_idx, -1, 1, -1, 0, bytes.data());
}

void GpuNttMultiplier::multiply(mpz_t result, const mpz_t a, const mpz_t b) {
    // Fall back to GMP for small operands
    if (!should_use_gpu(a, b)) {
        mpz_mul(result, a, b);
        return;
    }

    // Determine sign
    int sign_a = mpz_sgn(a);
    int sign_b = mpz_sgn(b);
    int result_sign = sign_a * sign_b;

    if (result_sign == 0) {
        mpz_set_ui(result, 0);
        return;
    }

    // Convert to base-2^15 (CPU, thread-safe, no GPU needed)
    std::vector<uint32_t> digits_a, digits_b;
    mpz_to_base15(a, digits_a);
    mpz_to_base15(b, digits_b);

    size_t max_result_len = digits_a.size() + digits_b.size();
    std::vector<uint32_t> result_digits(max_result_len, 0);

    // Select a GPU and lock it for this multiplication
    GpuContext& ctx = select_gpu();
    size_t actual_len;
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        actual_len = ctx.engine->multiply(
            digits_a.data(), digits_a.size(),
            digits_b.data(), digits_b.size(),
            result_digits.data(), max_result_len
        );
    }

    // Convert back to GMP (CPU, thread-safe)
    base15_to_mpz(result_digits.data(), actual_len, result);

    if (result_sign < 0) {
        mpz_neg(result, result);
    }
}

void GpuNttMultiplier::square(mpz_t result, const mpz_t a) {
    multiply(result, a, a);
}

} // namespace pi

#endif // PI_CUDA_ENABLED
