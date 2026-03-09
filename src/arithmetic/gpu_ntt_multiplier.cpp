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
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

namespace pi {

// Base for GPU representation: 2^12 = 4096
// For FFT convolution, max value = B^2 * N must fit in double precision (2^53).
// With B=2^12 and N=2^24: (2^12)^2 * 2^24 = 2^48 < 2^53 ✓ (safe with margin)
// With B=2^15 and N=2^23: (2^15)^2 * 2^23 = 2^53 — AT THE LIMIT, fails on A100
static constexpr uint32_t GPU_BASE_BITS = 12;
static constexpr uint32_t GPU_BASE = 1u << GPU_BASE_BITS;  // 4096

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
    using Clock = std::chrono::high_resolution_clock;

    // Fall back to GMP for small operands
    if (!should_use_gpu(a, b)) {
        stats_.cpu_fallback_calls.fetch_add(1);
        mpz_mul(result, a, b);
        return;
    }

    auto total_start = Clock::now();
    stats_.gpu_calls.fetch_add(1);

    // Determine sign
    int sign_a = mpz_sgn(a);
    int sign_b = mpz_sgn(b);
    int result_sign = sign_a * sign_b;

    if (result_sign == 0) {
        mpz_set_ui(result, 0);
        return;
    }

    // Convert to base-2^12 (CPU, thread-safe)
    auto conv_to_start = Clock::now();
    std::vector<uint32_t> digits_a, digits_b;
    mpz_to_base15(a, digits_a);
    mpz_to_base15(b, digits_b);
    auto conv_to_end = Clock::now();

    size_t max_result_len = digits_a.size() + digits_b.size();
    std::vector<uint32_t> result_digits(max_result_len, 0);

    // Select a GPU and lock it for this multiplication
    auto gpu_start = Clock::now();
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
    auto gpu_end = Clock::now();

    // Convert back to GMP (CPU, thread-safe)
    auto conv_from_start = Clock::now();
    base15_to_mpz(result_digits.data(), actual_len, result);
    auto conv_from_end = Clock::now();

    if (result_sign < 0) {
        mpz_neg(result, result);
    }

    auto total_end = Clock::now();

    // Accumulate stats (nanoseconds)
    auto ns = [](auto start, auto end) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    };
    stats_.convert_to_ns.fetch_add(ns(conv_to_start, conv_to_end));
    stats_.gpu_compute_ns.fetch_add(ns(gpu_start, gpu_end));
    stats_.convert_from_ns.fetch_add(ns(conv_from_start, conv_from_end));
    stats_.total_gpu_ns.fetch_add(ns(total_start, total_end));
}

void GpuNttMultiplier::square(mpz_t result, const mpz_t a) {
    multiply(result, a, a);
}

void GpuNttMultiplier::print_stats() const {
    auto to_sec = [](uint64_t ns) { return static_cast<double>(ns) / 1e9; };

    uint64_t gpu = stats_.gpu_calls.load();
    uint64_t cpu = stats_.cpu_fallback_calls.load();
    double conv_to = to_sec(stats_.convert_to_ns.load());
    double gpu_compute = to_sec(stats_.gpu_compute_ns.load());
    double conv_from = to_sec(stats_.convert_from_ns.load());
    double total = to_sec(stats_.total_gpu_ns.load());

    std::cout << "  GPU Multiplier Stats:" << std::endl;
    std::cout << "    GPU multiply calls:    " << gpu << std::endl;
    std::cout << "    CPU fallback calls:    " << cpu << std::endl;
    if (gpu > 0) {
        std::cout << "    GMP→base12 conversion: " << std::fixed << std::setprecision(3)
                  << conv_to << "s" << std::endl;
        std::cout << "    GPU FFT compute:       " << gpu_compute << "s" << std::endl;
        std::cout << "    base12→GMP conversion: " << conv_from << "s" << std::endl;
        std::cout << "    Total GPU path time:   " << total << "s" << std::endl;
        if (total > 0) {
            std::cout << "    Breakdown: convert "
                      << std::setprecision(1) << ((conv_to + conv_from) / total * 100) << "%"
                      << " | GPU " << (gpu_compute / total * 100) << "%"
                      << " | other " << ((total - conv_to - gpu_compute - conv_from) / total * 100) << "%"
                      << std::endl;
        }
    }
}

void GpuNttMultiplier::reset_stats() {
    stats_.gpu_calls = 0;
    stats_.cpu_fallback_calls = 0;
    stats_.convert_to_ns = 0;
    stats_.gpu_compute_ns = 0;
    stats_.convert_from_ns = 0;
    stats_.total_gpu_ns = 0;
}

} // namespace pi

#endif // PI_CUDA_ENABLED
