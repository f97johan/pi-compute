/**
 * @file int_ntt_multiplier.cpp
 * @brief Integer NTT multiplier — bridges GMP and CUDA integer NTT.
 */

#ifdef PI_CUDA_ENABLED

#include "int_ntt_multiplier.h"
#include <gmp.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

namespace pi {

// Use base 2^15 = 32768 for the digit representation
// With 3 NTT primes (~10^9 each), max convolution value = 32768^2 * N
// For N = 2^23 (8M elements): 32768^2 * 8M ≈ 8.6 * 10^15
// Product of 3 primes ≈ 7.4 * 10^26 — plenty of room
static constexpr uint32_t DIGIT_BASE_BITS = 15;
static constexpr uint64_t DIGIT_BASE = 1ULL << DIGIT_BASE_BITS;  // 32768

IntNttMultiplier::IntNttMultiplier(size_t threshold, int num_gpus)
    : threshold_(threshold) {
    int available = 0;
    cudaGetDeviceCount(&available);
    if (available == 0) throw std::runtime_error("No CUDA GPUs available");

    int to_use = (num_gpus <= 0) ? available : std::min(num_gpus, available);
    for (int i = 0; i < to_use; i++) {
        auto ctx = std::make_unique<IntNttGpuContext>(i);
        ctx->engine = std::make_unique<gpu::IntNttEngine>(i);
        gpu_contexts_.push_back(std::move(ctx));
    }
}

std::string IntNttMultiplier::device_name() const {
    std::string names;
    for (size_t i = 0; i < gpu_contexts_.size(); i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpu_contexts_[i]->device_id);
        if (i > 0) names += ", ";
        names += prop.name;
    }
    return names;
}

IntNttGpuContext& IntNttMultiplier::select_gpu() {
    uint64_t idx = next_gpu_.fetch_add(1) % gpu_contexts_.size();
    return *gpu_contexts_[idx];
}

bool IntNttMultiplier::should_use_gpu(const mpz_t a, const mpz_t b) const {
    size_t min_limbs = std::min(mpz_size(a), mpz_size(b));
    return min_limbs >= threshold_;
}

void IntNttMultiplier::mpz_to_digits(const mpz_t n, std::vector<uint32_t>& digits, uint32_t base_bits) {
    if (mpz_sgn(n) == 0) { digits.assign(1, 0); return; }

    size_t bits = mpz_sizeinbase(n, 2);
    size_t num_bytes = (bits + 7) / 8;
    std::vector<uint8_t> bytes(num_bytes + 1, 0);
    size_t count = 0;
    mpz_export(bytes.data(), &count, -1, 1, -1, 0, n);

    size_t num_digits = (bits + base_bits - 1) / base_bits;
    digits.resize(num_digits);
    uint32_t mask = (1u << base_bits) - 1;

    uint32_t acc = 0;
    int acc_bits = 0;
    size_t byte_idx = 0;
    for (size_t d = 0; d < num_digits; d++) {
        while (acc_bits < (int)base_bits && byte_idx < count) {
            acc |= (static_cast<uint32_t>(bytes[byte_idx]) << acc_bits);
            acc_bits += 8;
            byte_idx++;
        }
        digits[d] = acc & mask;
        acc >>= base_bits;
        acc_bits -= base_bits;
        if (acc_bits < 0) acc_bits = 0;
    }
    while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
}

void IntNttMultiplier::digits_to_mpz(const uint32_t* digits, size_t len, mpz_t result, uint32_t base_bits) {
    if (len == 0) { mpz_set_ui(result, 0); return; }

    size_t total_bits = len * base_bits;
    size_t num_bytes = (total_bits + 7) / 8;
    std::vector<uint8_t> bytes(num_bytes + 4, 0);

    uint32_t acc = 0;
    int acc_bits = 0;
    size_t byte_idx = 0;
    for (size_t d = 0; d < len; d++) {
        acc |= (static_cast<uint32_t>(digits[d]) << acc_bits);
        acc_bits += base_bits;
        while (acc_bits >= 8) {
            bytes[byte_idx++] = static_cast<uint8_t>(acc & 0xFF);
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    if (acc_bits > 0) bytes[byte_idx++] = static_cast<uint8_t>(acc & 0xFF);

    mpz_import(result, byte_idx, -1, 1, -1, 0, bytes.data());
}

void IntNttMultiplier::multiply(mpz_t result, const mpz_t a, const mpz_t b) {
    using Clock = std::chrono::high_resolution_clock;

    if (!should_use_gpu(a, b)) {
        stats_.cpu_fallback_calls.fetch_add(1);
        mpz_mul(result, a, b);
        return;
    }

    auto start = Clock::now();
    stats_.gpu_calls.fetch_add(1);

    int sign = mpz_sgn(a) * mpz_sgn(b);
    if (sign == 0) { mpz_set_ui(result, 0); return; }

    std::vector<uint32_t> da, db;
    mpz_to_digits(a, da, DIGIT_BASE_BITS);
    mpz_to_digits(b, db, DIGIT_BASE_BITS);

    size_t max_len = da.size() + db.size();
    std::vector<uint32_t> dr(max_len, 0);

    IntNttGpuContext& ctx = select_gpu();
    size_t actual_len;
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        actual_len = ctx.engine->multiply(
            da.data(), da.size(), db.data(), db.size(),
            dr.data(), max_len, DIGIT_BASE);
    }

    // If NTT returned 0, the arrays were too large — fall back to CPU
    if (actual_len == 0) {
        stats_.cpu_fallback_calls.fetch_add(1);
        mpz_mul(result, a, b);
        return;
    }

    digits_to_mpz(dr.data(), actual_len, result, DIGIT_BASE_BITS);
    if (sign < 0) mpz_neg(result, result);

    auto end = Clock::now();
    stats_.total_gpu_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

void IntNttMultiplier::square(mpz_t result, const mpz_t a) {
    multiply(result, a, a);
}

void IntNttMultiplier::print_stats() const {
    auto to_sec = [](uint64_t ns) { return static_cast<double>(ns) / 1e9; };
    uint64_t gpu = stats_.gpu_calls.load();
    uint64_t cpu = stats_.cpu_fallback_calls.load();
    double total = to_sec(stats_.total_gpu_ns.load());

    std::cout << "  Int NTT Multiplier Stats:" << std::endl;
    std::cout << "    GPU multiply calls:    " << gpu << std::endl;
    std::cout << "    CPU fallback calls:    " << cpu << std::endl;
    if (gpu > 0) {
        std::cout << "    Total GPU path time:   " << std::fixed << std::setprecision(3)
                  << total << "s" << std::endl;
        std::cout << "    Avg per GPU call:      " << std::setprecision(1)
                  << (total / gpu * 1000) << "ms" << std::endl;
    }
}

void IntNttMultiplier::reset_stats() {
    stats_.gpu_calls = 0;
    stats_.cpu_fallback_calls = 0;
    stats_.total_gpu_ns = 0;
}

} // namespace pi

#endif // PI_CUDA_ENABLED
