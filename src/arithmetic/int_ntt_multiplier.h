#pragma once

/**
 * @file int_ntt_multiplier.h
 * @brief GPU multiplication using integer NTT (no floating-point).
 *
 * Uses modular arithmetic over NTT-friendly primes instead of cuFFT.
 * No precision issues, uses INT64 throughput instead of FP64.
 */

#ifdef PI_CUDA_ENABLED

#include "multiplier.h"
#include "../gpu/int_ntt.h"
#include <cstddef>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <atomic>
#include <chrono>

namespace pi {

struct IntNttGpuContext {
    std::unique_ptr<gpu::IntNttEngine> engine;
    std::mutex mutex;
    int device_id;
    explicit IntNttGpuContext(int id) : device_id(id) {}
};

class IntNttMultiplier : public Multiplier {
public:
    explicit IntNttMultiplier(size_t threshold = 10000, int num_gpus = 0);
    ~IntNttMultiplier() override = default;

    void multiply(mpz_t result, const mpz_t a, const mpz_t b) override;
    void square(mpz_t result, const mpz_t a) override;

    std::string device_name() const;
    int gpu_count() const { return static_cast<int>(gpu_contexts_.size()); }
    size_t threshold() const { return threshold_; }

    void print_stats() const;
    void reset_stats();

    struct Stats {
        std::atomic<uint64_t> gpu_calls{0};
        std::atomic<uint64_t> cpu_fallback_calls{0};
        std::atomic<uint64_t> total_gpu_ns{0};
    };

private:
    std::vector<std::unique_ptr<IntNttGpuContext>> gpu_contexts_;
    size_t threshold_;
    std::atomic<uint64_t> next_gpu_{0};
    mutable Stats stats_;

    static void mpz_to_digits(const mpz_t n, std::vector<uint32_t>& digits, uint32_t base_bits);
    static void digits_to_mpz(const uint32_t* digits, size_t len, mpz_t result, uint32_t base_bits);
    bool should_use_gpu(const mpz_t a, const mpz_t b) const;
    IntNttGpuContext& select_gpu();
};

} // namespace pi

#endif // PI_CUDA_ENABLED
