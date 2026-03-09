#pragma once

/**
 * @file gpu_ntt_multiplier.h
 * @brief GPU-accelerated multiplication using cuFFT-based NTT.
 *
 * Supports multi-GPU: auto-detects available GPUs and creates one NTT engine
 * per GPU. Multiplications are dispatched to GPUs round-robin, with per-GPU
 * mutexes for thread safety.
 *
 * On single-GPU systems, behaves identically to a simple serialized multiplier.
 */

#ifdef PI_CUDA_ENABLED

#include "multiplier.h"
#include "../gpu/ntt_engine.h"
#include <cstddef>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <atomic>

namespace pi {

/**
 * @brief Per-GPU context: NTT engine + mutex for thread-safe access.
 */
struct GpuContext {
    std::unique_ptr<gpu::NttEngine> engine;
    std::mutex mutex;
    int device_id;

    explicit GpuContext(int id) : device_id(id) {}
};

class GpuNttMultiplier : public Multiplier {
public:
    /**
     * @brief Construct a GPU multiplier.
     * @param threshold Minimum number of GMP limbs for GPU path.
     *                  Below this, falls back to GMP CPU multiply.
     * @param num_gpus Number of GPUs to use (0 = auto-detect all available)
     */
    explicit GpuNttMultiplier(size_t threshold = 10000, int num_gpus = 0);
    ~GpuNttMultiplier() override = default;

    void multiply(mpz_t result, const mpz_t a, const mpz_t b) override;
    void square(mpz_t result, const mpz_t a) override;

    /**
     * @brief Get the GPU device name(s).
     */
    std::string device_name() const;

    /**
     * @brief Print timing statistics for GPU operations.
     */
    void print_stats() const;

    /**
     * @brief Reset timing statistics.
     */
    void reset_stats();

    /**
     * @brief Get number of GPUs in use.
     */
    int gpu_count() const { return static_cast<int>(gpu_contexts_.size()); }

    /**
     * @brief Set the limb threshold for GPU vs CPU selection.
     */
    void set_threshold(size_t threshold) { threshold_ = threshold; }

    /**
     * @brief Get current threshold.
     */
    size_t threshold() const { return threshold_; }

    /// Timing statistics (accumulated across all multiply calls)
    struct Stats {
        std::atomic<uint64_t> gpu_calls{0};
        std::atomic<uint64_t> cpu_fallback_calls{0};
        std::atomic<uint64_t> convert_to_ns{0};    ///< GMP → base-2^12 conversion
        std::atomic<uint64_t> gpu_compute_ns{0};    ///< GPU FFT multiply (including transfer)
        std::atomic<uint64_t> convert_from_ns{0};   ///< base-2^12 → GMP conversion
        std::atomic<uint64_t> total_gpu_ns{0};      ///< Total time in GPU multiply path
    };

private:
    std::vector<std::unique_ptr<GpuContext>> gpu_contexts_;
    size_t threshold_;
    std::atomic<uint64_t> next_gpu_{0};
    mutable Stats stats_;

    static void mpz_to_base15(const mpz_t n, std::vector<uint32_t>& digits);
    static void base15_to_mpz(const uint32_t* digits, size_t len, mpz_t result);
    bool should_use_gpu(const mpz_t a, const mpz_t b) const;
    GpuContext& select_gpu();

};

} // namespace pi

#endif // PI_CUDA_ENABLED
