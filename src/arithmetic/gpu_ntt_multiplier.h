#pragma once

/**
 * @file gpu_ntt_multiplier.h
 * @brief GPU-accelerated multiplication using cuFFT-based NTT.
 *
 * Implements the Multiplier interface, converting GMP mpz_t integers
 * to base-2^24 digit arrays, performing FFT convolution on GPU,
 * and converting back to GMP format.
 *
 * Only available when compiled with ENABLE_CUDA=ON.
 */

#ifdef PI_CUDA_ENABLED

#include "multiplier.h"
#include "../gpu/ntt_engine.h"
#include <cstddef>

namespace pi {

class GpuNttMultiplier : public Multiplier {
public:
    /**
     * @brief Construct a GPU multiplier.
     * @param threshold Minimum number of GMP limbs for GPU path.
     *                  Below this, falls back to GMP CPU multiply.
     */
    explicit GpuNttMultiplier(size_t threshold = 1000);
    ~GpuNttMultiplier() override = default;

    void multiply(mpz_t result, const mpz_t a, const mpz_t b) override;
    void square(mpz_t result, const mpz_t a) override;

    /**
     * @brief Get the GPU device name.
     */
    std::string device_name() const;

    /**
     * @brief Set the limb threshold for GPU vs CPU selection.
     */
    void set_threshold(size_t threshold) { threshold_ = threshold; }

    /**
     * @brief Get current threshold.
     */
    size_t threshold() const { return threshold_; }

private:
    gpu::NttEngine ntt_engine_;
    size_t threshold_;

    /**
     * @brief Convert GMP mpz_t to base-2^24 digit array.
     * @param n The GMP integer (absolute value is used)
     * @param digits Output vector of base-2^24 digits (LSB first)
     */
    static void mpz_to_base24(const mpz_t n, std::vector<uint32_t>& digits);

    /**
     * @brief Convert base-2^24 digit array back to GMP mpz_t.
     * @param digits Base-2^24 digits (LSB first)
     * @param len Number of digits
     * @param result Output GMP integer
     */
    static void base24_to_mpz(const uint32_t* digits, size_t len, mpz_t result);

    /**
     * @brief Check if operands are large enough to benefit from GPU.
     */
    bool should_use_gpu(const mpz_t a, const mpz_t b) const;
};

} // namespace pi

#endif // PI_CUDA_ENABLED
