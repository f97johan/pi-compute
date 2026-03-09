#pragma once

/**
 * @file ntt_engine.h
 * @brief cuFFT-based Number Theoretic Transform for large integer multiplication.
 *
 * Uses double-precision complex FFT (cuFFT) to perform convolution-based
 * multiplication. Numbers are represented in base 2^24 so that products
 * of two digits (2^48) fit within the 53-bit mantissa of IEEE 754 double.
 *
 * Flow:
 *   1. Forward FFT of both operands
 *   2. Pointwise complex multiplication
 *   3. Inverse FFT
 *   4. Round to nearest integer and propagate carries
 */

#ifdef PI_CUDA_ENABLED

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <cufft.h>

namespace pi {
namespace gpu {

/**
 * @brief Wrapper around cuFFT for forward/inverse transforms.
 *
 * Manages cuFFT plans and device memory for double-precision complex transforms.
 */
class NttEngine {
public:
    /**
     * @brief Construct an NTT engine bound to a specific GPU.
     * @param device_id CUDA device ID (default: 0)
     */
    explicit NttEngine(int device_id = 0);
    ~NttEngine();

    // Non-copyable
    NttEngine(const NttEngine&) = delete;
    NttEngine& operator=(const NttEngine&) = delete;

    /**
     * @brief Multiply two arrays of base-2^24 digits using FFT convolution.
     *
     * @param a First operand digits (base 2^24, least-significant first)
     * @param a_len Number of digits in a
     * @param b Second operand digits (base 2^24, least-significant first)
     * @param b_len Number of digits in b
     * @param result Output array (must be pre-allocated, size >= a_len + b_len)
     * @param result_len Size of result array
     * @return Actual number of result digits used
     */
    size_t multiply(const uint32_t* a, size_t a_len,
                    const uint32_t* b, size_t b_len,
                    uint32_t* result, size_t result_len);

    /**
     * @brief Get the GPU device name for logging.
     */
    static std::string get_device_name();

    /**
     * @brief Check if a CUDA-capable GPU is available.
     */
    static bool is_available();

private:
    /**
     * @brief Ensure FFT plan is created for the given size.
     */
    void ensure_plan(size_t fft_size);

    /**
     * @brief Round up to next power of 2.
     */
    static size_t next_power_of_2(size_t n);

    int device_id_ = 0;  ///< CUDA device this engine is bound to
    cufftHandle plan_ = 0;
    size_t current_plan_size_ = 0;

    // Device memory pointers (reused across calls)
    cufftDoubleComplex* d_a_ = nullptr;
    cufftDoubleComplex* d_b_ = nullptr;
    cufftDoubleComplex* d_c_ = nullptr;
    double* d_result_ = nullptr;
    size_t allocated_size_ = 0;

    /**
     * @brief Set the CUDA device for this engine's operations.
     */
    void activate_device() const;
};

} // namespace gpu
} // namespace pi

#endif // PI_CUDA_ENABLED
