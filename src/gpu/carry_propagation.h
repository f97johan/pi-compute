#pragma once

/**
 * @file carry_propagation.h
 * @brief CUDA-accelerated carry propagation for base-2^24 digit arrays.
 *
 * After FFT convolution, each element may exceed the base (2^24).
 * Carry propagation ripples excess values from lower to higher digits.
 */

#ifdef PI_CUDA_ENABLED

#include <cstddef>
#include <cstdint>

namespace pi {
namespace gpu {

/**
 * @brief Propagate carries through a convolution result array.
 *
 * Takes double-precision convolution results (on GPU), converts to
 * base-2^24 digits with carry propagation, and copies result to host.
 *
 * @param d_conv_result Device array of convolution values (doubles)
 * @param h_result Host output array of base-2^24 digits
 * @param len Number of elements in convolution result
 * @param base The number base (2^24 = 16777216)
 * @return Actual number of digits in result (may be less than len)
 */
size_t propagate_carries_gpu(const double* d_conv_result,
                              uint32_t* h_result,
                              size_t len,
                              uint64_t base);

} // namespace gpu
} // namespace pi

#endif // PI_CUDA_ENABLED
