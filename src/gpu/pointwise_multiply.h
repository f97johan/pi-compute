#pragma once

/**
 * @file pointwise_multiply.h
 * @brief CUDA kernel for pointwise complex multiplication in frequency domain.
 */

#ifdef PI_CUDA_ENABLED

#include <cufft.h>
#include <cstddef>

namespace pi {
namespace gpu {

/**
 * @brief Pointwise multiply two complex arrays: c[i] = a[i] * b[i]
 * @param a First frequency-domain array
 * @param b Second frequency-domain array
 * @param c Output array
 * @param n Number of elements
 */
void pointwise_complex_multiply(const cufftDoubleComplex* a,
                                 const cufftDoubleComplex* b,
                                 cufftDoubleComplex* c,
                                 size_t n);

/**
 * @brief Extract real parts from complex array and normalize by 1/n.
 * @param complex_data Input complex array (result of inverse FFT)
 * @param real_data Output double array (rounded convolution values)
 * @param n Number of elements
 */
void extract_and_normalize(const cufftDoubleComplex* complex_data,
                           double* real_data,
                           size_t n);

} // namespace gpu
} // namespace pi

#endif // PI_CUDA_ENABLED
