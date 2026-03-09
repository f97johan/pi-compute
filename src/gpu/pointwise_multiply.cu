/**
 * @file pointwise_multiply.cu
 * @brief CUDA kernels for pointwise complex multiplication and normalization.
 */

#ifdef PI_CUDA_ENABLED

#include "pointwise_multiply.h"
#include <cuda_runtime.h>
#include <cmath>

namespace pi {
namespace gpu {

/**
 * @brief CUDA kernel: pointwise complex multiplication c = a * b
 *
 * Complex multiplication: (a.x + i*a.y) * (b.x + i*b.y)
 *   = (a.x*b.x - a.y*b.y) + i*(a.x*b.y + a.y*b.x)
 */
__global__ void pointwise_multiply_kernel(const cufftDoubleComplex* a,
                                           const cufftDoubleComplex* b,
                                           cufftDoubleComplex* c,
                                           size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double ar = a[idx].x, ai = a[idx].y;
        double br = b[idx].x, bi = b[idx].y;
        c[idx].x = ar * br - ai * bi;
        c[idx].y = ar * bi + ai * br;
    }
}

/**
 * @brief CUDA kernel: extract real parts and normalize by 1/n.
 *
 * After inverse FFT, the real parts contain the convolution values
 * (scaled by n because cuFFT's inverse is unnormalized).
 * We divide by n and round to nearest integer.
 */
__global__ void extract_normalize_kernel(const cufftDoubleComplex* complex_data,
                                          double* real_data,
                                          size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Normalize by dividing by FFT size, then round
        real_data[idx] = round(complex_data[idx].x / static_cast<double>(n));
    }
}

void pointwise_complex_multiply(const cufftDoubleComplex* a,
                                 const cufftDoubleComplex* b,
                                 cufftDoubleComplex* c,
                                 size_t n) {
    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    pointwise_multiply_kernel<<<grid_size, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();
}

void extract_and_normalize(const cufftDoubleComplex* complex_data,
                           double* real_data,
                           size_t n) {
    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    extract_normalize_kernel<<<grid_size, block_size>>>(complex_data, real_data, n);
    cudaDeviceSynchronize();
}

} // namespace gpu
} // namespace pi

#endif // PI_CUDA_ENABLED
