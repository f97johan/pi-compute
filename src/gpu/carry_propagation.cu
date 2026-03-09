/**
 * @file carry_propagation.cu
 * @brief Carry propagation implementation.
 *
 * Carry propagation is inherently sequential (each digit depends on the
 * carry from the previous digit). We do this on the CPU after copying
 * the convolution result back from GPU. For our target scale (10M-100M digits),
 * carry propagation is a tiny fraction of total time vs the FFT.
 *
 * A GPU-parallel carry propagation using prefix sums is possible but adds
 * complexity for minimal gain at our scale.
 */

#ifdef PI_CUDA_ENABLED

#include "carry_propagation.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace pi {
namespace gpu {

size_t propagate_carries_gpu(const double* d_conv_result,
                              uint32_t* h_result,
                              size_t len,
                              uint64_t base) {
    // Copy convolution result from GPU to host
    std::vector<double> h_conv(len);
    cudaError_t err = cudaMemcpy(h_conv.data(), d_conv_result,
                                  len * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed in carry propagation");
    }

    // Sequential carry propagation on CPU
    // Each element is a potentially large double that needs to be reduced mod base
    int64_t carry = 0;
    size_t actual_len = 0;

    for (size_t i = 0; i < len; ++i) {
        int64_t val = static_cast<int64_t>(llround(h_conv[i])) + carry;

        if (val >= 0) {
            h_result[i] = static_cast<uint32_t>(val % static_cast<int64_t>(base));
            carry = val / static_cast<int64_t>(base);
        } else {
            // Handle negative values (shouldn't happen in normal multiplication,
            // but can occur due to floating-point rounding in FFT)
            int64_t adjusted = val + static_cast<int64_t>(base) * ((-val / static_cast<int64_t>(base)) + 1);
            h_result[i] = static_cast<uint32_t>(adjusted % static_cast<int64_t>(base));
            carry = (val - static_cast<int64_t>(h_result[i])) / static_cast<int64_t>(base);
        }

        if (h_result[i] != 0 || carry != 0) {
            actual_len = i + 1;
        }
    }

    // Handle remaining carry
    while (carry > 0 && actual_len < len) {
        h_result[actual_len] = static_cast<uint32_t>(carry % static_cast<int64_t>(base));
        carry /= static_cast<int64_t>(base);
        actual_len++;
    }

    return actual_len;
}

} // namespace gpu
} // namespace pi

#endif // PI_CUDA_ENABLED
