/**
 * @file ntt_engine.cu
 * @brief cuFFT-based NTT multiplication implementation.
 */

#ifdef PI_CUDA_ENABLED

#include "ntt_engine.h"
#include "pointwise_multiply.h"
#include "carry_propagation.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) \
            + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult err = (call); \
    if (err != CUFFT_SUCCESS) { \
        throw std::runtime_error(std::string("cuFFT error: ") + std::to_string(err) \
            + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

namespace pi {
namespace gpu {

NttEngine::NttEngine() {}

NttEngine::~NttEngine() {
    if (plan_ != 0) {
        cufftDestroy(plan_);
    }
    if (d_a_) cudaFree(d_a_);
    if (d_b_) cudaFree(d_b_);
    if (d_c_) cudaFree(d_c_);
    if (d_result_) cudaFree(d_result_);
}

bool NttEngine::is_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

std::string NttEngine::get_device_name() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return "unknown";
    return std::string(prop.name);
}

size_t NttEngine::next_power_of_2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

void NttEngine::ensure_plan(size_t fft_size) {
    if (fft_size == current_plan_size_) return;

    // Destroy old plan
    if (plan_ != 0) {
        cufftDestroy(plan_);
        plan_ = 0;
    }

    // Create new plan for complex-to-complex double-precision FFT
    CUFFT_CHECK(cufftPlan1d(&plan_, static_cast<int>(fft_size), CUFFT_Z2Z, 1));
    current_plan_size_ = fft_size;

    // Reallocate device memory if needed
    if (fft_size > allocated_size_) {
        if (d_a_) cudaFree(d_a_);
        if (d_b_) cudaFree(d_b_);
        if (d_c_) cudaFree(d_c_);
        if (d_result_) cudaFree(d_result_);

        size_t complex_bytes = fft_size * sizeof(cufftDoubleComplex);
        size_t double_bytes = fft_size * sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_a_, complex_bytes));
        CUDA_CHECK(cudaMalloc(&d_b_, complex_bytes));
        CUDA_CHECK(cudaMalloc(&d_c_, complex_bytes));
        CUDA_CHECK(cudaMalloc(&d_result_, double_bytes));

        allocated_size_ = fft_size;
    }
}

size_t NttEngine::multiply(const uint32_t* a, size_t a_len,
                           const uint32_t* b, size_t b_len,
                           uint32_t* result, size_t result_len) {
    // The convolution result has length a_len + b_len - 1
    size_t conv_len = a_len + b_len;
    size_t fft_size = next_power_of_2(conv_len);

    ensure_plan(fft_size);

    // Prepare host complex arrays (zero-padded)
    std::vector<cufftDoubleComplex> h_a(fft_size, {0.0, 0.0});
    std::vector<cufftDoubleComplex> h_b(fft_size, {0.0, 0.0});

    // Convert uint32 digits to complex (real part = digit value, imag = 0)
    for (size_t i = 0; i < a_len; ++i) {
        h_a[i].x = static_cast<double>(a[i]);
        h_a[i].y = 0.0;
    }
    for (size_t i = 0; i < b_len; ++i) {
        h_b[i].x = static_cast<double>(b[i]);
        h_b[i].y = 0.0;
    }

    size_t complex_bytes = fft_size * sizeof(cufftDoubleComplex);

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a_, h_a.data(), complex_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_, h_b.data(), complex_bytes, cudaMemcpyHostToDevice));

    // Forward FFT of both operands
    CUFFT_CHECK(cufftExecZ2Z(plan_, d_a_, d_a_, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecZ2Z(plan_, d_b_, d_b_, CUFFT_FORWARD));

    // Pointwise complex multiplication
    pointwise_complex_multiply(d_a_, d_b_, d_c_, fft_size);

    // Inverse FFT
    CUFFT_CHECK(cufftExecZ2Z(plan_, d_c_, d_c_, CUFFT_INVERSE));

    // Extract real parts and normalize (divide by fft_size)
    // The inverse FFT in cuFFT is unnormalized, so we divide by N
    extract_and_normalize(d_c_, d_result_, fft_size);

    // Carry propagation: convert from double to uint32 with carries
    // Base is 2^15 = 32768 (chosen so that B^2 * N < 2^53 for large FFTs)
    // With B=2^15 and N=2^20: (2^15)^2 * 2^20 = 2^50 < 2^53 ✓
    size_t actual_len = propagate_carries_gpu(d_result_, result, conv_len, 32768);

    return actual_len;
}

} // namespace gpu
} // namespace pi

#endif // PI_CUDA_ENABLED
