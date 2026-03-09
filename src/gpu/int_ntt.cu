/**
 * @file int_ntt.cu
 * @brief Integer NTT implementation using CUDA.
 *
 * Core operations:
 * 1. Modular multiplication: (a * b) % p using __uint128_t or Barrett reduction
 * 2. Modular exponentiation: a^e % p (for computing roots of unity)
 * 3. NTT butterfly: in-place Cooley-Tukey butterfly with modular arithmetic
 * 4. Pointwise modular multiply
 * 5. Chinese Remainder Theorem to combine results from 3 primes
 */

#ifdef PI_CUDA_ENABLED

#include "int_ntt.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) \
            + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

namespace pi {
namespace gpu {

// ============================================================
// Device functions: modular arithmetic
// ============================================================

/**
 * @brief Modular multiplication: (a * b) % mod
 * Uses 64-bit multiplication with careful overflow handling.
 * For mod < 2^31, a*b fits in uint64_t.
 */
__device__ __forceinline__
uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    // Since our primes are < 2^30, and a,b < mod, a*b < 2^60 — fits in uint64_t
    return (a * b) % mod;
}

/**
 * @brief Modular exponentiation: base^exp % mod
 */
__device__
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, mod);
        }
        exp >>= 1;
        base = mod_mul(base, base, mod);
    }
    return result;
}

// Host version of mod_pow for computing roots of unity
static uint64_t host_mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = (__uint128_t)result * base % mod;
        exp >>= 1;
        base = (__uint128_t)base * base % mod;
    }
    return result;
}

// ============================================================
// NTT Butterfly Kernel
// ============================================================

/**
 * @brief In-place NTT butterfly kernel (one layer of the transform).
 *
 * For each butterfly pair (i, i + half):
 *   u = a[i]
 *   v = a[i + half] * w[k]
 *   a[i]        = (u + v) % mod
 *   a[i + half] = (u - v + mod) % mod
 *
 * @param a The array being transformed (in-place)
 * @param n Total array size (power of 2)
 * @param half Half the current butterfly span
 * @param mod The prime modulus
 * @param root_pow The twiddle factor base for this layer
 */
__global__
void ntt_butterfly_kernel(uint64_t* a, size_t n, size_t half,
                           uint64_t mod, const uint64_t* twiddles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_butterflies = n / 2;

    if (idx >= total_butterflies) return;

    // Determine which butterfly group and position within group
    size_t group_size = half * 2;
    size_t group = idx / half;
    size_t pos = idx % half;

    size_t i = group * group_size + pos;
    size_t j = i + half;

    uint64_t u = a[i];
    uint64_t v = mod_mul(a[j], twiddles[pos], mod);

    a[i] = (u + v) % mod;
    a[j] = (u - v + mod) % mod;
}

/**
 * @brief Pointwise modular multiplication: c[i] = (a[i] * b[i]) % mod
 */
__global__
void pointwise_mod_mul_kernel(const uint64_t* a, const uint64_t* b,
                               uint64_t* c, size_t n, uint64_t mod) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = mod_mul(a[idx], b[idx], mod);
    }
}

/**
 * @brief Scale array by inverse of n: a[i] = (a[i] * inv_n) % mod
 */
__global__
void scale_kernel(uint64_t* a, size_t n, uint64_t inv_n, uint64_t mod) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = mod_mul(a[idx], inv_n, mod);
    }
}

/**
 * @brief Bit-reversal permutation
 */
__global__
void bit_reverse_kernel(uint64_t* a, size_t n, int log_n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Compute bit-reversed index
    size_t rev = 0;
    size_t tmp = idx;
    for (int i = 0; i < log_n; i++) {
        rev = (rev << 1) | (tmp & 1);
        tmp >>= 1;
    }

    // Only swap if rev > idx (to avoid double-swapping)
    if (rev > idx) {
        uint64_t temp = a[idx];
        a[idx] = a[rev];
        a[rev] = temp;
    }
}

// ============================================================
// Host-side NTT orchestration
// ============================================================

/**
 * @brief Perform forward or inverse NTT on device array.
 */
static void perform_ntt(uint64_t* d_a, size_t n, uint64_t prime, uint64_t prim_root,
                         bool inverse, int device_id) {
    int log_n = 0;
    size_t tmp = n;
    while (tmp > 1) { log_n++; tmp >>= 1; }

    const int block_size = 256;

    // Bit-reversal permutation
    int grid_br = (n + block_size - 1) / block_size;
    bit_reverse_kernel<<<grid_br, block_size>>>(d_a, n, log_n);

    // Compute the principal n-th root of unity
    // g is a primitive root of the prime
    // w = g^((p-1)/n) mod p is the n-th root of unity
    uint64_t w = host_mod_pow(prim_root, (prime - 1) / n, prime);
    if (inverse) {
        // For inverse NTT, use w^(-1)
        w = host_mod_pow(w, prime - 2, prime);  // Fermat's little theorem
    }

    // For each NTT layer
    for (int s = 0; s < log_n; s++) {
        size_t half = 1ULL << s;
        size_t group_size = half * 2;

        // Precompute twiddle factors for this layer
        std::vector<uint64_t> twiddles(half);
        uint64_t wk = 1;
        uint64_t w_step = host_mod_pow(w, n / group_size, prime);
        for (size_t k = 0; k < half; k++) {
            twiddles[k] = wk;
            wk = (__uint128_t)wk * w_step % prime;
        }

        // Upload twiddle factors
        uint64_t* d_twiddles;
        CUDA_CHECK(cudaMalloc(&d_twiddles, half * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_twiddles, twiddles.data(), half * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        // Launch butterfly kernel
        size_t total_butterflies = n / 2;
        int grid = (total_butterflies + block_size - 1) / block_size;
        ntt_butterfly_kernel<<<grid, block_size>>>(d_a, n, half, prime, d_twiddles);

        cudaFree(d_twiddles);
    }

    // For inverse NTT, scale by 1/n
    if (inverse) {
        uint64_t inv_n = host_mod_pow(n, prime - 2, prime);
        int grid = (n + block_size - 1) / block_size;
        scale_kernel<<<grid, block_size>>>(d_a, n, inv_n, prime);
    }

    cudaDeviceSynchronize();
}

// ============================================================
// Chinese Remainder Theorem
// ============================================================

/**
 * @brief CRT to combine results from 3 primes into a single value.
 *
 * Given r0 mod p0, r1 mod p1, r2 mod p2, compute the unique value
 * mod (p0*p1*p2) using Garner's algorithm.
 *
 * Returns the result as a signed int64 (the convolution value before carry propagation).
 */
static int64_t crt_combine(uint64_t r0, uint64_t r1, uint64_t r2,
                            uint64_t p0, uint64_t p1, uint64_t p2) {
    // Garner's algorithm
    // x = r0
    // x = r0 + p0 * ((r1 - r0) * inv(p0, p1) % p1)
    // x = above + p0*p1 * ((r2 - x_mod_p2) * inv(p0*p1, p2) % p2)

    uint64_t inv_p0_mod_p1 = host_mod_pow(p0, p1 - 2, p1);
    uint64_t inv_p0p1_mod_p2 = host_mod_pow((__uint128_t)p0 * p1 % p2, p2 - 2, p2);

    // Step 1: x1 = (r1 - r0) * inv(p0, p1) mod p1
    uint64_t diff1 = (r1 + p1 - r0 % p1) % p1;
    uint64_t x1 = (__uint128_t)diff1 * inv_p0_mod_p1 % p1;

    // Partial result: val = r0 + p0 * x1
    __uint128_t val = r0 + (__uint128_t)p0 * x1;

    // Step 2: x2 = (r2 - val % p2) * inv(p0*p1, p2) mod p2
    uint64_t val_mod_p2 = val % p2;
    uint64_t diff2 = (r2 + p2 - val_mod_p2) % p2;
    uint64_t x2 = (__uint128_t)diff2 * inv_p0p1_mod_p2 % p2;

    // Final: result = val + p0*p1*x2
    __uint128_t result = val + (__uint128_t)p0 * p1 * x2;

    // The convolution values should be small relative to p0*p1*p2
    // Convert to signed (values near p0*p1*p2 are negative)
    __uint128_t half_product = (__uint128_t)p0 * p1 * p2 / 2;
    if (result > half_product) {
        return -static_cast<int64_t>((__uint128_t)p0 * p1 * p2 - result);
    }
    return static_cast<int64_t>(result);
}

// ============================================================
// IntNttEngine implementation
// ============================================================

IntNttEngine::IntNttEngine(int device_id) : device_id_(device_id) {}

IntNttEngine::~IntNttEngine() {
    for (int i = 0; i < NUM_PRIMES; i++) {
        if (buffers_[i].d_a) cudaFree(buffers_[i].d_a);
        if (buffers_[i].d_b) cudaFree(buffers_[i].d_b);
    }
}

void IntNttEngine::activate_device() const {
    cudaSetDevice(device_id_);
}

bool IntNttEngine::is_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string IntNttEngine::get_device_name() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "unknown";
    return std::string(prop.name);
}

size_t IntNttEngine::next_power_of_2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

void IntNttEngine::ensure_buffers(size_t ntt_size) {
    if (ntt_size <= current_ntt_size_) return;

    activate_device();

    for (int i = 0; i < NUM_PRIMES; i++) {
        if (buffers_[i].d_a) cudaFree(buffers_[i].d_a);
        if (buffers_[i].d_b) cudaFree(buffers_[i].d_b);

        size_t bytes = ntt_size * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&buffers_[i].d_a, bytes));
        CUDA_CHECK(cudaMalloc(&buffers_[i].d_b, bytes));
        buffers_[i].allocated = ntt_size;
    }

    current_ntt_size_ = ntt_size;
}

size_t IntNttEngine::multiply(const uint32_t* a, size_t a_len,
                               const uint32_t* b, size_t b_len,
                               uint32_t* result, size_t result_len,
                               uint64_t base) {
    size_t conv_len = a_len + b_len;
    size_t ntt_size = next_power_of_2(conv_len);

    // Check NTT size doesn't exceed prime limits
    for (int i = 0; i < NUM_PRIMES; i++) {
        if (ntt_size > (1ULL << NTT_PRIMES[i].max_log2)) {
            throw std::runtime_error("NTT size exceeds prime limit");
        }
    }

    activate_device();
    ensure_buffers(ntt_size);

    // For each prime: NTT(a) * NTT(b), then inverse NTT
    // We store the results on host for CRT combination
    std::vector<std::vector<uint64_t>> conv_results(NUM_PRIMES);

    for (int pi = 0; pi < NUM_PRIMES; pi++) {
        uint64_t prime = NTT_PRIMES[pi].p;
        uint64_t prim_root = NTT_PRIMES[pi].g;

        // Prepare host arrays (zero-padded, reduced mod prime)
        std::vector<uint64_t> h_a(ntt_size, 0);
        std::vector<uint64_t> h_b(ntt_size, 0);
        for (size_t i = 0; i < a_len; i++) h_a[i] = a[i] % prime;
        for (size_t i = 0; i < b_len; i++) h_b[i] = b[i] % prime;

        size_t bytes = ntt_size * sizeof(uint64_t);

        // Upload to GPU
        CUDA_CHECK(cudaMemcpy(buffers_[pi].d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buffers_[pi].d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

        // Forward NTT of both
        perform_ntt(buffers_[pi].d_a, ntt_size, prime, prim_root, false, device_id_);
        perform_ntt(buffers_[pi].d_b, ntt_size, prime, prim_root, false, device_id_);

        // Pointwise multiply
        int block_size = 256;
        int grid = (ntt_size + block_size - 1) / block_size;
        pointwise_mod_mul_kernel<<<grid, block_size>>>(
            buffers_[pi].d_a, buffers_[pi].d_b, buffers_[pi].d_a, ntt_size, prime);

        // Inverse NTT
        perform_ntt(buffers_[pi].d_a, ntt_size, prime, prim_root, true, device_id_);

        // Download result
        conv_results[pi].resize(ntt_size);
        CUDA_CHECK(cudaMemcpy(conv_results[pi].data(), buffers_[pi].d_a, bytes,
                              cudaMemcpyDeviceToHost));
    }

    // CRT combination + carry propagation
    int64_t carry = 0;
    size_t actual_len = 0;

    for (size_t i = 0; i < conv_len; i++) {
        int64_t val = crt_combine(
            conv_results[0][i], conv_results[1][i], conv_results[2][i],
            NTT_PRIMES[0].p, NTT_PRIMES[1].p, NTT_PRIMES[2].p
        ) + carry;

        if (val >= 0) {
            result[i] = static_cast<uint32_t>(val % static_cast<int64_t>(base));
            carry = val / static_cast<int64_t>(base);
        } else {
            int64_t adjusted = val + static_cast<int64_t>(base) * ((-val / static_cast<int64_t>(base)) + 1);
            result[i] = static_cast<uint32_t>(adjusted % static_cast<int64_t>(base));
            carry = (val - static_cast<int64_t>(result[i])) / static_cast<int64_t>(base);
        }

        if (result[i] != 0 || carry != 0) actual_len = i + 1;
    }

    while (carry > 0 && actual_len < result_len) {
        result[actual_len] = static_cast<uint32_t>(carry % static_cast<int64_t>(base));
        carry /= static_cast<int64_t>(base);
        actual_len++;
    }

    return actual_len;
}

} // namespace gpu
} // namespace pi

#endif // PI_CUDA_ENABLED
