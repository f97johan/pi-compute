/**
 * @file int_ntt.cu
 * @brief Integer NTT with precomputed twiddle factors.
 *
 * Optimization: all twiddle factors for all layers and all primes are
 * precomputed once and stored persistently on GPU. The NTT transform
 * then only launches kernels — no cudaMalloc/cudaMemcpy/cudaFree per layer.
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

// Host modular exponentiation using __uint128_t for overflow safety
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
// CUDA Kernels
// ============================================================

__device__ __forceinline__
uint64_t dev_mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    // For mod < 2^32, a*b < 2^64 — fits in uint64_t
    return (a * b) % mod;
}

/**
 * @brief NTT butterfly kernel using precomputed twiddle factors.
 * twiddles points to the twiddle array for this specific layer.
 */
__global__
void ntt_butterfly_kernel(uint64_t* a, size_t n, size_t half,
                           uint64_t mod, const uint64_t* twiddles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return;

    size_t group_size = half * 2;
    size_t group = idx / half;
    size_t pos = idx % half;
    size_t i = group * group_size + pos;
    size_t j = i + half;

    uint64_t u = a[i];
    uint64_t v = dev_mod_mul(a[j], twiddles[pos], mod);
    a[i] = (u + v) % mod;
    a[j] = (u - v + mod) % mod;
}

__global__
void pointwise_mod_mul_kernel(const uint64_t* a, const uint64_t* b,
                               uint64_t* c, size_t n, uint64_t mod) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = dev_mod_mul(a[idx], b[idx], mod);
}

__global__
void scale_kernel(uint64_t* a, size_t n, uint64_t inv_n, uint64_t mod) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] = dev_mod_mul(a[idx], inv_n, mod);
}

__global__
void bit_reverse_kernel(uint64_t* a, size_t n, int log_n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t rev = 0;
    size_t tmp = idx;
    for (int i = 0; i < log_n; i++) {
        rev = (rev << 1) | (tmp & 1);
        tmp >>= 1;
    }
    if (rev > idx) {
        uint64_t temp = a[idx];
        a[idx] = a[rev];
        a[rev] = temp;
    }
}

// ============================================================
// Precomputed twiddle factors
// ============================================================

struct TwiddleCache {
    // For each prime, for each layer, store device pointer to twiddle array
    // twiddles[prime_idx][layer] points to GPU memory of size 2^layer
    std::vector<std::vector<uint64_t*>> d_twiddles_fwd;  // Forward NTT
    std::vector<std::vector<uint64_t*>> d_twiddles_inv;  // Inverse NTT
    int log_n = 0;
    size_t ntt_size = 0;

    void build(size_t n, int device_id) {
        if (n == ntt_size) return;
        clear();

        cudaSetDevice(device_id);
        ntt_size = n;
        log_n = 0;
        size_t tmp = n;
        while (tmp > 1) { log_n++; tmp >>= 1; }

        d_twiddles_fwd.resize(NUM_PRIMES);
        d_twiddles_inv.resize(NUM_PRIMES);

        for (int pi = 0; pi < NUM_PRIMES; pi++) {
            uint64_t prime = NTT_PRIMES[pi].p;
            uint64_t prim_root = NTT_PRIMES[pi].g;

            // Forward root of unity: w = g^((p-1)/n) mod p
            uint64_t w_fwd = host_mod_pow(prim_root, (prime - 1) / n, prime);
            // Inverse root: w^(-1)
            uint64_t w_inv = host_mod_pow(w_fwd, prime - 2, prime);

            d_twiddles_fwd[pi].resize(log_n);
            d_twiddles_inv[pi].resize(log_n);

            for (int s = 0; s < log_n; s++) {
                size_t half = 1ULL << s;
                size_t group_size = half * 2;

                // Compute twiddle factors for this layer
                std::vector<uint64_t> tw_fwd(half), tw_inv(half);

                uint64_t w_step_fwd = host_mod_pow(w_fwd, n / group_size, prime);
                uint64_t w_step_inv = host_mod_pow(w_inv, n / group_size, prime);

                uint64_t wk = 1;
                for (size_t k = 0; k < half; k++) {
                    tw_fwd[k] = wk;
                    wk = (__uint128_t)wk * w_step_fwd % prime;
                }
                wk = 1;
                for (size_t k = 0; k < half; k++) {
                    tw_inv[k] = wk;
                    wk = (__uint128_t)wk * w_step_inv % prime;
                }

                // Upload to GPU (persistent)
                size_t bytes = half * sizeof(uint64_t);
                CUDA_CHECK(cudaMalloc(&d_twiddles_fwd[pi][s], bytes));
                CUDA_CHECK(cudaMemcpy(d_twiddles_fwd[pi][s], tw_fwd.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMalloc(&d_twiddles_inv[pi][s], bytes));
                CUDA_CHECK(cudaMemcpy(d_twiddles_inv[pi][s], tw_inv.data(), bytes, cudaMemcpyHostToDevice));
            }
        }
    }

    void clear() {
        for (auto& prime_tw : d_twiddles_fwd)
            for (auto* p : prime_tw) if (p) cudaFree(p);
        for (auto& prime_tw : d_twiddles_inv)
            for (auto* p : prime_tw) if (p) cudaFree(p);
        d_twiddles_fwd.clear();
        d_twiddles_inv.clear();
        ntt_size = 0;
        log_n = 0;
    }

    ~TwiddleCache() { clear(); }
};

// ============================================================
// Fast NTT using precomputed twiddles
// ============================================================

static void perform_ntt_fast(uint64_t* d_a, size_t n, int prime_idx,
                              bool inverse, const TwiddleCache& cache) {
    const int block_size = 256;
    int log_n = cache.log_n;

    // Bit-reversal
    int grid_br = (n + block_size - 1) / block_size;
    bit_reverse_kernel<<<grid_br, block_size>>>(d_a, n, log_n);

    uint64_t prime = NTT_PRIMES[prime_idx].p;
    const auto& twiddles = inverse ? cache.d_twiddles_inv[prime_idx]
                                    : cache.d_twiddles_fwd[prime_idx];

    // Butterfly layers — just kernel launches, no allocation
    for (int s = 0; s < log_n; s++) {
        size_t half = 1ULL << s;
        int grid = ((n / 2) + block_size - 1) / block_size;
        ntt_butterfly_kernel<<<grid, block_size>>>(d_a, n, half, prime, twiddles[s]);
    }

    // Scale by 1/n for inverse
    if (inverse) {
        uint64_t inv_n = host_mod_pow(n, prime - 2, prime);
        int grid = (n + block_size - 1) / block_size;
        scale_kernel<<<grid, block_size>>>(d_a, n, inv_n, prime);
    }

    cudaDeviceSynchronize();
}

// ============================================================
// CRT combination
// ============================================================

static int64_t crt_combine(uint64_t r0, uint64_t r1, uint64_t r2,
                            uint64_t p0, uint64_t p1, uint64_t p2) {
    uint64_t inv_p0_mod_p1 = host_mod_pow(p0, p1 - 2, p1);
    uint64_t inv_p0p1_mod_p2 = host_mod_pow((__uint128_t)p0 * p1 % p2, p2 - 2, p2);

    uint64_t diff1 = (r1 + p1 - r0 % p1) % p1;
    uint64_t x1 = (__uint128_t)diff1 * inv_p0_mod_p1 % p1;
    __uint128_t val = r0 + (__uint128_t)p0 * x1;

    uint64_t val_mod_p2 = val % p2;
    uint64_t diff2 = (r2 + p2 - val_mod_p2) % p2;
    uint64_t x2 = (__uint128_t)diff2 * inv_p0p1_mod_p2 % p2;
    __uint128_t result = val + (__uint128_t)p0 * p1 * x2;

    __uint128_t half_product = (__uint128_t)p0 * p1 * p2 / 2;
    if (result > half_product) {
        return -static_cast<int64_t>((__uint128_t)p0 * p1 * p2 - result);
    }
    return static_cast<int64_t>(result);
}

// ============================================================
// IntNttEngine
// ============================================================

IntNttEngine::IntNttEngine(int device_id) : device_id_(device_id) {
    twiddle_cache_ptr_ = new TwiddleCache();
}

IntNttEngine::~IntNttEngine() {
    for (int i = 0; i < NUM_PRIMES; i++) {
        if (buffers_[i].d_a) cudaFree(buffers_[i].d_a);
        if (buffers_[i].d_b) cudaFree(buffers_[i].d_b);
    }
    delete static_cast<TwiddleCache*>(twiddle_cache_ptr_);
}

void IntNttEngine::activate_device() const { cudaSetDevice(device_id_); }

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
    activate_device();

    if (ntt_size > current_ntt_size_) {
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

    // Build twiddle cache if needed (precompute all twiddle factors)
    static_cast<TwiddleCache*>(twiddle_cache_ptr_)->build(ntt_size, device_id_);
}

size_t IntNttEngine::multiply(const uint32_t* a, size_t a_len,
                               const uint32_t* b, size_t b_len,
                               uint32_t* result, size_t result_len,
                               uint64_t base) {
    size_t conv_len = a_len + b_len;
    size_t ntt_size = next_power_of_2(conv_len);

    for (int i = 0; i < NUM_PRIMES; i++) {
        if (ntt_size > (1ULL << NTT_PRIMES[i].max_log2)) {
            throw std::runtime_error("NTT size " + std::to_string(ntt_size) +
                " exceeds prime " + std::to_string(NTT_PRIMES[i].p) +
                " limit 2^" + std::to_string(NTT_PRIMES[i].max_log2));
        }
    }

    activate_device();
    ensure_buffers(ntt_size);

    std::vector<std::vector<uint64_t>> conv_results(NUM_PRIMES);

    for (int pi = 0; pi < NUM_PRIMES; pi++) {
        uint64_t prime = NTT_PRIMES[pi].p;

        std::vector<uint64_t> h_a(ntt_size, 0), h_b(ntt_size, 0);
        for (size_t i = 0; i < a_len; i++) h_a[i] = a[i] % prime;
        for (size_t i = 0; i < b_len; i++) h_b[i] = b[i] % prime;

        size_t bytes = ntt_size * sizeof(uint64_t);
        CUDA_CHECK(cudaMemcpy(buffers_[pi].d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buffers_[pi].d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

        const TwiddleCache& cache = *static_cast<TwiddleCache*>(twiddle_cache_ptr_);

        // Forward NTT — uses precomputed twiddles, no allocation
        perform_ntt_fast(buffers_[pi].d_a, ntt_size, pi, false, cache);
        perform_ntt_fast(buffers_[pi].d_b, ntt_size, pi, false, cache);

        // Pointwise multiply
        int block_size = 256;
        int grid = (ntt_size + block_size - 1) / block_size;
        pointwise_mod_mul_kernel<<<grid, block_size>>>(
            buffers_[pi].d_a, buffers_[pi].d_b, buffers_[pi].d_a, ntt_size, prime);

        // Inverse NTT
        perform_ntt_fast(buffers_[pi].d_a, ntt_size, pi, true, cache);

        conv_results[pi].resize(ntt_size);
        CUDA_CHECK(cudaMemcpy(conv_results[pi].data(), buffers_[pi].d_a, bytes,
                              cudaMemcpyDeviceToHost));
    }

    // CRT + carry propagation
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
