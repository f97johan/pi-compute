/**
 * @file test_gpu_ntt_multiplier.cpp
 * @brief Tests for GPU NTT multiplication — only compiled with CUDA.
 *
 * These tests verify that the GPU multiply path produces identical
 * results to GMP's CPU multiply for various operand sizes.
 */

#ifdef PI_CUDA_ENABLED

#include <gtest/gtest.h>
#include <gmp.h>
#include <string>
#include "arithmetic/gpu_ntt_multiplier.h"
#include "arithmetic/gmp_multiplier.h"
#include "engine/pi_engine.h"

class GpuNttMultiplierTest : public ::testing::Test {
protected:
    pi::GpuNttMultiplier gpu_mult{1};  // threshold=1 to force GPU path
    pi::GmpMultiplier cpu_mult;
    mpz_t a, b, gpu_result, cpu_result;

    void SetUp() override {
        mpz_init(a);
        mpz_init(b);
        mpz_init(gpu_result);
        mpz_init(cpu_result);
    }

    void TearDown() override {
        mpz_clear(a);
        mpz_clear(b);
        mpz_clear(gpu_result);
        mpz_clear(cpu_result);
    }

    void assert_gpu_matches_cpu(const char* test_name) {
        gpu_mult.multiply(gpu_result, a, b);
        cpu_mult.multiply(cpu_result, a, b);
        EXPECT_EQ(mpz_cmp(gpu_result, cpu_result), 0)
            << test_name << ": GPU and CPU results differ"
            << "\n  a size: " << mpz_sizeinbase(a, 10) << " digits"
            << "\n  b size: " << mpz_sizeinbase(b, 10) << " digits";
    }
};

TEST_F(GpuNttMultiplierTest, IsAvailable) {
    EXPECT_TRUE(pi::gpu::NttEngine::is_available())
        << "No CUDA GPU available — GPU tests require NVIDIA GPU";
}

TEST_F(GpuNttMultiplierTest, DeviceName) {
    std::string name = gpu_mult.device_name();
    EXPECT_FALSE(name.empty());
    EXPECT_NE(name, "unknown");
    std::cout << "GPU: " << name << std::endl;
}

TEST_F(GpuNttMultiplierTest, MultiplySmallNumbers) {
    mpz_set_ui(a, 12345);
    mpz_set_ui(b, 67890);
    assert_gpu_matches_cpu("SmallNumbers");
}

TEST_F(GpuNttMultiplierTest, MultiplyByZero) {
    mpz_set_ui(a, 999999);
    mpz_set_ui(b, 0);
    assert_gpu_matches_cpu("MultiplyByZero");
}

TEST_F(GpuNttMultiplierTest, MultiplyByOne) {
    mpz_set_str(a, "123456789012345678901234567890", 10);
    mpz_set_ui(b, 1);
    assert_gpu_matches_cpu("MultiplyByOne");
}

TEST_F(GpuNttMultiplierTest, MultiplyNegative) {
    mpz_set_si(a, -12345);
    mpz_set_si(b, 67890);
    assert_gpu_matches_cpu("MultiplyNegative");
}

TEST_F(GpuNttMultiplierTest, Multiply1000DigitNumbers) {
    // Two ~1000-digit numbers
    mpz_ui_pow_ui(a, 2, 3321);  // ~1000 decimal digits
    mpz_sub_ui(a, a, 1);
    mpz_ui_pow_ui(b, 3, 2095);  // ~1000 decimal digits
    mpz_sub_ui(b, b, 1);
    assert_gpu_matches_cpu("1000DigitNumbers");
}

TEST_F(GpuNttMultiplierTest, Multiply10000DigitNumbers) {
    // Two ~10,000-digit numbers
    mpz_ui_pow_ui(a, 2, 33219);  // ~10,000 decimal digits
    mpz_sub_ui(a, a, 1);
    mpz_ui_pow_ui(b, 3, 20959);  // ~10,000 decimal digits
    mpz_sub_ui(b, b, 1);
    assert_gpu_matches_cpu("10000DigitNumbers");
}

TEST_F(GpuNttMultiplierTest, Multiply100000DigitNumbers) {
    // Two ~100,000-digit numbers
    mpz_ui_pow_ui(a, 2, 332192);  // ~100,000 decimal digits
    mpz_sub_ui(a, a, 1);
    mpz_ui_pow_ui(b, 3, 209590);  // ~100,000 decimal digits
    mpz_sub_ui(b, b, 1);
    assert_gpu_matches_cpu("100000DigitNumbers");
}

TEST_F(GpuNttMultiplierTest, SquareMatchesCpu) {
    mpz_ui_pow_ui(a, 2, 33219);  // ~10,000 decimal digits
    mpz_sub_ui(a, a, 1);

    gpu_mult.square(gpu_result, a);
    cpu_mult.square(cpu_result, a);

    EXPECT_EQ(mpz_cmp(gpu_result, cpu_result), 0)
        << "GPU square and CPU square differ";
}

TEST_F(GpuNttMultiplierTest, ThresholdFallback) {
    // With high threshold, should fall back to CPU (GMP) internally
    pi::GpuNttMultiplier high_threshold_mult(999999);
    mpz_set_ui(a, 12345);
    mpz_set_ui(b, 67890);

    mpz_t result;
    mpz_init(result);
    high_threshold_mult.multiply(result, a, b);

    mpz_t expected;
    mpz_init(expected);
    mpz_mul(expected, a, b);

    EXPECT_EQ(mpz_cmp(result, expected), 0);

    mpz_clear(result);
    mpz_clear(expected);
}

// Integration test: compute pi with GPU multiplier
TEST_F(GpuNttMultiplierTest, ComputePi1000DigitsWithGpu) {
    pi::GpuNttMultiplier mult(100);  // Low threshold to exercise GPU
    pi::PiEngine engine(mult);
    std::string result = engine.compute_digits(1000);

    // Verify first 50 digits
    std::string expected_start = "3.14159265358979323846264338327950288419716939937510";
    EXPECT_EQ(result.substr(0, expected_start.size()), expected_start)
        << "GPU pi computation doesn't match expected digits";
}

TEST_F(GpuNttMultiplierTest, ComputePi10000DigitsWithGpu) {
    pi::GpuNttMultiplier mult(100);
    pi::PiEngine engine(mult);
    std::string result = engine.compute_digits(10000);

    EXPECT_EQ(result.substr(0, 7), "3.14159");
    EXPECT_GE(result.size(), 10001u);
}

#endif // PI_CUDA_ENABLED
