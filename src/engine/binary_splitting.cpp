/**
 * @file binary_splitting.cpp
 * @brief Chudnovsky binary splitting with parallel merge + checkpointing.
 */

#include "binary_splitting.h"
#include <cmath>
#include <cassert>
#include <future>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iostream>
#include <sys/stat.h>

namespace pi {

// --- BSResult ---

BSResult::BSResult() { mpz_init(P); mpz_init(Q); mpz_init(R); }
BSResult::~BSResult() { mpz_clear(P); mpz_clear(Q); mpz_clear(R); }

BSResult::BSResult(BSResult&& other) noexcept {
    mpz_init(P); mpz_init(Q); mpz_init(R);
    mpz_swap(P, other.P); mpz_swap(Q, other.Q); mpz_swap(R, other.R);
}

BSResult& BSResult::operator=(BSResult&& other) noexcept {
    if (this != &other) {
        mpz_swap(P, other.P); mpz_swap(Q, other.Q); mpz_swap(R, other.R);
    }
    return *this;
}

// --- Constants ---
static const char* C3_OVER_24 = "10939058860032000";

// --- BinarySplitting ---

BinarySplitting::BinarySplitting(Multiplier& multiplier, unsigned int num_threads)
    : multiplier_(multiplier),
      num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) {
    if (num_threads_ == 0) num_threads_ = 1;
}

unsigned long BinarySplitting::terms_needed(size_t digits) {
    return static_cast<unsigned long>(std::ceil(static_cast<double>(digits) / 14.1816474627)) + 1;
}

BSResult BinarySplitting::base_case(unsigned long a) {
    BSResult result;
    if (a == 0) {
        mpz_set_ui(result.P, 1);
        mpz_set_ui(result.Q, 1);
        mpz_set_ui(result.R, 13591409);
    } else {
        mpz_t t1, t2, t3;
        mpz_init(t1); mpz_init(t2); mpz_init(t3);
        mpz_set_ui(t1, 6 * a - 5);
        mpz_set_ui(t2, 2 * a - 1);
        mpz_set_ui(t3, 6 * a - 1);
        mpz_mul(result.P, t1, t2);
        mpz_mul(result.P, result.P, t3);
        mpz_neg(result.P, result.P);

        mpz_set_str(result.Q, C3_OVER_24, 10);
        mpz_t a_cubed;
        mpz_init(a_cubed);
        mpz_set_ui(a_cubed, a);
        mpz_pow_ui(a_cubed, a_cubed, 3);
        mpz_mul(result.Q, result.Q, a_cubed);

        mpz_t linear;
        mpz_init(linear);
        mpz_set_ui(linear, 545140134);
        mpz_mul_ui(linear, linear, a);
        mpz_add_ui(linear, linear, 13591409);
        mpz_mul(result.R, result.P, linear);

        mpz_clear(t1); mpz_clear(t2); mpz_clear(t3);
        mpz_clear(a_cubed); mpz_clear(linear);
    }
    return result;
}

BSResult BinarySplitting::merge_parallel(BSResult& left, BSResult& right) {
    BSResult result;

    // Parallel merge: run 4 multiplications concurrently
    // temp1 = Q_right * R_left
    // temp2 = P_left * R_right
    // P_new = P_left * P_right
    // Q_new = Q_left * Q_right
    // Then: R_new = temp1 + temp2

    mpz_t temp1, temp2;
    mpz_init(temp1);
    mpz_init(temp2);

    // Check if the numbers are large enough to benefit from parallel merge
    size_t max_size = std::max({mpz_size(left.P), mpz_size(left.Q),
                                 mpz_size(right.P), mpz_size(right.Q)});

    if (max_size > 1000 && num_threads_ > 1) {
        // Large numbers: parallelize the 4 multiplications
        // Pre-allocate all output variables (mpz_t can't be returned from lambdas)
        mpz_t t_qr, t_pr, t_pp;
        mpz_init(t_qr); mpz_init(t_pr); mpz_init(t_pp);

        auto f1 = std::async(std::launch::async, [&]() {
            multiplier_.multiply(t_qr, right.Q, left.R);
        });
        auto f2 = std::async(std::launch::async, [&]() {
            multiplier_.multiply(t_pr, left.P, right.R);
        });
        auto f3 = std::async(std::launch::async, [&]() {
            multiplier_.multiply(t_pp, left.P, right.P);
        });

        // Q*Q in current thread
        multiplier_.multiply(result.Q, left.Q, right.Q);

        f1.get(); f2.get(); f3.get();

        mpz_add(result.R, t_qr, t_pr);
        mpz_swap(result.P, t_pp);

        mpz_clear(t_qr); mpz_clear(t_pr); mpz_clear(t_pp);
    } else {
        // Small numbers: sequential (thread overhead not worth it)
        multiplier_.multiply(temp1, right.Q, left.R);
        multiplier_.multiply(temp2, left.P, right.R);
        mpz_add(result.R, temp1, temp2);
        multiplier_.multiply(result.P, left.P, right.P);
        multiplier_.multiply(result.Q, left.Q, right.Q);
    }

    mpz_clear(temp1);
    mpz_clear(temp2);

    // Memory efficiency: clear the inputs early (they're no longer needed)
    // This is safe because the caller's BSResult will be destroyed anyway
    mpz_set_ui(left.P, 0); mpz_set_ui(left.Q, 0); mpz_set_ui(left.R, 0);
    mpz_set_ui(right.P, 0); mpz_set_ui(right.Q, 0); mpz_set_ui(right.R, 0);

    return result;
}

BSResult BinarySplitting::compute_sequential(unsigned long a, unsigned long b) {
    assert(b > a);
    if (b - a == 1) return base_case(a);

    unsigned long m = a + (b - a) / 2;
    BSResult left = compute_sequential(a, m);
    BSResult right = compute_sequential(m, b);
    return merge_parallel(left, right);
}

BSResult BinarySplitting::compute_parallel(unsigned long a, unsigned long b, int depth) {
    assert(b > a);
    if (b - a == 1) return base_case(a);

    unsigned long m = a + (b - a) / 2;
    bool should_parallelize = (depth > 0) && (b - a >= PARALLEL_THRESHOLD);

    if (should_parallelize) {
        std::future<BSResult> right_future = std::async(
            std::launch::async,
            [this, m, b, depth]() { return compute_parallel(m, b, depth - 1); }
        );
        BSResult left = compute_parallel(a, m, depth - 1);
        BSResult right = right_future.get();
        return merge_parallel(left, right);
    } else {
        return compute_sequential(a, b);
    }
}

BSResult BinarySplitting::compute(unsigned long a, unsigned long b) {
    assert(b > a);

    // Try to resume from checkpoint
    if (checkpointing_enabled_) {
        BSResult resumed;
        if (try_resume(a, b, resumed)) {
            return resumed;
        }
    }

    BSResult result;
    if (num_threads_ <= 1) {
        result = compute_sequential(a, b);
    } else {
        int max_depth = 0;
        unsigned int t = num_threads_;
        while (t > 1) { max_depth++; t >>= 1; }
        result = compute_parallel(a, b, max_depth);
    }

    // Save checkpoint
    if (checkpointing_enabled_) {
        save_checkpoint(a, b, result);
    }

    return result;
}

// --- Checkpointing ---

void BinarySplitting::enable_checkpointing(const std::string& dir, int interval_seconds) {
    checkpointing_enabled_ = true;
    checkpoint_dir_ = dir;
    checkpoint_interval_ = interval_seconds;

    // Create directory if it doesn't exist
    mkdir(dir.c_str(), 0755);
}

void BinarySplitting::save_checkpoint(unsigned long a, unsigned long b, const BSResult& result) {
    std::string path = checkpoint_dir_ + "/bs_" + std::to_string(a) + "_" + std::to_string(b) + ".ckpt";

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;

    // Write P, Q, R as raw GMP export
    auto write_mpz = [&](const mpz_t val) {
        size_t count = 0;
        void* data = mpz_export(nullptr, &count, -1, 1, -1, 0, val);
        int sign = mpz_sgn(val);
        fwrite(&sign, sizeof(int), 1, f);
        fwrite(&count, sizeof(size_t), 1, f);
        if (count > 0 && data) {
            fwrite(data, 1, count, f);
            free(data);
        }
    };

    write_mpz(result.P);
    write_mpz(result.Q);
    write_mpz(result.R);

    fclose(f);
}

bool BinarySplitting::try_resume(unsigned long a, unsigned long b, BSResult& result) {
    std::string path = checkpoint_dir_ + "/bs_" + std::to_string(a) + "_" + std::to_string(b) + ".ckpt";

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    auto read_mpz = [&](mpz_t val) -> bool {
        int sign;
        size_t count;
        if (fread(&sign, sizeof(int), 1, f) != 1) return false;
        if (fread(&count, sizeof(size_t), 1, f) != 1) return false;
        if (count > 0) {
            std::vector<uint8_t> data(count);
            if (fread(data.data(), 1, count, f) != count) return false;
            mpz_import(val, count, -1, 1, -1, 0, data.data());
            if (sign < 0) mpz_neg(val, val);
        } else {
            mpz_set_ui(val, 0);
        }
        return true;
    };

    bool ok = read_mpz(result.P) && read_mpz(result.Q) && read_mpz(result.R);
    fclose(f);

    if (ok) {
        std::cout << "  Resumed from checkpoint: [" << a << ", " << b << ")" << std::endl;
    }

    return ok;
}

} // namespace pi
