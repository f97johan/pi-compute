/**
 * @file binary_splitting.cpp
 * @brief Chudnovsky binary splitting with parallel merge + dynamic checkpointing.
 *
 * Dynamic checkpointing: saves intermediate results every N minutes.
 * On resume, loads the most recent checkpoint for each sub-range and
 * only recomputes what's missing.
 */

#include "binary_splitting.h"
#include <cmath>
#include <cassert>
#include <future>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iostream>
#include <atomic>
#include <sys/stat.h>

namespace pi {

// --- BSResult ---
BSResult::BSResult() { mpz_init(P); mpz_init(Q); mpz_init(R); }
BSResult::~BSResult() { mpz_clear(P); mpz_clear(Q); mpz_clear(R); }
BSResult::BSResult(BSResult&& o) noexcept {
    mpz_init(P); mpz_init(Q); mpz_init(R);
    mpz_swap(P, o.P); mpz_swap(Q, o.Q); mpz_swap(R, o.R);
}
BSResult& BSResult::operator=(BSResult&& o) noexcept {
    if (this != &o) { mpz_swap(P, o.P); mpz_swap(Q, o.Q); mpz_swap(R, o.R); }
    return *this;
}

static const char* C3_OVER_24 = "10939058860032000";

// --- Checkpoint I/O ---

static std::string ckpt_path(const std::string& dir, unsigned long a, unsigned long b) {
    return dir + "/bs_" + std::to_string(a) + "_" + std::to_string(b) + ".ckpt";
}

static bool save_bs_result(const std::string& path, const BSResult& r) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    auto write_mpz = [&](const mpz_t val) {
        int sign = mpz_sgn(val);
        size_t count = 0;
        void* data = mpz_export(nullptr, &count, -1, 1, -1, 0, val);
        fwrite(&sign, sizeof(int), 1, f);
        fwrite(&count, sizeof(size_t), 1, f);
        if (count > 0 && data) { fwrite(data, 1, count, f); free(data); }
    };
    write_mpz(r.P); write_mpz(r.Q); write_mpz(r.R);
    fclose(f);
    return true;
}

static bool load_bs_result(const std::string& path, BSResult& r) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    auto read_mpz = [&](mpz_t val) -> bool {
        int sign; size_t count;
        if (fread(&sign, sizeof(int), 1, f) != 1) return false;
        if (fread(&count, sizeof(size_t), 1, f) != 1) return false;
        if (count > 0) {
            std::vector<uint8_t> data(count);
            if (fread(data.data(), 1, count, f) != count) return false;
            mpz_import(val, count, -1, 1, -1, 0, data.data());
            if (sign < 0) mpz_neg(val, val);
        } else { mpz_set_ui(val, 0); }
        return true;
    };
    bool ok = read_mpz(r.P) && read_mpz(r.Q) && read_mpz(r.R);
    fclose(f);
    return ok;
}

// --- BinarySplitting ---

BinarySplitting::BinarySplitting(Multiplier& multiplier, unsigned int num_threads)
    : multiplier_(multiplier),
      num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) {
    if (num_threads_ == 0) num_threads_ = 1;
}

unsigned long BinarySplitting::terms_needed(size_t digits) {
    return static_cast<unsigned long>(std::ceil(static_cast<double>(digits) / 14.1816474627)) + 1;
}

void BinarySplitting::enable_checkpointing(const std::string& dir, int interval_seconds) {
    checkpointing_enabled_ = true;
    checkpoint_dir_ = dir;
    checkpoint_interval_ = interval_seconds;
    mkdir(dir.c_str(), 0755);
}

bool BinarySplitting::try_resume(unsigned long a, unsigned long b, BSResult& result) {
    if (!checkpointing_enabled_) return false;
    std::string path = ckpt_path(checkpoint_dir_, a, b);
    if (load_bs_result(path, result)) {
        std::cout << "  Resumed from checkpoint: [" << a << ", " << b << ")" << std::endl;
        return true;
    }
    return false;
}

void BinarySplitting::save_checkpoint(unsigned long a, unsigned long b, const BSResult& result) {
    if (!checkpointing_enabled_) return;
    std::string path = ckpt_path(checkpoint_dir_, a, b);
    save_bs_result(path, result);
}

BSResult BinarySplitting::base_case(unsigned long a) {
    BSResult result;
    if (a == 0) {
        mpz_set_ui(result.P, 1); mpz_set_ui(result.Q, 1); mpz_set_ui(result.R, 13591409);
    } else {
        mpz_t t1, t2, t3;
        mpz_init(t1); mpz_init(t2); mpz_init(t3);
        mpz_set_ui(t1, 6*a-5); mpz_set_ui(t2, 2*a-1); mpz_set_ui(t3, 6*a-1);
        mpz_mul(result.P, t1, t2); mpz_mul(result.P, result.P, t3);
        mpz_neg(result.P, result.P);

        mpz_set_str(result.Q, C3_OVER_24, 10);
        mpz_t ac; mpz_init(ac); mpz_set_ui(ac, a); mpz_pow_ui(ac, ac, 3);
        mpz_mul(result.Q, result.Q, ac);

        mpz_t lin; mpz_init(lin);
        mpz_set_ui(lin, 545140134); mpz_mul_ui(lin, lin, a); mpz_add_ui(lin, lin, 13591409);
        mpz_mul(result.R, result.P, lin);

        mpz_clear(t1); mpz_clear(t2); mpz_clear(t3); mpz_clear(ac); mpz_clear(lin);
    }
    return result;
}

BSResult BinarySplitting::merge_parallel(BSResult& left, BSResult& right) {
    BSResult result;
    mpz_t temp1, temp2;
    mpz_init(temp1); mpz_init(temp2);

    size_t max_size = std::max({mpz_size(left.P), mpz_size(left.Q),
                                 mpz_size(right.P), mpz_size(right.Q)});

    if (max_size > 1000 && num_threads_ > 1) {
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
        multiplier_.multiply(result.Q, left.Q, right.Q);
        f1.get(); f2.get(); f3.get();

        mpz_add(result.R, t_qr, t_pr);
        mpz_swap(result.P, t_pp);
        mpz_clear(t_qr); mpz_clear(t_pr); mpz_clear(t_pp);
    } else {
        multiplier_.multiply(temp1, right.Q, left.R);
        multiplier_.multiply(temp2, left.P, right.R);
        mpz_add(result.R, temp1, temp2);
        multiplier_.multiply(result.P, left.P, right.P);
        multiplier_.multiply(result.Q, left.Q, right.Q);
    }

    mpz_clear(temp1); mpz_clear(temp2);
    mpz_set_ui(left.P, 0); mpz_set_ui(left.Q, 0); mpz_set_ui(left.R, 0);
    mpz_set_ui(right.P, 0); mpz_set_ui(right.Q, 0); mpz_set_ui(right.R, 0);
    return result;
}

BSResult BinarySplitting::compute_sequential(unsigned long a, unsigned long b) {
    assert(b > a);
    if (b - a == 1) return base_case(a);

    // Check for checkpoint
    if (checkpointing_enabled_) {
        BSResult cached;
        if (load_bs_result(ckpt_path(checkpoint_dir_, a, b), cached)) return cached;
    }

    unsigned long m = a + (b - a) / 2;
    BSResult left = compute_sequential(a, m);
    BSResult right = compute_sequential(m, b);
    BSResult result = merge_parallel(left, right);

    // Save checkpoint if range is large enough (avoid tiny checkpoints)
    if (checkpointing_enabled_ && (b - a) >= 1000) {
        static std::atomic<int64_t> last_ckpt_time{0};
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        int64_t now_sec = std::chrono::duration_cast<std::chrono::seconds>(now).count();
        int64_t last = last_ckpt_time.load();
        if (now_sec - last >= checkpoint_interval_) {
            if (last_ckpt_time.compare_exchange_strong(last, now_sec)) {
                save_checkpoint(a, b, result);
                std::cout << "  Checkpoint saved: [" << a << ", " << b << ")" << std::endl;
            }
        }
    }

    return result;
}

BSResult BinarySplitting::compute_parallel(unsigned long a, unsigned long b, int depth) {
    assert(b > a);
    if (b - a == 1) return base_case(a);

    // Check for checkpoint
    if (checkpointing_enabled_) {
        BSResult cached;
        if (load_bs_result(ckpt_path(checkpoint_dir_, a, b), cached)) {
            std::cout << "  Loaded checkpoint: [" << a << ", " << b << ")" << std::endl;
            return cached;
        }
    }

    unsigned long m = a + (b - a) / 2;
    bool should_parallelize = (depth > 0) && (b - a >= PARALLEL_THRESHOLD);

    BSResult result;
    if (should_parallelize) {
        std::future<BSResult> right_future = std::async(
            std::launch::async,
            [this, m, b, depth]() { return compute_parallel(m, b, depth - 1); }
        );
        BSResult left = compute_parallel(a, m, depth - 1);
        BSResult right = right_future.get();
        result = merge_parallel(left, right);
    } else {
        result = compute_sequential(a, b);
    }

    // Save checkpoint for large ranges, time-based
    if (checkpointing_enabled_ && (b - a) >= 1000) {
        static std::atomic<int64_t> last_ckpt_time_p{0};
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        int64_t now_sec = std::chrono::duration_cast<std::chrono::seconds>(now).count();
        int64_t last = last_ckpt_time_p.load();
        if (now_sec - last >= checkpoint_interval_) {
            if (last_ckpt_time_p.compare_exchange_strong(last, now_sec)) {
                save_checkpoint(a, b, result);
                std::cout << "  Checkpoint saved: [" << a << ", " << b << ")" << std::endl;
            }
        }
    }

    return result;
}

BSResult BinarySplitting::compute(unsigned long a, unsigned long b) {
    assert(b > a);

    // Try full-range checkpoint first
    if (checkpointing_enabled_) {
        BSResult resumed;
        if (try_resume(a, b, resumed)) return resumed;
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

    // Always save final checkpoint
    save_checkpoint(a, b, result);

    return result;
}

} // namespace pi
