/**
 * @file binary_splitting.cpp
 * @brief Chudnovsky binary splitting with parallel merge + dynamic checkpointing.
 *
 * Dynamic checkpointing: saves intermediate results every N minutes.
 * On resume, loads the most recent checkpoint for each sub-range and
 * only recomputes what's missing.
 */

#include "binary_splitting.h"
#include "pi_engine.h"
#include <cmath>
#include <cassert>
#include <vector>
#include <future>
#include <thread>
#include <mutex>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iostream>
#include <atomic>
#include <stdexcept>
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
    if (!f) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "  WARNING: Cannot write checkpoint: " << path
                      << " (permission denied or disk full)" << std::endl;
            warned = true;
        }
        return false;
    }
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

    // Delete child checkpoints — they're redundant now that the parent is saved.
    // This prevents checkpoint files from accumulating and filling the disk.
    unsigned long m = a + (b - a) / 2;
    if (b - a > 1) {
        std::string left_path = ckpt_path(checkpoint_dir_, a, m);
        std::string right_path = ckpt_path(checkpoint_dir_, m, b);
        remove(left_path.c_str());
        remove(right_path.c_str());
    }
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

    size_t max_size = std::max({mpz_size(left.P), mpz_size(left.Q),
                                 mpz_size(right.P), mpz_size(right.Q)});

    // Three-tier merge strategy based on operand size:
    //
    // Tier 1 (small, <1K limbs): Sequential — no parallelism overhead
    // Tier 2 (medium, 1K–50M limbs): Full parallel — 4 concurrent muls, 4 cores
    // Tier 3 (large, >50M limbs): Semi-parallel — 2 concurrent muls at a time,
    //         2 cores, with aggressive early freeing between pairs
    //
    // Tier 3 gives 2x throughput over sequential while keeping memory bounded:
    // - Only 2 multiplications alive at once (vs 4 in tier 2)
    // - Inputs freed between pairs, so peak is ~2x one multiplication
    static constexpr size_t PARALLEL_MERGE_MAX_LIMBS = 50000000;

    if (max_size > 1000 && max_size < PARALLEL_MERGE_MAX_LIMBS && num_threads_ > 1) {
        // TIER 2: Full parallel merge — 4 multiplications run concurrently.
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

        // Free inputs immediately
        mpz_realloc2(left.P, 0); mpz_realloc2(left.Q, 0); mpz_realloc2(left.R, 0);
        mpz_realloc2(right.P, 0); mpz_realloc2(right.Q, 0); mpz_realloc2(right.R, 0);

        mpz_add(result.R, t_qr, t_pr);
        mpz_swap(result.P, t_pp);
        mpz_clear(t_qr); mpz_clear(t_pr); mpz_clear(t_pp);

    } else if (max_size >= PARALLEL_MERGE_MAX_LIMBS && num_threads_ > 1) {
        // TIER 3: Semi-parallel merge — 2 concurrent muls at a time.
        // Uses 2 cores while keeping memory bounded.
        //
        // Pair 1: temp1 = Q_right * R_left  ||  result.R = P_left * R_right
        // Then free R_left, R_right
        // Pair 2: result.P = P_left * P_right  ||  result.Q = Q_left * Q_right
        // Then free all remaining inputs

        mpz_t temp1;
        mpz_init(temp1);

        // Pair 1: both R computations in parallel
        {
            auto f1 = std::async(std::launch::async, [&]() {
                multiplier_.multiply(temp1, right.Q, left.R);
            });
            multiplier_.multiply(result.R, left.P, right.R);
            f1.get();
        }
        // Free R_left and R_right (both fully consumed)
        mpz_realloc2(left.R, 0);
        mpz_realloc2(right.R, 0);

        // Accumulate R
        mpz_add(result.R, result.R, temp1);
        mpz_clear(temp1);

        // Pair 2: P*P and Q*Q in parallel
        {
            auto f2 = std::async(std::launch::async, [&]() {
                multiplier_.multiply(result.Q, left.Q, right.Q);
            });
            multiplier_.multiply(result.P, left.P, right.P);
            f2.get();
        }
        // Free all remaining inputs
        mpz_realloc2(left.P, 0); mpz_realloc2(left.Q, 0);
        mpz_realloc2(right.P, 0); mpz_realloc2(right.Q, 0);

    } else {
        // TIER 1: Sequential merge (single-threaded or tiny operands).
        mpz_t temp1;
        mpz_init(temp1);

        multiplier_.multiply(temp1, right.Q, left.R);
        mpz_realloc2(left.R, 0);

        multiplier_.multiply(result.R, left.P, right.R);
        mpz_realloc2(right.R, 0);

        mpz_add(result.R, result.R, temp1);
        mpz_clear(temp1);

        multiplier_.multiply(result.P, left.P, right.P);
        mpz_realloc2(left.P, 0);
        mpz_realloc2(right.P, 0);

        multiplier_.multiply(result.Q, left.Q, right.Q);
        mpz_realloc2(left.Q, 0);
        mpz_realloc2(right.Q, 0);
    }

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

    // For ranges small enough that memory isn't a concern, use parallel
    // tree traversal to keep all cores busy. The threshold is set so that
    // the P/Q/R values at this level are small enough for concurrent branches.
    // ~10K terms → numbers with ~100K digits → ~12K limbs → safe to parallelize.
    static constexpr unsigned long PARALLEL_SUBTREE_THRESHOLD = 10000;

    BSResult left, right;
    if (num_threads_ > 1 && (b - a) <= PARALLEL_SUBTREE_THRESHOLD && (b - a) >= PARALLEL_THRESHOLD) {
        // Parallel subtree: fork left/right, both branches are small
        int sub_depth = 0;
        unsigned int t = num_threads_;
        while (t > 1) { sub_depth++; t >>= 1; }
        auto right_future = std::async(std::launch::async,
            [this, m, b, sub_depth]() { return compute_parallel(m, b, sub_depth); }
        );
        left = compute_parallel(a, m, sub_depth);
        right = right_future.get();
    } else {
        left = compute_sequential(a, m);
        right = compute_sequential(m, b);
    }

    BSResult result = merge_parallel(left, right);

    // Save checkpoint if range is large enough to be worth the I/O cost.
    // For 50B digits (~3.5B terms), ranges <100K terms are cheap to recompute
    // but their checkpoint files accumulate and can fill the disk.
    if (checkpointing_enabled_ && (b - a) >= 100000) {
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
    if (checkpointing_enabled_ && (b - a) >= 100000) {
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

void BinarySplitting::enable_out_of_core(unsigned int num_chunks) {
    out_of_core_enabled_ = true;
    ooc_num_chunks_ = num_chunks;
}

BSResult BinarySplitting::compute_out_of_core(unsigned long a, unsigned long b) {
    assert(b > a);
    assert(checkpointing_enabled_);  // OOC requires checkpoint dir for temp files

    unsigned long range = b - a;

    // Determine number of chunks: each chunk should be small enough to compute
    // quickly in RAM, but large enough to amortize disk I/O.
    // Target: ~50K-500K terms per chunk → ~700K-7M digits → ~300MB-3GB RAM each.
    unsigned int num_chunks = ooc_num_chunks_;
    if (num_chunks == 0) {
        // Auto: aim for ~100K terms per chunk, rounded up to power of 2
        num_chunks = static_cast<unsigned int>(range / 100000);
        if (num_chunks < 2) num_chunks = 2;
        // Round up to next power of 2 (required for balanced merge tree)
        unsigned int p = 1;
        while (p < num_chunks) p <<= 1;
        num_chunks = p;
        // Cap at reasonable maximum
        if (num_chunks > 4096) num_chunks = 4096;
    }

    // Ensure power of 2
    {
        unsigned int p = 1;
        while (p < num_chunks) p <<= 1;
        num_chunks = p;
    }

    unsigned long chunk_size = range / num_chunks;
    if (chunk_size == 0) {
        // Range too small for OOC, fall back to sequential
        return compute_sequential(a, b);
    }

    std::cout << "  Out-of-core: " << num_chunks << " chunks of ~"
              << chunk_size << " terms each" << std::endl;

    // ================================================================
    // Phase 1: COMPUTE — parallel computation of all chunks
    // ================================================================
    std::cout << "  Phase 1: Computing " << num_chunks << " chunks in parallel..." << std::endl;
    auto phase1_start = std::chrono::steady_clock::now();

    // Build chunk boundaries
    std::vector<unsigned long> boundaries;
    for (unsigned int i = 0; i <= num_chunks; ++i) {
        boundaries.push_back(a + (range * i) / num_chunks);
    }

    // Check which chunks already have results on disk
    std::vector<bool> chunk_done(num_chunks, false);
    unsigned int chunks_cached = 0;
    for (unsigned int i = 0; i < num_chunks; ++i) {
        std::string path = ckpt_path(checkpoint_dir_, boundaries[i], boundaries[i + 1]);
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fclose(f);
            chunk_done[i] = true;
            chunks_cached++;
        }
    }
    if (chunks_cached > 0) {
        std::cout << "    " << chunks_cached << "/" << num_chunks
                  << " chunks already on disk (resuming)" << std::endl;
    }

    // Compute remaining chunks using a thread pool
    std::atomic<unsigned int> next_chunk{0};
    std::atomic<unsigned int> completed{chunks_cached};
    std::mutex cout_mutex;

    auto worker = [&]() {
        while (true) {
            unsigned int idx = next_chunk.fetch_add(1);
            if (idx >= num_chunks) break;
            if (chunk_done[idx]) continue;

            unsigned long ca = boundaries[idx];
            unsigned long cb = boundaries[idx + 1];

            // Compute this chunk sequentially (parallelism comes from running
            // multiple workers concurrently, not from within each chunk).
            // compute_sequential still uses merge_parallel for each merge,
            // giving 2-4 cores per merge via the tier system.
            BSResult chunk_result = compute_sequential(ca, cb);

            // Save to disk and free RAM
            save_bs_result(ckpt_path(checkpoint_dir_, ca, cb), chunk_result);

            unsigned int done = completed.fetch_add(1) + 1;
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "    Chunk " << done << "/" << num_chunks
                          << " done: [" << ca << ", " << cb << ")"
                          << " | RSS: " << pi::PiEngine::get_rss_mb() << " MB"
                          << std::endl;
            }
        }
    };

    // Launch worker threads — but limit concurrency to avoid OOM.
    // Each chunk uses ~10 * (chunk_digits) * 0.415 bytes of RAM.
    // With chunk_size terms, chunk_digits ≈ chunk_size * 14.18.
    // RAM per chunk ≈ chunk_size * 14.18 * 10 * 0.415 ≈ chunk_size * 59 bytes.
    // For 100K terms: ~6 MB per chunk → can run many concurrently.
    // For 1M terms: ~59 MB per chunk → still fine.
    // Limit to num_threads_ concurrent workers to avoid thread explosion.
    unsigned int num_workers = std::min(num_threads_, num_chunks);
    std::vector<std::thread> workers;
    for (unsigned int i = 0; i < num_workers; ++i) {
        workers.emplace_back(worker);
    }
    for (auto& w : workers) {
        w.join();
    }

    auto phase1_end = std::chrono::steady_clock::now();
    double phase1_time = std::chrono::duration<double>(phase1_end - phase1_start).count();
    std::cout << "  Phase 1 complete: " << phase1_time << "s"
              << " | RSS: " << pi::PiEngine::get_rss_mb() << " MB" << std::endl;

    // ================================================================
    // Phase 2: MERGE — bottom-up cascade, loading pairs from disk
    // ================================================================
    std::cout << "  Phase 2: Merging " << num_chunks << " chunks bottom-up..." << std::endl;
    auto phase2_start = std::chrono::steady_clock::now();

    // Current level's chunk boundaries
    std::vector<unsigned long> current_bounds = boundaries;
    unsigned int current_count = num_chunks;

    while (current_count > 1) {
        unsigned int next_count = current_count / 2;
        std::cout << "    Merge level: " << current_count << " → " << next_count
                  << " chunks" << std::endl;

        // Merge pairs: [0,1], [2,3], [4,5], ...
        // For small enough merges, parallelize across pairs
        for (unsigned int i = 0; i < current_count; i += 2) {
            unsigned long la = current_bounds[i];
            unsigned long lm = current_bounds[i + 1];
            unsigned long rb = current_bounds[i + 2];

            std::string left_path = ckpt_path(checkpoint_dir_, la, lm);
            std::string right_path = ckpt_path(checkpoint_dir_, lm, rb);
            std::string merged_path = ckpt_path(checkpoint_dir_, la, rb);

            // Check if merged result already exists (resume support)
            FILE* mf = fopen(merged_path.c_str(), "rb");
            if (mf) {
                fclose(mf);
                // Delete children
                remove(left_path.c_str());
                remove(right_path.c_str());
                continue;
            }

            // Load left and right from disk
            BSResult left, right;
            if (!load_bs_result(left_path, left)) {
                throw std::runtime_error("Failed to load chunk: " + left_path);
            }
            if (!load_bs_result(right_path, right)) {
                throw std::runtime_error("Failed to load chunk: " + right_path);
            }

            // Merge
            BSResult merged = merge_parallel(left, right);

            // Save merged result, delete children
            save_bs_result(merged_path, merged);
            remove(left_path.c_str());
            remove(right_path.c_str());

            std::cout << "      Merged [" << la << ", " << rb << ")"
                      << " | RSS: " << pi::PiEngine::get_rss_mb() << " MB"
                      << std::endl;
        }

        // Build next level boundaries
        std::vector<unsigned long> next_bounds;
        for (unsigned int i = 0; i <= current_count; i += 2) {
            next_bounds.push_back(current_bounds[i]);
        }
        current_bounds = next_bounds;
        current_count = next_count;
    }

    auto phase2_end = std::chrono::steady_clock::now();
    double phase2_time = std::chrono::duration<double>(phase2_end - phase2_start).count();
    std::cout << "  Phase 2 complete: " << phase2_time << "s"
              << " | RSS: " << pi::PiEngine::get_rss_mb() << " MB" << std::endl;

    // Load final result from disk
    std::string final_path = ckpt_path(checkpoint_dir_, a, b);
    BSResult result;
    if (!load_bs_result(final_path, result)) {
        throw std::runtime_error("Failed to load final result: " + final_path);
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

    // Out-of-core mode: compute wide, merge narrow
    if (out_of_core_enabled_ && checkpointing_enabled_ && num_threads_ > 1) {
        return compute_out_of_core(a, b);
    }

    BSResult result;
    if (num_threads_ <= 1) {
        result = compute_sequential(a, b);
    } else {
        int max_depth = 0;
        unsigned int t = num_threads_;
        while (t > 1) { max_depth++; t >>= 1; }

        // Cap parallel depth for large computations to limit peak memory.
        unsigned long range = b - a;
        if (range > 100000000 && max_depth > 1) {
            max_depth = 1;
        } else if (range > 50000000 && max_depth > 2) {
            max_depth = 2;
        } else if (range > 10000000 && max_depth > 3) {
            max_depth = 3;
        }

        result = compute_parallel(a, b, max_depth);
    }

    // Always save final checkpoint
    save_checkpoint(a, b, result);

    return result;
}

} // namespace pi
