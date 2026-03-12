#pragma once

/**
 * @file binary_splitting.h
 * @brief Chudnovsky algorithm with binary splitting.
 *
 * Optimizations:
 * 1. Multi-threaded: left/right halves computed in parallel
 * 2. Parallel merge: 4 of 5 multiplications run concurrently
 * 3. Memory-efficient: depth-first traversal, frees intermediates early
 * 4. Checkpointing: saves/restores state for crash recovery
 */

#include <gmp.h>
#include <cstddef>
#include <string>
#include <thread>
#include "../arithmetic/multiplier.h"

namespace pi {

struct BSResult {
    mpz_t P, Q, R;

    BSResult();
    ~BSResult();
    BSResult(const BSResult&) = delete;
    BSResult& operator=(const BSResult&) = delete;
    BSResult(BSResult&& other) noexcept;
    BSResult& operator=(BSResult&& other) noexcept;
};

class BinarySplitting {
public:
    /**
     * @param multiplier Multiplication backend
     * @param num_threads Thread count (0 = auto-detect)
     */
    explicit BinarySplitting(Multiplier& multiplier, unsigned int num_threads = 0);

    BSResult compute(unsigned long a, unsigned long b);
    static unsigned long terms_needed(size_t digits);
    unsigned int thread_count() const { return num_threads_; }

    /**
     * @brief Enable checkpointing.
     * @param dir Directory to save checkpoint files
     * @param interval_seconds Save checkpoint every N seconds
     */
    void enable_checkpointing(const std::string& dir, int interval_seconds = 60);

    /**
     * @brief Enable out-of-core mode: compute wide, merge narrow.
     *
     * Phase 1: Splits the range into many small subtrees, computes each
     * in parallel using all cores, and serializes results to disk.
     * Phase 2: Merges results from disk in a bottom-up cascade,
     * loading only two at a time.
     *
     * Requires checkpointing to be enabled (uses the same directory).
     * Gives much better CPU utilization for large computations at the
     * cost of disk I/O (~400 GB for 50B digits).
     *
     * @param num_chunks Number of subtrees to split into (0 = auto, ~2x threads)
     */
    void enable_out_of_core(unsigned int num_chunks = 0);

    /**
     * @brief Try to resume from a checkpoint.
     * @return true if a valid checkpoint was found and loaded
     */
    bool try_resume(unsigned long a, unsigned long b, BSResult& result);

private:
    Multiplier& multiplier_;
    unsigned int num_threads_;

    // Checkpointing state
    bool checkpointing_enabled_ = false;
    std::string checkpoint_dir_;
    int checkpoint_interval_ = 60;

    // Out-of-core state
    bool out_of_core_enabled_ = false;
    unsigned int ooc_num_chunks_ = 0;

    static constexpr unsigned long PARALLEL_THRESHOLD = 64;

    BSResult base_case(unsigned long a);

    /**
     * @brief Merge with parallel multiplications.
     * Runs P*P, Q*Q, Q*R_left, P*R_right concurrently.
     */
    BSResult merge_parallel(BSResult& left, BSResult& right);

    BSResult compute_sequential(unsigned long a, unsigned long b);
    BSResult compute_parallel(unsigned long a, unsigned long b, int depth);
    BSResult compute_out_of_core(unsigned long a, unsigned long b);

    void save_checkpoint(unsigned long a, unsigned long b, const BSResult& result);
};

} // namespace pi
