#pragma once

/**
 * @file pi_engine.h
 * @brief Top-level orchestrator for computing pi to arbitrary precision.
 */

#include <string>
#include <cstddef>
#include <functional>
#include "../arithmetic/multiplier.h"

namespace pi {

struct PiConfig {
    size_t digits = 1000;
    std::string output_file = "pi_digits.txt";
    bool verbose = false;
    std::string checkpoint_dir;
    bool resume = false;
    unsigned int num_threads = 0;  ///< 0 = auto-detect
};

struct PiResult {
    std::string digits;
    double elapsed_seconds;
    unsigned long terms_used;
};

class PiEngine {
public:
    explicit PiEngine(Multiplier& multiplier);
    PiResult compute(const PiConfig& config);
    std::string compute_digits(size_t digits);

    /// Get current RSS in MB (Linux: /proc/self/status, macOS: task_info)
    static size_t get_rss_mb();

private:
    Multiplier& multiplier_;
};

} // namespace pi
