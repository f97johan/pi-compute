#pragma once

/**
 * @file pi_engine.h
 * @brief Top-level orchestrator for computing pi to arbitrary precision.
 *
 * Coordinates binary splitting, division, square root, base conversion,
 * and output writing.
 */

#include <string>
#include <cstddef>
#include <functional>
#include "../arithmetic/multiplier.h"

namespace pi {

/**
 * @brief Configuration for pi computation.
 */
struct PiConfig {
    size_t digits = 1000;                     ///< Number of decimal digits to compute
    std::string output_file = "pi_digits.txt"; ///< Output file path
    bool verbose = false;                      ///< Print progress messages
};

/**
 * @brief Result of pi computation.
 */
struct PiResult {
    std::string digits;       ///< The computed digits as a string (e.g., "3.14159...")
    double elapsed_seconds;   ///< Wall-clock time for computation
    unsigned long terms_used; ///< Number of Chudnovsky series terms used
};

class PiEngine {
public:
    /**
     * @brief Construct a PiEngine with the given multiplier strategy.
     * @param multiplier The multiplication backend (CPU or GPU)
     */
    explicit PiEngine(Multiplier& multiplier);

    /**
     * @brief Compute pi to the specified number of decimal digits.
     * @param config Computation configuration
     * @return PiResult containing the digit string and timing info
     */
    PiResult compute(const PiConfig& config);

    /**
     * @brief Compute pi and return just the digit string.
     * @param digits Number of decimal digits
     * @return String like "3.14159265..."
     */
    std::string compute_digits(size_t digits);

private:
    Multiplier& multiplier_;
};

} // namespace pi
