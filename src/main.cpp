/**
 * @file main.cpp
 * @brief CLI entry point for pi_compute.
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include <memory>
#include "engine/pi_engine.h"
#include "arithmetic/gmp_multiplier.h"
#include "io/chunked_writer.h"

#ifdef PI_CUDA_ENABLED
#include "arithmetic/gpu_ntt_multiplier.h"
#endif

void print_usage() {
    std::cout << R"(
pi_compute - High-performance pi digit calculator

USAGE:
    pi_compute [OPTIONS] --digits <N>

OPTIONS:
    --digits <N>        Number of decimal digits to compute (required)
    --gpu               Enable GPU acceleration (requires CUDA build)
    --gpu-threshold <N> Min GMP limbs for GPU path (default: 1000)
    --output <FILE>     Output file path (default: pi_digits.txt)
    --verbose           Verbose progress output
    --help              Show this help message

EXAMPLES:
    pi_compute --digits 1000000 --verbose
    pi_compute --digits 10000000 --gpu --verbose
    pi_compute --digits 100000000 --gpu --output pi_100M.txt --verbose
)" << std::endl;
}

int main(int argc, char* argv[]) {
    pi::PiConfig config;
    bool has_digits = false;
    bool use_gpu = false;
    size_t gpu_threshold = 1000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--digits" && i + 1 < argc) {
            config.digits = std::stoul(argv[++i]);
            has_digits = true;
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--gpu-threshold" && i + 1 < argc) {
            gpu_threshold = std::stoul(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }

    if (!has_digits) {
        std::cerr << "Error: --digits is required" << std::endl;
        print_usage();
        return 1;
    }

    if (config.digits == 0) {
        std::cerr << "Error: --digits must be > 0" << std::endl;
        return 1;
    }

    try {
        // Select multiplier: GPU or CPU
        std::unique_ptr<pi::Multiplier> multiplier;

        if (use_gpu) {
#ifdef PI_CUDA_ENABLED
            auto gpu_mult = std::make_unique<pi::GpuNttMultiplier>(gpu_threshold);
            if (config.verbose) {
                std::cout << "GPU: " << gpu_mult->device_name()
                          << " (threshold: " << gpu_threshold << " limbs)" << std::endl;
            }
            multiplier = std::move(gpu_mult);
#else
            std::cerr << "Error: --gpu requires CUDA build (cmake -DENABLE_CUDA=ON)" << std::endl;
            std::cerr << "Falling back to CPU..." << std::endl;
            multiplier = std::make_unique<pi::GmpMultiplier>();
#endif
        } else {
            multiplier = std::make_unique<pi::GmpMultiplier>();
        }

        pi::PiEngine engine(*multiplier);
        pi::PiResult result = engine.compute(config);

        // Write to file
        pi::ChunkedWriter writer(config.output_file);
        writer.write(result.digits);
        writer.close();

        std::cout << "Computed " << config.digits << " digits of pi in "
                  << result.elapsed_seconds << " seconds ("
                  << result.terms_used << " terms"
                  << (use_gpu ? ", GPU" : ", CPU") << ")." << std::endl;
        std::cout << "Output written to: " << config.output_file << std::endl;

        // Print first 80 characters as preview
        if (result.digits.size() > 80) {
            std::cout << "Preview: " << result.digits.substr(0, 80) << "..." << std::endl;
        } else {
            std::cout << "Result: " << result.digits << std::endl;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
