/**
 * @file main.cpp
 * @brief CLI entry point for pi_compute.
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include <memory>
#include <functional>
#include <thread>
#include "engine/pi_engine.h"
#include "arithmetic/gmp_multiplier.h"
#include "io/chunked_writer.h"

#ifdef PI_FLINT_ENABLED
#include "arithmetic/flint_multiplier.h"
#endif

#ifdef PI_CUDA_ENABLED
#include "arithmetic/gpu_ntt_multiplier.h"
#include "arithmetic/int_ntt_multiplier.h"
#endif

void print_usage() {
    std::cout << R"(
pi_compute - High-performance pi digit calculator

USAGE:
    pi_compute [OPTIONS] --digits <N>

OPTIONS:
    --digits <N>        Number of decimal digits to compute (required)
    --gpu               Enable GPU via cuFFT (FP64, best for data center GPUs)
    --ntt               Enable GPU via integer NTT (INT64, best for consumer GPUs)
    --gpus <N>          Number of GPUs to use (0 = auto-detect all, default: 0)
    --gpu-threshold <N> Min GMP limbs for GPU path (default: 10000)
    --threads <N>       Number of CPU threads (0 = auto-detect, default: 0)
    --flint             Use FLINT library for multi-threaded multiplication
                        (requires build with -DENABLE_FLINT=ON)
    --integer-math      Use integer-only sqrt+divide (avoids single-threaded mpf)
    --out-of-core       Enable out-of-core mode: compute wide, merge narrow
                        (better CPU utilization, requires --checkpoint)
    --output <FILE>     Output file path (default: pi_digits.txt)
    --checkpoint <DIR>  Enable checkpointing to directory (for crash recovery)
    --resume            Resume from checkpoint (requires --checkpoint)
    --verbose           Verbose progress output
    --help              Show this help message

EXAMPLES:
    pi_compute --digits 1000000 --verbose
    pi_compute --digits 10000000 --gpu --verbose
    pi_compute --digits 100000000 --gpu --gpus 8 --output pi_100M.txt --verbose
)" << std::endl;
}

int main(int argc, char* argv[]) {
    pi::PiConfig config;
    bool has_digits = false;
    bool use_gpu = false;
    bool use_ntt = false;
    bool use_flint = false;
    size_t gpu_threshold = 10000;
    int num_gpus = 0;  // 0 = auto-detect

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
        } else if (arg == "--flint") {
            use_flint = true;
        } else if (arg == "--integer-math") {
            config.integer_math = true;
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--ntt") {
            use_ntt = true;
        } else if (arg == "--gpus" && i + 1 < argc) {
            num_gpus = std::stoi(argv[++i]);
            use_gpu = true;
        } else if (arg == "--gpu-threshold" && i + 1 < argc) {
            gpu_threshold = std::stoul(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoul(argv[++i]);
        } else if (arg == "--out-of-core") {
            config.out_of_core = true;
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            config.checkpoint_dir = argv[++i];
        } else if (arg == "--resume") {
            config.resume = true;
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
        // Select multiplier: GPU (cuFFT), NTT (integer), or CPU
        std::unique_ptr<pi::Multiplier> multiplier;
        std::string mode_name = "CPU";

#ifdef PI_CUDA_ENABLED
        // Function pointer for printing stats (set if GPU/NTT mode)
        std::function<void()> print_gpu_stats;
#endif

        if (use_ntt) {
#ifdef PI_CUDA_ENABLED
            auto ntt_mult = std::make_unique<pi::IntNttMultiplier>(gpu_threshold, num_gpus);
            if (config.verbose) {
                std::cout << "Int NTT: " << ntt_mult->device_name()
                          << " (" << ntt_mult->gpu_count() << " GPU(s)"
                          << ", threshold: " << gpu_threshold << " limbs)" << std::endl;
            }
            auto* ntt_ptr = ntt_mult.get();
            print_gpu_stats = [ntt_ptr]() { ntt_ptr->print_stats(); };
            multiplier = std::move(ntt_mult);
            mode_name = "NTT";
#else
            std::cerr << "Error: --ntt requires CUDA build" << std::endl;
            multiplier = std::make_unique<pi::GmpMultiplier>();
#endif
        } else if (use_gpu) {
#ifdef PI_CUDA_ENABLED
            auto gpu_mult = std::make_unique<pi::GpuNttMultiplier>(gpu_threshold, num_gpus);
            if (config.verbose) {
                std::cout << "GPU: " << gpu_mult->device_name()
                          << " (" << gpu_mult->gpu_count() << " GPU(s)"
                          << ", threshold: " << gpu_threshold << " limbs)" << std::endl;
            }
            auto* gpu_ptr = gpu_mult.get();
            print_gpu_stats = [gpu_ptr]() { gpu_ptr->print_stats(); };
            multiplier = std::move(gpu_mult);
            mode_name = "GPU";
#else
            std::cerr << "Error: --gpu requires CUDA build" << std::endl;
            multiplier = std::make_unique<pi::GmpMultiplier>();
#endif
        } else if (use_flint) {
#ifdef PI_FLINT_ENABLED
            multiplier = std::make_unique<pi::FlintMultiplier>(config.num_threads);
            if (config.verbose) {
                unsigned int nt = config.num_threads == 0 ? std::thread::hardware_concurrency() : config.num_threads;
                std::cout << "FLINT: multi-threaded multiplication ("
                          << nt << " threads)" << std::endl;
            }
            mode_name = "FLINT";
#else
            std::cerr << "Error: --flint requires build with -DENABLE_FLINT=ON" << std::endl;
            std::cerr << "  cmake -B build -DENABLE_FLINT=ON" << std::endl;
            return 1;
#endif
        } else {
            multiplier = std::make_unique<pi::GmpMultiplier>();
        }

        pi::PiEngine engine(*multiplier);
        pi::PiResult result = engine.compute(config);

#ifdef PI_CUDA_ENABLED
        if (config.verbose && print_gpu_stats) {
            print_gpu_stats();
        }
#endif

        // Write to file
        pi::ChunkedWriter writer(config.output_file);
        writer.write(result.digits);
        writer.close();

        std::cout << "Computed " << config.digits << " digits of pi in "
                  << result.elapsed_seconds << " seconds ("
                  << result.terms_used << " terms, " << mode_name << ")." << std::endl;
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
