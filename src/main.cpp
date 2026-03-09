/**
 * @file main.cpp
 * @brief CLI entry point for pi_compute.
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include "engine/pi_engine.h"
#include "arithmetic/gmp_multiplier.h"
#include "io/chunked_writer.h"

void print_usage() {
    std::cout << R"(
pi_compute - High-performance pi digit calculator

USAGE:
    pi_compute [OPTIONS] --digits <N>

OPTIONS:
    --digits <N>        Number of decimal digits to compute (required)
    --output <FILE>     Output file path (default: pi_digits.txt)
    --verbose           Verbose progress output
    --help              Show this help message

EXAMPLES:
    pi_compute --digits 1000
    pi_compute --digits 1000000 --output pi_1M.txt --verbose
)" << std::endl;
}

int main(int argc, char* argv[]) {
    pi::PiConfig config;
    bool has_digits = false;

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
        // Use CPU multiplier (Phase 1)
        pi::GmpMultiplier multiplier;
        pi::PiEngine engine(multiplier);

        pi::PiResult result = engine.compute(config);

        // Write to file
        pi::ChunkedWriter writer(config.output_file);
        writer.write(result.digits);
        writer.close();

        std::cout << "Computed " << config.digits << " digits of pi in "
                  << result.elapsed_seconds << " seconds ("
                  << result.terms_used << " terms)." << std::endl;
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
