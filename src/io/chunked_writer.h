#pragma once

/**
 * @file chunked_writer.h
 * @brief Streaming output of pi digits to file in configurable chunks.
 */

#include <string>
#include <fstream>
#include <cstddef>

namespace pi {

class ChunkedWriter {
public:
    /**
     * @brief Construct a chunked writer.
     * @param filepath Output file path
     * @param chunk_size Number of digits per write operation (default: 1,000,000)
     */
    explicit ChunkedWriter(const std::string& filepath, size_t chunk_size = 1000000);
    ~ChunkedWriter();

    /**
     * @brief Write a string of digits to the output file.
     * @param digits The digit string to write (may include "3." prefix on first call)
     */
    void write(const std::string& digits);

    /**
     * @brief Flush and close the output file.
     */
    void close();

    /**
     * @brief Get the total number of characters written.
     */
    size_t bytes_written() const { return bytes_written_; }

private:
    std::ofstream file_;
    size_t chunk_size_;
    size_t bytes_written_ = 0;
};

} // namespace pi
