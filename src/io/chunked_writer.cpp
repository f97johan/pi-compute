/**
 * @file chunked_writer.cpp
 * @brief Chunked file writer implementation.
 */

#include "chunked_writer.h"
#include <stdexcept>

namespace pi {

ChunkedWriter::ChunkedWriter(const std::string& filepath, size_t chunk_size)
    : chunk_size_(chunk_size) {
    file_.open(filepath, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filepath);
    }
}

ChunkedWriter::~ChunkedWriter() {
    close();
}

void ChunkedWriter::write(const std::string& digits) {
    if (!file_.is_open()) {
        throw std::runtime_error("File is not open for writing");
    }

    // Write in chunks for better I/O performance
    size_t offset = 0;
    while (offset < digits.size()) {
        size_t len = std::min(chunk_size_, digits.size() - offset);
        file_.write(digits.data() + offset, static_cast<std::streamsize>(len));
        offset += len;
        bytes_written_ += len;
    }

    file_.flush();
}

void ChunkedWriter::close() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

} // namespace pi
