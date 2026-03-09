/**
 * @file test_chunked_writer.cpp
 * @brief TDD tests for ChunkedWriter.
 */

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <cstdio>
#include "io/chunked_writer.h"

class ChunkedWriterTest : public ::testing::Test {
protected:
    std::string test_file;

    void SetUp() override {
        test_file = std::string(TEST_DATA_DIR) + "/test_output_" +
                    std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + ".txt";
    }

    void TearDown() override {
        std::remove(test_file.c_str());
    }

    std::string read_file(const std::string& path) {
        std::ifstream f(path);
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    }
};

TEST_F(ChunkedWriterTest, WriteSimpleString) {
    {
        pi::ChunkedWriter writer(test_file, 1000);
        writer.write("3.14159265");
        writer.close();
    }

    std::string content = read_file(test_file);
    EXPECT_EQ(content, "3.14159265");
}

TEST_F(ChunkedWriterTest, WriteInChunks) {
    {
        pi::ChunkedWriter writer(test_file, 5);  // Tiny chunk size
        writer.write("1234567890");
        writer.close();
    }

    std::string content = read_file(test_file);
    EXPECT_EQ(content, "1234567890");
}

TEST_F(ChunkedWriterTest, MultipleWrites) {
    {
        pi::ChunkedWriter writer(test_file, 1000);
        writer.write("3.");
        writer.write("14159");
        writer.write("26535");
        writer.close();
    }

    std::string content = read_file(test_file);
    EXPECT_EQ(content, "3.1415926535");
}

TEST_F(ChunkedWriterTest, BytesWrittenTracking) {
    pi::ChunkedWriter writer(test_file, 1000);
    EXPECT_EQ(writer.bytes_written(), 0u);

    writer.write("12345");
    EXPECT_EQ(writer.bytes_written(), 5u);

    writer.write("67890");
    EXPECT_EQ(writer.bytes_written(), 10u);

    writer.close();
}

TEST_F(ChunkedWriterTest, LargeWrite) {
    std::string large(100000, '9');
    {
        pi::ChunkedWriter writer(test_file, 10000);
        writer.write(large);
        writer.close();
    }

    std::string content = read_file(test_file);
    EXPECT_EQ(content.size(), 100000u);
    EXPECT_EQ(content, large);
}

TEST_F(ChunkedWriterTest, InvalidPathThrows) {
    EXPECT_THROW(pi::ChunkedWriter("/nonexistent/path/file.txt"), std::runtime_error);
}
