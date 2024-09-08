#ifndef TOKENIZER_TEST_HPP
#define TOKENIZER_TEST_HPP

#include "tokenizer.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cstdio>

// Simple test framework
#define RUN_TEST(test) do { \
    std::cout << "Running " << #test << "... "; \
    if (test()) { \
        std::cout << "PASSED" << std::endl; \
    } else { \
        std::cout << "FAILED" << std::endl; \
        failed_tests++; \
    } \
    total_tests++; \
} while(0)

// Global test counters
int total_tests = 0;
int failed_tests = 0;

// Utility function for comparing vectors
template<typename T>
bool compare_vectors(const std::vector<T>& a, const std::vector<T>& b) {
    return a == b;
}

// Test cases
bool test_basic_encode_decode() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    std::string input = "Hello, world!";
    auto tokens = tokenizer.encode(input);
    auto decoded = tokenizer.decode(tokens);
    return input == decoded;
}

bool test_unicode_encoding() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    std::string input = "こんにちは世界! Здравствуй, мир! 🌍🌎🌏";
    auto tokens = tokenizer.encode(input);
    auto decoded = tokenizer.decode(tokens);
    return input == decoded;
}

bool test_whitespace_handling() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    std::string input = "  This  is \n a \t test  \r\n  with    spaces  ";
    auto tokens = tokenizer.encode(input);
    auto decoded = tokenizer.decode(tokens);
    return input == decoded;
}

bool test_number_encoding() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    std::string input = "123 45.67 1,000,000 3.14159265359";
    auto tokens = tokenizer.encode(input);
    auto decoded = tokenizer.decode(tokens);
    return input == decoded;
}

bool test_url_and_email_encoding() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    std::string input = "Visit https://www.example.com or email user@example.com";
    auto tokens = tokenizer.encode(input);
    auto decoded = tokenizer.decode(tokens);
    return input == decoded;
}

bool test_contractions_and_possessives() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    std::string input = "It's John's car. They're going to the store. I've been there.";
    auto tokens = tokenizer.encode(input);
    auto decoded = tokenizer.decode(tokens);
    return input == decoded;
}

bool test_long_input_encoding() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    std::string input(100000, 'a'); // 100,000 'a' characters
    auto tokens = tokenizer.encode(input);
    auto decoded = tokenizer.decode(tokens);
    return input == decoded;
}

bool test_empty_string_encoding() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    std::string input = "";
    auto tokens = tokenizer.encode(input);
    auto decoded = tokenizer.decode(tokens);
    return input == decoded && tokens.empty();
}

bool test_single_character_encoding() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    for (char c = 32; c < 127; ++c) { // Printable ASCII characters
        std::string input(1, c);
        auto tokens = tokenizer.encode(input);
        auto decoded = tokenizer.decode(tokens);
        if (input != decoded) {
            std::cout << "Failed for character: " << c << std::endl;
            return false;
        }
    }
    return true;
}

bool test_special_characters() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    std::string input = "!@#$%^&*()_+{}|:\"<>?`-=[]\\;',./";
    auto tokens = tokenizer.encode(input);
    auto decoded = tokenizer.decode(tokens);
    return input == decoded;
}

// Main function to run all tests
int run_all_tests() {
    RUN_TEST(test_basic_encode_decode);
    RUN_TEST(test_unicode_encoding);
    RUN_TEST(test_whitespace_handling);
    RUN_TEST(test_number_encoding);
    RUN_TEST(test_url_and_email_encoding);
    RUN_TEST(test_contractions_and_possessives);
    RUN_TEST(test_long_input_encoding);
    RUN_TEST(test_empty_string_encoding);
    RUN_TEST(test_single_character_encoding);
    RUN_TEST(test_special_characters);

    std::cout << "\nTest Summary:" << std::endl;
    std::cout << "Total tests: " << total_tests << std::endl;
    std::cout << "Passed: " << (total_tests - failed_tests) << std::endl;
    std::cout << "Failed: " << failed_tests << std::endl;

    return failed_tests;
}

int main() {
    return run_all_tests();
}

#endif // TOKENIZER_TEST_HPP