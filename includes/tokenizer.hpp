/**
 * This file contains the tokenizer class.
 * 
 * It is based on the tiktoken tokenizer, used for LLaMa 3.1. The paper quotes "We use a vocabulary with 128K tokens. Our token 
 * vocabulary combines 100K tokens from the tiktoken3 tokenizer with 28K additional tokens to better support non-English languages". 
 * Meta did not release any implementation details on the 28K additional tokens, so I implemented the base 
 * Tiktoken-style GPT-4o regex (o200k_base) + BPE tokenizer below.
 * 
 */

#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <unicode/regex.h>

class Tokenizer {
public:
    using Rank = uint32_t;
    using ByteString = std::string;

    Tokenizer(const std::string& encoder_file);
    ~Tokenizer();

    // Core functionality
    std::vector<Rank> encode(const std::string& text, const std::unordered_set<std::string>& allowed_special = {}) const;
    ByteString decode(const std::vector<Rank>& tokens) const;

    // Single token operations
    Rank encode_single_token(const ByteString& piece) const;
    std::vector<Rank> encode_single_piece(const ByteString& piece) const;
    ByteString decode_single_token_bytes(Rank token) const;

    // Miscellaneous
    std::vector<ByteString> token_byte_values() const;

private:
    std::unordered_map<ByteString, Rank> encoder_;
    std::unordered_map<std::string, Rank> special_tokens_encoder_;
    std::unordered_map<Rank, ByteString> decoder_;
    std::unordered_map<Rank, ByteString> special_tokens_decoder_;
    std::vector<ByteString> sorted_token_bytes_;

    // ICU regex patterns
    icu::RegexPattern* regex_pattern_;
    icu::RegexPattern* special_regex_pattern_;

    // Helper methods
    std::vector<ByteString> regex_split(const std::string& text) const;
    std::vector<Rank> byte_pair_encode(const ByteString& piece) const;
    std::vector<ByteString> byte_pair_split(const ByteString& piece) const;
    std::vector<std::pair<size_t, Rank>> _byte_pair_merge(const ByteString& piece) const;

    // Encoding methods
    std::vector<Rank> _encode_ordinary_native(const std::string& text) const;
    std::pair<std::vector<Rank>, size_t> _encode_native(const std::string& text, const std::unordered_set<std::string>& allowed_special) const;
    std::pair<std::vector<Rank>, std::unordered_set<std::vector<Rank>>> _encode_unstable_native(const std::string& text, const std::unordered_set<std::string>& allowed_special) const;

    // Initialize ICU regex patterns
    void initialize_regex_patterns();
};

#endif // TOKENIZER_HPP