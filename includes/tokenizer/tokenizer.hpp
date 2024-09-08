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
#include <memory>
#include <optional>
#include <functional>

// Custom hash function for std::vector<unsigned int>
namespace std {
    template<>
    struct hash<vector<unsigned int>> {
        size_t operator()(const vector<unsigned int>& v) const {
            size_t seed = v.size();
            for(auto& i : v) {
                seed ^= hash<unsigned int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}

// Vector hasher for custom hash tables
struct VectorHasher {
    size_t operator()(const std::vector<unsigned int>& v) const {
        size_t seed = v.size();
        for (auto& i : v) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Main Tokenizer class
class Tokenizer {
public:
    using Rank = uint32_t;
    using ByteString = std::string;

    // Constructor and destructor
    Tokenizer(const std::string& encoder_file);
    ~Tokenizer();

    // Core functionality
    std::vector<Rank> encode(const std::string& text, const std::unordered_set<std::string>& allowed_special = {}) const;
    ByteString decode(const std::vector<Rank>& tokens) const;

    // Encoding methods
    std::vector<Rank> encode_ordinary(const std::string& text) const;
    std::pair<std::vector<Rank>, std::unordered_set<std::vector<Rank>>> encode_with_unstable(const std::string& text, const std::unordered_set<std::string>& allowed_special) const;

    // Single token operations
    std::optional<Rank> encode_single_token(const ByteString& piece) const;
    std::vector<Rank> encode_single_piece(const ByteString& piece) const;
    std::optional<ByteString> decode_single_token_bytes(Rank token) const;

    // Miscellaneous
    std::vector<ByteString> token_byte_values() const;

private:
    // Internal data structures
    std::unordered_map<ByteString, Rank> encoder_;
    std::unordered_map<std::string, Rank> special_tokens_encoder_;
    std::unordered_map<Rank, ByteString> decoder_;
    std::unordered_map<Rank, ByteString> special_tokens_decoder_;
    std::vector<ByteString> sorted_token_bytes_;

    // ICU regex patterns
    std::unique_ptr<icu::RegexPattern> regex_pattern_;
    std::unique_ptr<icu::RegexPattern> special_regex_pattern_;

    // Helper methods
    std::vector<ByteString> regex_split(const std::string& text) const;
    std::vector<Rank> byte_pair_encode(const ByteString& piece) const;
    std::vector<std::pair<size_t, Rank>> byte_pair_merge(const ByteString& piece) const;

    // New encoding methods
    std::pair<std::vector<Rank>, size_t> encode_native(const std::string& text, const std::unordered_set<std::string>& allowed_special) const;

    // Initialization methods
    void initialize_regex_patterns();
    void initialize_special_tokens();

    // Constant GPT-4o REGEX pattern
    static constexpr const char* REGEX_PATTERN = R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

    // Additional methods
    std::vector<ByteString> byte_pair_split(const ByteString& piece) const;

    // Helper methods
    static bool is_token_all_space(const ByteString& token);
    std::vector<Rank> find_single_token_completions(
        const std::vector<ByteString>& sorted_token_bytes,
        const std::string& unstable_str) const;
};

#endif // TOKENIZER_HPP