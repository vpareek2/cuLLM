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
#include <regex>

class Tokenizer {
public:
    using Rank = uint32_t;
    using ByteVector = std::vector<uint8_t>;

    Tokenizer();
    ~Tokenizer();

    // Core functionality
    std::vector<Rank> encode(const std::string& text) const;
    std::string decode(const std::vector<Rank>& tokens) const;

    // Special tokens
    static constexpr const char* ENDOFTEXT = "ENDOFTEXT";
    static constexpr const char* ENDOFPROMPT = "ENDOFPROMPT";
    static const std::unordered_map<std::string, Rank> SPECIAL_TOKENS;

private:
    std::unordered_map<ByteVector, Rank> encoder_;
    std::unordered_map<Rank, ByteVector> decoder_;
    std::regex regex_pattern_;

    // Regex pattern definition
    static const std::string REGEX_PATTERN;

    // Helper methods
    std::vector<std::string> regex_split(const std::string& text) const;
    std::vector<Rank> bpe_encode(const std::string& token) const;

    // BPE merge step
    std::vector<Tokenizer::Rank> Tokenizer::byte_pair_merge(const ByteVector& piece, const std::unordered_map<ByteVector, Rank>& ranks) const;
};

// GPT-4o (o200k_base) regex patterns
const std::vector<std::string> REGEX_PATTERNS = {
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?)",
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?)",
    R"(\p{N}{1,3})",
    R"( ?[^\s\p{L}\p{N}]+[\r\n/]*)",
    R"(\s*[\r\n]+)",
    R"(\s+(?!\S))",
    R"(\s+)"
};

// Join the patterns with '|' (or operator)
const std::string Tokenizer::REGEX_PATTERN =
    []{
        std::string joined;
        for (size_t i = 0; i < REGEX_PATTERNS.size(); ++i) {
            if (i > 0) joined += "|";
            joined += REGEX_PATTERNS[i];
        }
        return joined;
    }();

// Define special tokens as done in the o200k_base/GPT4-o tokenizer
const std::unordered_map<std::string, Tokenizer::Rank> Tokenizer::SPECIAL_TOKENS = {
    {ENDOFTEXT, 199999},
    {ENDOFPROMPT, 200018}
};

#endif // TOKENIZER_HPP