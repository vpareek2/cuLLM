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
#include <pcre2.h>

class Tokenizer {
public:
    using Rank = uint32_t;
    using ByteString = std::string;

    Tokenizer();
    ~Tokenizer();

    // Core functionality
    std::vector<Rank> encode(const std::string& text) const;
    std::string decode(const std::vector<Rank>& tokens) const;

    // Special tokens
    static constexpr const char* ENDOFTEXT = "ENDOFTEXT";
    static constexpr const char* ENDOFPROMPT = "ENDOFPROMPT";
    static constexpr Rank ENDOFTEXT_TOKEN = 199999;
    static constexpr Rank ENDOFPROMPT_TOKEN = 200018;

private:
    std::unordered_map<ByteString, Rank> encoder_;
    std::unordered_map<Rank, ByteString> decoder_;
    pcre2_code* regex_pattern_;

    inline static const std::string REGEX_PATTERN =
        R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|)"
        R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|)"
        R"(\p{N}{1,3}|)"
        R"( ?[^\s\p{L}\p{N}]+[\r\n/]*|)"
        R"(\s*[\r\n]+|)"
        R"(\s+(?!\S)|)"
        R"(\s+)";

    // Helper methods
    std::vector<std::string> regex_split(const std::string& text) const;
    std::vector<Rank> bpe_encode(const std::string& token) const;

    // BPE merge step
    std::vector<Rank> byte_pair_merge(const ByteString& piece, const std::unordered_map<ByteString, Rank>& ranks) const;
};

#endif // TOKENIZER_HPP