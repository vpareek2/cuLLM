/**
 * This file contains the tokenizer class.
 * 
 * It is based on the tiktoken tokenizer, used for LLaMa 3.1. The paper quotes "We use a vocabulary with 128K tokens. Our token 
 * vocabulary combines 100K tokens from the tiktoken3 tokenizer with 28K additional tokens to better support non-English languages". 
 * Meta did not release any implementation details on the 28K additional tokens, so I implemented the base 
 * Tiktoken-style GPT-4o regex (o200k_base) + BPE tokenizer below.
 * 
 */

// def o200k_base():
//     mergeable_ranks = load_tiktoken_bpe(
//         "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
//         expected_hash="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
//     )
//     special_tokens = {
//         ENDOFTEXT: 199999,
//         ENDOFPROMPT: 200018,
//     }
//     # This regex could be made more efficient
//     pat_str = "|".join(
//         [
//             r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
//             r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
//             r"""\p{N}{1,3}""",
//             r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
//             r"""\s*[\r\n]+""",
//             r"""\s+(?!\S)""",
//             r"""\s+""",
//         ]
//     )
//     return {
//         "name": "o200k_base",
//         "pat_str": pat_str,
//         "mergeable_ranks": mergeable_ranks,
//         "special_tokens": special_tokens,
//     }

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

    Tokenizer(const std::string& bpe_file, const std::string& regex_pattern);
    ~Tokenizer();

    // Core functionality
    std::vector<Rank> encode(const std::string& text) const;
    std::string decode(const std::vector<Rank>& tokens) const;

private:
    std::unordered_map<ByteVector, Rank> encoder_;
    std::unordered_map<Rank, ByteVector> decoder_;
    std::regex regex_pattern_;

    // Helper methods
    std::vector<std::string> regex_split(const std::string& text) const;
    std::vector<Rank> bpe_encode(const std::string& token) const;
    
    // BPE merge step
    static std::vector<Rank> byte_pair_merge(const ByteVector& piece,
                                             const std::unordered_map<ByteVector, Rank>& ranks);
};

#endif // TOKENIZER_HPP