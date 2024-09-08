/**
 * This file contains the implementation of the tokenizer class.
 * 
 * Tiktoken-style GPT-4o regex (o200k_base) + BPE tokenizer.
 */

#include "tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <vector>
#include <string>
#include <unicode/regex.h>
#include <cassert>

Tokenizer::Tokenizer(const std::string& encoder_file) {
    // Load encoder from file
    std::ifstream file(encoder_file);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        Rank rank;
        if (!(iss >> token >> rank)) { break; }
        encoder_[token] = rank;
        decoder_[rank] = token;
    }

    // Initialize special tokens
    special_tokens_encoder_["<|endoftext|>"] = 50256;
    special_tokens_decoder_[50256] = "<|endoftext|>";

    // Initialize sorted_token_bytes_
    sorted_token_bytes_.reserve(encoder_.size());
    for (const auto& pair : encoder_) {
        sorted_token_bytes_.push_back(pair.first);
    }
    std::sort(sorted_token_bytes_.begin(), sorted_token_bytes_.end());

    // Initialize ICU regex patterns
    initialize_regex_patterns();
}

Tokenizer::~Tokenizer() {
    delete regex_pattern_;
    delete special_regex_pattern_;
}

std::vector<Tokenizer::Rank> Tokenizer::encode(const std::string& text, const std::unordered_set<std::string>& allowed_special) const {
    return _encode_native(text, allowed_special).first;
}

std::vector<Tokenizer::Rank> Tokenizer::_encode_ordinary_native(const std::string& text) const {
    std::vector<Rank> ret;
    for (const auto& piece : regex_split(text)) {
        auto it = encoder_.find(piece);
        if (it != encoder_.end()) {
            ret.push_back(it->second);
        } else {
            auto encoded = byte_pair_encode(piece);
            ret.insert(ret.end(), encoded.begin(), encoded.end());
        }
    }
    return ret;
}

std::pair<std::vector<Tokenizer::Rank>, size_t> Tokenizer::_encode_native(const std::string& text, const std::unordered_set<std::string>& allowed_special) const {
    // Implement this method
    // It should handle special tokens and use _encode_ordinary_native for regular text
}

std::pair<std::vector<Tokenizer::Rank>, std::unordered_set<std::vector<Tokenizer::Rank>>> Tokenizer::_encode_unstable_native(const std::string& text, const std::unordered_set<std::string>& allowed_special) const {
    // Implement this method
    // It should handle unstable tokens and use _encode_native
}

Tokenizer::ByteString Tokenizer::decode(const std::vector<Rank>& tokens) const {
    ByteString result;
    for (const auto& token : tokens) {
        if (decoder_.find(token) != decoder_.end()) {
            result += decoder_.at(token);
        } else if (special_tokens_decoder_.find(token) != special_tokens_decoder_.end()) {
            result += special_tokens_decoder_.at(token);
        }
    }
    return result;
}

Tokenizer::Rank Tokenizer::encode_single_token(const ByteString& piece) const {
    auto it = encoder_.find(piece);
    if (it != encoder_.end()) {
        return it->second;
    }
    auto special_it = special_tokens_encoder_.find(piece);
    if (special_it != special_tokens_encoder_.end()) {
        return special_it->second;
    }
    throw std::runtime_error("Token not found");
}

std::vector<Tokenizer::Rank> Tokenizer::encode_single_piece(const ByteString& piece) const {
    auto it = encoder_.find(piece);
    if (it != encoder_.end()) {
        return {it->second};
    }
    return byte_pair_encode(piece);
}

Tokenizer::ByteString Tokenizer::decode_single_token_bytes(Rank token) const {
    auto it = decoder_.find(token);
    if (it != decoder_.end()) {
        return it->second;
    }
    auto special_it = special_tokens_decoder_.find(token);
    if (special_it != special_tokens_decoder_.end()) {
        return special_it->second;
    }
    throw std::runtime_error("Token not found");
}

std::vector<Tokenizer::ByteString> Tokenizer::token_byte_values() const {
    return sorted_token_bytes_;
}

std::vector<Tokenizer::ByteString> Tokenizer::regex_split(const std::string& text) const {
    std::vector<ByteString> result;
    
    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString utext = icu::UnicodeString::fromUTF8(text);
    icu::RegexMatcher* matcher = regex_pattern_->matcher(utext, status);

    if (U_FAILURE(status)) {
        // Handle error
        delete matcher;
        return result;
    }

    while (matcher->find()) {
        icu::UnicodeString match = matcher->group(status);
        std::string utf8Match;
        match.toUTF8String(utf8Match);
        result.push_back(utf8Match);
    }

    delete matcher;
    return result;
}

std::vector<std::pair<size_t, Tokenizer::Rank>> Tokenizer::_byte_pair_merge(const ByteString& piece) const {
    std::vector<std::pair<size_t, Rank>> parts;
    parts.reserve(piece.size() + 1);

    Rank min_rank = std::numeric_limits<Rank>::max();
    size_t min_rank_idx = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < piece.size() - 1; ++i) {
        ByteString token = piece.substr(i, 2);
        Rank rank = encoder_.count(token) ? encoder_.at(token) : std::numeric_limits<Rank>::max();
        if (rank < min_rank) {
            min_rank = rank;
            min_rank_idx = i;
        }
        parts.emplace_back(i, rank);
    }
    parts.emplace_back(piece.size() - 1, std::numeric_limits<Rank>::max());
    parts.emplace_back(piece.size(), std::numeric_limits<Rank>::max());

    auto get_rank = [this, &piece, &parts](size_t i) {
        if (i + 3 < parts.size()) {
            ByteString token = piece.substr(parts[i].first, parts[i + 3].first - parts[i].first);
            return encoder_.count(token) ? encoder_.at(token) : std::numeric_limits<Rank>::max();
        }
        return std::numeric_limits<Rank>::max();
    };

    while (min_rank != std::numeric_limits<Rank>::max()) {
        size_t i = min_rank_idx;
        if (i > 0) {
            parts[i - 1].second = get_rank(i - 1);
        }
        parts[i].second = get_rank(i);
        parts.erase(parts.begin() + i + 1);

        min_rank = std::numeric_limits<Rank>::max();
        min_rank_idx = std::numeric_limits<size_t>::max();
        for (size_t j = 0; j < parts.size() - 1; ++j) {
            if (parts[j].second < min_rank) {
                min_rank = parts[j].second;
                min_rank_idx = j;
            }
        }
    }

    return parts;
}

std::vector<Tokenizer::Rank> Tokenizer::byte_pair_encode(const ByteString& piece) const {
    assert(piece.size() > 1);
    auto merged = _byte_pair_merge(piece);
    std::vector<Rank> result;
    result.reserve(merged.size() - 1);

    for (size_t i = 0; i < merged.size() - 1; ++i) {
        ByteString token = piece.substr(merged[i].first, merged[i + 1].first - merged[i].first);
        result.push_back(encoder_.at(token));
    }

    return result;
}

std::vector<Tokenizer::ByteString> Tokenizer::byte_pair_split(const ByteString& piece) const {
    assert(piece.size() > 1);
    auto merged = _byte_pair_merge(piece);
    std::vector<ByteString> result;
    result.reserve(merged.size() - 1);

    for (size_t i = 0; i < merged.size() - 1; ++i) {
        result.push_back(piece.substr(merged[i].first, merged[i + 1].first - merged[i].first));
    }

    return result;
}

void Tokenizer::initialize_regex_patterns() {
    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString pattern = icu::UnicodeString::fromUTF8(
        R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|)"
        R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|)"
        R"(\p{N}{1,3}|)"
        R"( ?[^\s\p{L}\p{N}]+[\r\n/]*|)"
        R"(\s*[\r\n]+|)"
        R"(\s+(?!\S)|)"
        R"(\s+)"
    );
    regex_pattern_ = icu::RegexPattern::compile(pattern, 0, status);
    if (U_FAILURE(status)) {
        // Handle error
    }

    // Initialize special_regex_pattern_
    std::string special_pattern = ""; // Construct this based on your special tokens
    for (const auto& token : special_tokens_encoder_) {
        if (!special_pattern.empty()) {
            special_pattern += "|";
        }
        special_pattern += std::regex_replace(token.first, std::regex("[\\^$.*+?()\\[\\]{}\\\\|]"), "\\$&");
    }
    special_regex_pattern_ = icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(special_pattern), 0, status);
    if (U_FAILURE(status)) {
        throw std::runtime_error("Failed to compile special regex pattern");
    }
}
