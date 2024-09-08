#include "tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <vector>
#include <string>
#include <unicode/regex.h>
#include <cassert>
#include <iostream>
#include <optional>

namespace {

const std::string base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

// Add this function before base64_decode
bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

// Base64 decoding function
// This function decodes a base64 encoded string to its original form
// It uses a lookup table and bitwise operations to convert groups of 4 base64 characters into 3 bytes
std::string base64_decode(std::string_view encoded_string) {
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    std::array<unsigned char, 4> char_array_4;
    std::array<unsigned char, 3> char_array_3;
    std::string ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret += char_array_3[i];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
    }

    return ret;
}

// Move VectorHasher here if it's only used internally
struct VectorHasher {
    size_t operator()(const std::vector<unsigned int>& v) const {
        size_t seed = v.size();
        for (auto& i : v) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

} // anonymous namespace

Tokenizer::Tokenizer(const std::string& encoder_file) {
    // Load encoder from file
    std::ifstream file(encoder_file);
    std::string line;
    Rank rank = 0;  // Start rank from 0
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        if (!(iss >> token)) { break; }
        ByteString decoded_token = base64_decode(token);
        encoder_[decoded_token] = rank;
        decoder_[rank] = decoded_token;
        rank++;
    }

    // Initialize sorted_token_bytes_
    sorted_token_bytes_.reserve(encoder_.size());
    for (const auto& pair : encoder_) {
        sorted_token_bytes_.push_back(pair.first);
    }
    std::sort(sorted_token_bytes_.begin(), sorted_token_bytes_.end());

    // Initialize ICU regex patterns and special tokens
    initialize_regex_patterns();
    initialize_special_tokens();
}

Tokenizer::~Tokenizer() = default;

void Tokenizer::initialize_special_tokens() {
    special_tokens_encoder_["<|endoftext|>"] = 199999;
    special_tokens_encoder_["<|endofprompt|>"] = 200018;
    
    // Populate special_tokens_decoder_ with the reverse mapping
    for (const auto& pair : special_tokens_encoder_) {
        special_tokens_decoder_[pair.second] = pair.first;
    }
}

void Tokenizer::initialize_regex_patterns() {
    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString pattern = icu::UnicodeString::fromUTF8(REGEX_PATTERN);
    regex_pattern_ = std::make_unique<icu::RegexPattern>(icu::RegexPattern::compile(pattern, 0, status));
    if (U_FAILURE(status)) {
        throw std::runtime_error("Failed to compile main regex pattern");
    }

    // Initialize special_regex_pattern_
    std::string special_pattern;
    for (const auto& token : special_tokens_encoder_) {
        if (!special_pattern.empty()) {
            special_pattern += "|";
        }
        special_pattern += std::regex_replace(token.first, std::regex("[\\^$.*+?()\\[\\]{}\\\\|]"), "\\$&");
    }
    special_regex_pattern_ = std::make_unique<icu::RegexPattern>(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(special_pattern), 0, status));
    if (U_FAILURE(status)) {
        throw std::runtime_error("Failed to compile special regex pattern");
    }
}

std::vector<Tokenizer::Rank> Tokenizer::encode(const std::string& text, const std::unordered_set<std::string>& allowed_special) const {
    return encode_native(text, allowed_special).first;
}

std::vector<Tokenizer::Rank> Tokenizer::encode_ordinary(const std::string& text) const {
    std::vector<Rank> ret;
    ret.reserve(text.length());  // Reserve space based on input length
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

std::pair<std::vector<Tokenizer::Rank>, size_t> Tokenizer::encode_native(const std::string& text, const std::unordered_set<std::string>& allowed_special) const {
    std::vector<Rank> ret;
    ret.reserve(text.length());  // Reserve space based on input length
    size_t start = 0;
    size_t last_piece_token_len = 0;

    while (start < text.length()) {
        // Find the next allowed special token
        std::string::size_type next_special_start = std::string::npos;
        std::string::size_type next_special_end = std::string::npos;

        for (const auto& special : allowed_special) {
            auto pos = text.find(special, start);
            if (pos != std::string::npos && (pos < next_special_start || next_special_start == std::string::npos)) {
                next_special_start = pos;
                next_special_end = pos + special.length();
            }
        }

        // Process the text up to the next special token (or end of string)
        std::string::size_type end = (next_special_start != std::string::npos) ? next_special_start : text.length();
        auto ordinary_tokens = encode_ordinary(text.substr(start, end - start));
        ret.insert(ret.end(), ordinary_tokens.begin(), ordinary_tokens.end());
        last_piece_token_len = ordinary_tokens.size();

        // Process the special token if found
        if (next_special_start != std::string::npos) {
            std::string special_token = text.substr(next_special_start, next_special_end - next_special_start);
            ret.push_back(special_tokens_encoder_.at(special_token));
            start = next_special_end;
            last_piece_token_len = 0;
        } else {
            break;
        }
    }

    return {ret, last_piece_token_len};
}

// Helper function for encode_with_unstable
bool is_token_all_space(const Tokenizer::ByteString& token) {
    return std::all_of(token.begin(), token.end(), [](unsigned char c) {
        return c == ' ' || c == '\n' || c == '\t';
    });
}

// Helper function for encode_with_unstable
std::vector<Tokenizer::Rank> find_single_token_completions(
    const std::vector<Tokenizer::ByteString>& sorted_token_bytes,
    const std::unordered_map<Tokenizer::ByteString, Tokenizer::Rank>& encoder,
    const std::string& unstable_str) {
    
    std::vector<Tokenizer::Rank> completions;
    auto it = std::lower_bound(sorted_token_bytes.begin(), sorted_token_bytes.end(), unstable_str,
        [](const std::string& a, const std::string& b) {
            return a < b;
        });
    while (it != sorted_token_bytes.end() && it->substr(0, unstable_str.length()) == unstable_str) {
        completions.push_back(encoder.at(*it));
        ++it;
    }
    return completions;
}

std::pair<std::vector<Tokenizer::Rank>, std::unordered_set<std::vector<Tokenizer::Rank>>> 
Tokenizer::encode_with_unstable(const std::string& text, const std::unordered_set<std::string>& allowed_special) const {
    auto [tokens, last_piece_token_len] = encode_native(text, allowed_special);
    if (last_piece_token_len == 0) {
        return {tokens, {}};
    }

    // Increase last_piece_token_len for unstable regex splits
    while (last_piece_token_len < tokens.size() && 
           is_token_all_space(decoder_.at(tokens[tokens.size() - last_piece_token_len - 1]))) {
        last_piece_token_len++;
    }

    std::vector<unsigned char> unstable_bytes;
    for (size_t i = tokens.size() - last_piece_token_len; i < tokens.size(); ++i) {
        const auto& token_bytes = decoder_.at(tokens[i]);
        unstable_bytes.insert(unstable_bytes.end(), token_bytes.begin(), token_bytes.end());
    }
    tokens.erase(tokens.end() - last_piece_token_len, tokens.end());

    std::unordered_set<std::vector<Rank>> completions;
    if (unstable_bytes.empty()) {
        return {tokens, completions};
    }

    // Find single tokens that start with unstable_bytes
    std::string unstable_str(unstable_bytes.begin(), unstable_bytes.end());
    auto single_token_completions = find_single_token_completions(sorted_token_bytes_, encoder_, unstable_str);
    for (auto rank : single_token_completions) {
        completions.insert({rank});
    }

    // Handle more complex cases
    for (size_t i = 1; i < unstable_bytes.size(); ++i) {
        std::string prefix(unstable_bytes.begin(), unstable_bytes.begin() + i);
        std::string suffix(unstable_bytes.begin() + i, unstable_bytes.end());

        auto it = std::lower_bound(sorted_token_bytes_.begin(), sorted_token_bytes_.end(), suffix,
            [](const std::string& a, const std::string& b) {
                return a < b;
            });
        while (it != sorted_token_bytes_.end() && it->substr(0, suffix.length()) == suffix) {
            std::string possibility = prefix + *it;

            std::vector<Rank> encoded;
            try {
                encoded = encode_ordinary(possibility);
            } catch (...) {
                encoded = byte_pair_encode(ByteString(possibility.begin(), possibility.end()));
            }

            std::vector<Rank> seq;
            size_t seq_len = 0;
            for (Rank token : encoded) {
                seq.push_back(token);
                seq_len += decoder_.at(token).size();
                if (seq_len >= unstable_bytes.size()) {
                    break;
                }
            }
            completions.insert(seq);
            ++it;
        }
    }

    // Handle potential regex split issues
    if (unstable_bytes.size() > 1) {
        auto it = unstable_bytes.rbegin();
        while (it != unstable_bytes.rend() && (*it & 0xC0) == 0x80) ++it;
        size_t last_char_size = std::distance(it, unstable_bytes.rend());

        if (unstable_bytes.size() - last_char_size > 0 && std::isspace(*(unstable_bytes.end() - last_char_size))) {
            std::vector<Rank> reencoded;
            auto first_part = byte_pair_encode(ByteString(unstable_bytes.begin(), unstable_bytes.end() - last_char_size));
            auto second_part = byte_pair_encode(ByteString(unstable_bytes.end() - last_char_size, unstable_bytes.end()));
            reencoded.insert(reencoded.end(), first_part.begin(), first_part.end());
            reencoded.insert(reencoded.end(), second_part.begin(), second_part.end());
            completions.insert(reencoded);
        }
    }

    return {tokens, completions};
}

Tokenizer::ByteString Tokenizer::decode(const std::vector<Rank>& tokens) const {
    ByteString result;
    for (const auto& token : tokens) {
        auto it = decoder_.find(token);
        if (it != decoder_.end()) {
            result += it->second;
        } else {
            auto special_it = special_tokens_decoder_.find(token);
            if (special_it != special_tokens_decoder_.end()) {
                result += special_it->second;
            }
        }
    }
    return result;
}

std::optional<Tokenizer::Rank> Tokenizer::encode_single_token(const ByteString& piece) const {
    auto it = encoder_.find(piece);
    if (it != encoder_.end()) {
        return it->second;
    }
    auto special_it = special_tokens_encoder_.find(piece);
    if (special_it != special_tokens_encoder_.end()) {
        return special_it->second;
    }
    return std::nullopt;
}

std::vector<Tokenizer::Rank> Tokenizer::encode_single_piece(const ByteString& piece) const {
    auto it = encoder_.find(piece);
    if (it != encoder_.end()) {
        return {it->second};
    }
    return byte_pair_encode(piece);
}

std::optional<Tokenizer::ByteString> Tokenizer::decode_single_token_bytes(Rank token) const {
    auto it = decoder_.find(token);
    if (it != decoder_.end()) {
        return it->second;
    }
    auto special_it = special_tokens_decoder_.find(token);
    if (special_it != special_tokens_decoder_.end()) {
        return special_it->second;
    }
    return std::nullopt;
}

std::vector<Tokenizer::ByteString> Tokenizer::token_byte_values() const {
    return sorted_token_bytes_;
}

std::vector<Tokenizer::ByteString> Tokenizer::regex_split(const std::string& text) const {
    std::vector<ByteString> result;
    
    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString utext = icu::UnicodeString::fromUTF8(text);
    std::unique_ptr<icu::RegexMatcher> matcher(regex_pattern_->matcher(utext, status));

    if (U_FAILURE(status)) {
        // Log error or throw exception
        throw std::runtime_error("Failed to create regex matcher: " + std::string(u_errorName(status)));
    }

    while (matcher->find()) {
        icu::UnicodeString match = matcher->group(status);
        std::string utf8Match;
        match.toUTF8String(utf8Match);
        result.push_back(utf8Match);
    }

    return result;
}

std::vector<std::pair<size_t, Tokenizer::Rank>> Tokenizer::byte_pair_merge(const ByteString& piece) const {
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
    if (piece.size() <= 1) {
        throw std::invalid_argument("Input piece must be longer than 1 byte");
    }
    
    auto merged = byte_pair_merge(piece);
    std::vector<Rank> result;
    result.reserve(merged.size() - 1);

    for (size_t i = 0; i < merged.size() - 1; ++i) {
        ByteString token = piece.substr(merged[i].first, merged[i + 1].first - merged[i].first);
        auto it = encoder_.find(token);
        if (it == encoder_.end()) {
            throw std::runtime_error("Token not found in encoder: " + token);
        }
        result.push_back(it->second);
    }

    return result;
}

std::vector<Tokenizer::ByteString> Tokenizer::byte_pair_split(const ByteString& piece) const {
    if (piece.size() <= 1) {
        throw std::invalid_argument("Input piece must be longer than 1 byte");
    }
    
    auto merged = byte_pair_merge(piece);
    std::vector<ByteString> result;
    result.reserve(merged.size() - 1);

    for (size_t i = 0; i < merged.size() - 1; ++i) {
        result.push_back(piece.substr(merged[i].first, merged[i + 1].first - merged[i].first));
    }

    return result;
}