#include "tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>

Tokenizer::Tokenizer(const std::string& bpe_file) {
    // Initialize the regex pattern
    regex_pattern_ = std::regex(REGEX_PATTERN, std::regex::optimize);

    // Read and parse the BPE file
    std::ifstream file(bpe_file);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open BPE file: " + bpe_file);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        Rank rank;

        if (!(iss >> token >> rank)) {
            continue;
        }

        ByteVector bytes(token.begin(), token.end());
        encoder_[bytes] = rank;
        decoder_[rank] = bytes;
    }

    // Add special tokens to the encoder and decoder
    for (const auto& [token, rank] : SPECIAL_TOKENS) {
        ByteVector bytes(token.begin(), token.end());
        encoder_[bytes] = rank;
        decoder_[rank] = bytes;
    }
}

Tokenizer::~Tokenizer() {
    // No dynamic allocations, so nothing to clean up
}

std::vector<Tokenizer::Rank> Tokenizer::encode(const std::string& text) const {
    std::vector<Rank> encoded_tokens;
    
    // Split the text using regex
    std::vector<std::string> split_tokens = regex_split(text);
    
    // Encode each split token
    for (const auto& token : split_tokens) {
        // Apply BPE encoding
        std::vector<Rank> bpe_encoded = bpe_encode(token);
        
        // Add encoded tokens to the result
        encoded_tokens.insert(encoded_tokens.end(), bpe_encoded.begin(), bpe_encoded.end());
    }
    
    return encoded_tokens;
}

std::string Tokenizer::decode(const std::vector<Rank>& tokens) const {
    std::string decoded_text;
    decoded_text.reserve(tokens.size() * 2);
    
    for (const auto& rank : tokens) {
        const ByteVector& bytes = decoder_.at(rank);
        decoded_text.append(bytes.begin(), bytes.end());
    }
    
    return decoded_text;
}

/*
 * Helper Functions
 */

std::vector<std::string> Tokenizer::regex_split(const std::string& text) const {
    return {};
}

std::vector<Tokenizer::Rank> Tokenizer::bpe_encode(const std::string& token) const {
    return {};
}

std::vector<Tokenizer::Rank> Tokenizer::byte_pair_merge(const ByteVector& piece,
                                                        const std::unordered_map<ByteVector, Rank>& ranks) {
    return {};
}