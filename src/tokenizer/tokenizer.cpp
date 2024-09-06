#include "tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>

Tokenizer::Tokenizer() {
    // Initialize the regex pattern
    regex_pattern_ = std::regex(REGEX_PATTERN, std::regex::optimize);

    // Read and parse the BPE file
    std::ifstream file("../vocab/o200k_base.tiktoken");
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open BPE file: ../vocab/o200k_base.tiktoken");
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
    std::vector<std::string> result;
    std::sregex_iterator it(text.begin(), text.end(), regex_pattern_);
    std::sregex_iterator end;

    while (it != end) {
        result.push_back(it->str());
        ++it;
    }

    return result;
}

std::vector<Tokenizer::Rank> Tokenizer::bpe_encode(const std::string& token) const {
    ByteVector piece(token.begin(), token.end());
    return byte_pair_merge(piece, encoder_);
}

std::vector<Tokenizer::Rank> Tokenizer::byte_pair_merge(const ByteVector& piece,
                                                        const std::unordered_map<ByteVector, Rank>& ranks) const {
    std::vector<Rank> ids;
    ids.reserve(piece.size());

    // Initialize ids with ranks of individual bytes
    for (uint8_t byte : piece) {
        ByteVector single_byte = {byte};
        ids.push_back(ranks.at(single_byte));
    }

    bool changes = true;
    while (changes) {
        changes = false;
        for (size_t i = 0; i < ids.size() - 1; ++i) {
            ByteVector bigram = {static_cast<uint8_t>(ids[i]), static_cast<uint8_t>(ids[i + 1])};
            auto it = ranks.find(bigram);
            if (it != ranks.end()) {
                Rank new_id = it->second;
                ids[i] = new_id;
                ids.erase(ids.begin() + i + 1);
                changes = true;
                break;
            }
        }
    }

    return ids;
}