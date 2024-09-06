/**
 * This file contains the implementation of the tokenizer class.
 * 
 * Tiktoken-style GPT-4o regex (o200k_base) + BPE tokenizer.
 */
#include "includes/tokenizer.hpp"

Tokenizer::Tokenizer() {
    // Initialize the PCRE2 regex pattern
    int errorcode;
    PCRE2_SIZE erroroffset;
    regex_pattern_ = pcre2_compile(
        (PCRE2_SPTR)REGEX_PATTERN.c_str(),
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP,
        &errorcode,
        &erroroffset,
        nullptr
    );
    if (regex_pattern_ == nullptr) {
        throw std::runtime_error("PCRE2 compilation failed");
    }

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

        // Change: Use token directly as ByteString
        encoder_[token] = rank;
        decoder_[rank] = token;
    }

    // Add special tokens to the encoder and decoder
    encoder_[ENDOFTEXT] = ENDOFTEXT_TOKEN;
    decoder_[ENDOFTEXT_TOKEN] = ENDOFTEXT;
    encoder_[ENDOFPROMPT] = ENDOFPROMPT_TOKEN;
    decoder_[ENDOFPROMPT_TOKEN] = ENDOFPROMPT;
}

Tokenizer::~Tokenizer() {
    if (regex_pattern_ != nullptr) {
        pcre2_code_free(regex_pattern_);
    }
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
        const ByteString& bytes = decoder_.at(rank);
        decoded_text += bytes; // Change: Directly append ByteString
    }
    
    return decoded_text;
}

/*
 * Helper Functions
 */

std::vector<std::string> Tokenizer::regex_split(const std::string& text) const {
    std::vector<std::string> result;
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(regex_pattern_, nullptr);
    
    int rc;
    PCRE2_SIZE offset = 0;
    PCRE2_SIZE* ovector;

    while ((rc = pcre2_match(regex_pattern_, (PCRE2_SPTR)text.c_str(), text.length(), offset, 0, match_data, nullptr)) > 0) {
        ovector = pcre2_get_ovector_pointer(match_data);
        result.push_back(text.substr(ovector[0], ovector[1] - ovector[0]));
        offset = ovector[1];
    }

    pcre2_match_data_free(match_data);
    return result;
}

std::vector<Tokenizer::Rank> Tokenizer::bpe_encode(const std::string& token) const {
    // Change: Use token directly as ByteString
    return byte_pair_merge(token, encoder_);
}

std::vector<Tokenizer::Rank> Tokenizer::byte_pair_merge(const ByteString& piece,
                                                        const std::unordered_map<ByteString, Rank>& ranks) const {
    std::vector<Rank> ids;
    ids.reserve(piece.size());

    // Initialize ids with ranks of individual bytes
    for (unsigned char byte : piece) {
        ByteString single_byte(1, byte);
        ids.push_back(ranks.at(single_byte));
    }

    bool changes = true;
    while (changes) {
        changes = false;
        for (size_t i = 0; i < ids.size() - 1; ++i) {
            // Change: Construct ByteString from two characters
            ByteString bigram;
            bigram += static_cast<char>(ids[i]);
            bigram += static_cast<char>(ids[i + 1]);
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