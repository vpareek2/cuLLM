#include "tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>

Tokenizer::Tokenizer(const std::string& bpe_file) {
    
}

Tokenizer::~Tokenizer() {
    
}

std::vector<Tokenizer::Rank> Tokenizer::encode(const std::string& text) const {
    return {};
}

std::string Tokenizer::decode(const std::vector<Rank>& tokens) const {
    return "";
}

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

