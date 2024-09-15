#include "tokenizer.hpp"
#include <iostream>

int main() {
    Tokenizer tokenizer("vocab/o200k_base.tiktoken");
    
    try {
        auto encoded = tokenizer.encode("Hello, World!");
        std::cout << "Encoded: ";
        for (auto token : encoded) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // Example text to tokenize
    std::string text = "This is the GPT-4o ToKENIzeR. 32714. :q'[]p`";

    // Encode the text
    std::vector<Tokenizer::Rank> tokens = tokenizer.encode(text);

    // Print the tokens
    std::cout << "Encoded tokens:" << std::endl;
    for (const auto& token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // Decode the tokens back to text
    std::string decoded_text = tokenizer.decode(tokens);

    // Print the decoded text
    std::cout << "Decoded text: " << decoded_text << std::endl;

    // Example usage of encode_single_token and decode_single_token_bytes
    std::string single_piece = "Hello";
    std::optional<Tokenizer::Rank> single_token = tokenizer.encode_single_token(single_piece);
    if (single_token) {
        std::cout << "Single token: " << *single_token << std::endl;
    } else {
        std::cout << "Failed to encode single token" << std::endl;
    }

    std::optional<Tokenizer::ByteString> decoded_single_token = tokenizer.decode_single_token_bytes(*single_token);
    if (decoded_single_token) {
        std::cout << "Decoded single token: " << *decoded_single_token << std::endl;
    } else {
        std::cout << "Failed to decode single token" << std::endl;
    }

    return 0;
}