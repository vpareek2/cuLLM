#include "tokenizer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

int main() {
    // Initialize the tokenizer with the encoder file
    Tokenizer tokenizer("path/to/encoder_file.txt");

    // Example text to tokenize
    std::string text = "Hello, world! This is a test of the tokenizer.";

    // Encode the text
    std::unordered_set<std::string> allowed_special;
    std::vector<Tokenizer::Rank> tokens = tokenizer.encode(text, allowed_special);

    // Print the tokens
    std::cout << "Encoded tokens:" << std::endl;
    for (const auto& token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // Decode the tokens back to text
    Tokenizer::ByteString decoded_text = tokenizer.decode(tokens);

    // Print the decoded text
    std::cout << "Decoded text: " << decoded_text << std::endl;

    // Example of single token operations
    Tokenizer::ByteString single_piece = "example";
    Tokenizer::Rank single_token = tokenizer.encode_single_token(single_piece);
    std::cout << "Single token for 'example': " << single_token << std::endl;

    std::vector<Tokenizer::Rank> single_piece_tokens = tokenizer.encode_single_piece(single_piece);
    std::cout << "Tokens for 'example': ";
    for (const auto& token : single_piece_tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    Tokenizer::ByteString decoded_single_token = tokenizer.decode_single_token_bytes(single_token);
    std::cout << "Decoded single token: " << decoded_single_token << std::endl;

    // Get all token byte values
    std::vector<Tokenizer::ByteString> all_tokens = tokenizer.token_byte_values();
    std::cout << "Total number of tokens: " << all_tokens.size() << std::endl;

    return 0;
}