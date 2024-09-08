#include "tokenizer.hpp"
#include <iostream>

int main() {
    Tokenizer tokenizer;

    // Example text to tokenize
    std::string text = "Hello, world! This is a test of my the tokenizer.";

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

    return 0;
}