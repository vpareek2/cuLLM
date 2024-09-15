#include "common.cuh"
#include "config.cuh"
#include "transformer.cuh"
#include "utils.cuh"
#include "tokenizer.hpp"

/**
 * Generate text using a transformer model and tokenizer
 * @param transformer Pointer to the transformer model
 * @param tokenizer Pointer to the tokenizer
 * @param prompt The input prompt for text generation
 * @param max_toks The maximum number of tokens to generate
 */

void generate(Transformer *transformer, Tokenizer *tokenizer, const char *prompt, int max_toks) {
    // Handle null prompt by using an empty string
    const char *empty_prompt = "";
    if (prompt == NULL) {
        prompt = empty_prompt;
    }

    // Encode the prompt into tokens
    std::vector<Tokenizer::Rank> prompt_tokens = tokenizer->encode(prompt);
    int num_prompt_tokens = prompt_tokens.size();

    // Ensure we have at least one token to work with
    if (num_prompt_tokens < 1) {
        exit(EXIT_FAILURE);
    }

    long start = 0;
    Tokenizer::Rank next;
    Tokenizer::Rank token = prompt_tokens[0];
    int pos = 0;

    // Main generation loop
    while (pos < max_toks - 1) {
        // Get logits from the transformer
        float *logits = forward(transformer, token, pos);

        // Determine the next token
        if (pos < num_prompt_tokens - 1) {
            // If we're still processing the prompt, use the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // Otherwise, sample the next token from the model's output
            next = sample_argmax(logits, transformer->config.vocab_size);
        }

        pos++;

        // Check for end of sequence token
        if (next == 1) { break; }

        // Decode and print the generated piece
        std::string piece = tokenizer->decode({token, next});
        safe_printf(piece.c_str());
        fflush(stdout);

        token = next;

        // Start timing after the first token is generated
        if (start == 0) { start = time_in_ms(); }
    }

    printf("\n");

    // Print generation statistics if more than one token was generated
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "Token count: %d, elapsed: %fs, %d tokens/s\n",
                pos + 1, (float) (end - start) / 1000, (int) ((pos - 1) / (double) (end - start) * 1000));
    }

    // Clean up
    free(prompt_tokens);
}

/**
 * Main function to initialize and run the text generation
 * @param argc Number of command-line arguments
 * @param argv Command-line arguments (including the path to the model checkpoint and tokenizer)
 * @return 0 on success, non-zero on failure
 */

int main(int argc, char *argv[]) {
    // Default paths for the model checkpoint and tokenizer
    char *checkpoint_path = (char *) "stories15M.bin";
    char *tokenizer_path = (char *) "vocab/o200k_base.tiktoken";

    // Maximum number of tokens to generate
    int max_toks = 50;

    // Default prompt is empty, but can be provided as a command-line argument
    char *prompt = (char *) "";
    if (argc >= 2) { prompt = argv[1]; }

    // Initialize the transformer
    Transformer transformer;
    int num_kv_heads = 8; // as per the llama3.1 paper
    build_transformer(&transformer, checkpoint_path, num_kv_heads);

    // Initialize the tokenizer
    Tokenizer tokenizer(tokenizer_path);

    // Set up cuBLAS resources
    create_cublas_handle();

    // Generate text based on the prompt
    generate(&transformer, &tokenizer, prompt, max_toks);

    // Clean up resources
    free_transformer(&transformer);
    destroy_cublas_handle();

    return 0;
}