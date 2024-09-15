#include "common.cuh"
#include "config.cuh"
#include "transformer.cuh"
#include "utils.cuh"
#include "tokenizer.hpp"
#include "attention.cuh"



// ----------------------------------------------------------------------------
// generation loop
// ----------------------------------------------------------------------------
void generate(Transformer *transformer, Tokenizer *tokenizer, const char *prompt, int max_new_tokens) {
    const char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    std::vector<Tokenizer::Rank> prompt_tokens = tokenizer->encode(prompt);
    int num_prompt_tokens = prompt_tokens.size();

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    Tokenizer::Rank next;        // will store the next token in the sequence
    Tokenizer::Rank token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < max_new_tokens - 1) {
        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            next = sample_argmax(logits, transformer->config.vocab_size);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        std::string piece = tokenizer->decode({token, next});
        safe_printf(piece.c_str()); // same as printf("%s", piece.c_str()), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (Token count is assumed to be pos+1 because BOS token must be included)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "Token count: %d, elapsed: %fs, %d tokens/s\n",
                pos + 1, (float) (end - start) / 1000, (int) ((pos - 1) / (double) (end - start) * 1000));
    }

    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // default parameters
    char *checkpoint_path = (char *) "stories15M.bin";  // e.g. out/model.bin
    char *tokenizer_path = (char *) "tokenizer.bin";
    int max_new_tokens = 50;                            // number of max_new_tokens to run for
    char *prompt = (char *) "I have a dream";           // poor man's prompt string

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { prompt = argv[1]; }

    // build the Transformer via the model .bin file
    Transformer transformer;
    int num_kv_heads = 8; // Set this to the appropriate number of KV heads for your model
    build_transformer(&transformer, checkpoint_path, num_kv_heads);

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer(tokenizer_path);

    create_cublas_handle();

    // run!
    generate(&transformer, &tokenizer, prompt, max_new_tokens);

    // memory and file handles cleanup
    // free_tokenizer(&tokenizer); // Remove this line as Tokenizer is now handled by C++ destructor
    free_transformer(&transformer);
    destroy_cublas_handle();
    return 0;
}