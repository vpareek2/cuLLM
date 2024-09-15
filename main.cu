#include "common.cuh"
#include "config.cuh"
#include "transformer.cuh"
#include "utils.cuh"
#include "tokenizer.hpp"

void generate(Transformer *transformer, Tokenizer *tokenizer, const char *prompt, int max_new_tokens) {
    const char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }
    std::vector<Tokenizer::Rank> prompt_tokens = tokenizer->encode(prompt);
    int num_prompt_tokens = prompt_tokens.size();
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }
    long start = 0;
    Tokenizer::Rank next;
    Tokenizer::Rank token = prompt_tokens[0];
    int pos = 0;
    while (pos < max_new_tokens - 1) {
        float *logits = forward(transformer, token, pos);
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample_argmax(logits, transformer->config.vocab_size);
        }
        pos++;
        if (next == 1) { break; }
        std::string piece = tokenizer->decode({token, next});
        safe_printf(piece.c_str());
        fflush(stdout);
        token = next;
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "Token count: %d, elapsed: %fs, %d tokens/s\n",
                pos + 1, (float) (end - start) / 1000, (int) ((pos - 1) / (double) (end - start) * 1000));
    }
    free(prompt_tokens);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = (char *) "stories15M.bin";
    char *tokenizer_path = (char *) "vocab/o200k_base.tiktoken";
    int max_new_tokens = 50;
    char *prompt = (char *) "";
    if (argc >= 2) { prompt = argv[1]; }
    Transformer transformer;
    int num_kv_heads = 8; // as per the llama3.1 paper
    build_transformer(&transformer, checkpoint_path, num_kv_heads);
    Tokenizer tokenizer(tokenizer_path);
    create_cublas_handle();
    generate(&transformer, &tokenizer, prompt, max_new_tokens);
    free_transformer(&transformer);
    destroy_cublas_handle();
    return 0;
}