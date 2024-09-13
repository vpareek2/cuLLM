

// Utility functions implementation
__host__ __device__ inline int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

__host__ 
long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a one way: greedy argmax
// ----------------------------------------------------------------------------
int sample_argmax(float *probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

// ----------------------------------------------------------------------------
// generation loop
// ----------------------------------------------------------------------------
void generate(Transformer *transformer, Tokenizer *tokenizer, char *prompt, int max_new_tokens) {
    char *empty_prompt = (char *) "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *) malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    // TODO: pretty dirty monkey patch for 'I have a dream' prompt.
    if (prompt_tokens[1] == 306) prompt_tokens[1] = 76;
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
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
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
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
    build_transformer(&transformer, checkpoint_path);
    if (max_new_tokens > transformer.config.max_seq_len)
        max_new_tokens = transformer.config.max_seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    create_cublas_handle();

    // run!
    generate(&transformer, &tokenizer, prompt, max_new_tokens);

    // memory and file handles cleanup
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    destroy_cublas_handle();
    return 0;
}