/**
 * This file contains the implementation of the Transformer architecture.
 */

#include "transformer.cuh"

/**
 * @brief Builds the Transformer architecture.
 * @param t The Transformer structure to be built.
 * @param checkpoint_path The path to the checkpoint file.
 * @param num_kv_heads The number of key-value heads, for GQA.
 */ 
void build_transformer(Transformer *t, char *checkpoint_path, int num_kv_heads) {
    // Open the checkpoint file
    FILE *file = fopen(checkpoint_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(EXIT_FAILURE); }
    
    // Read the configuration from the file
    if (fread(&t->config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    fclose(file);

    // Set the number of key-value heads
    t->num_kv_heads = num_kv_heads;

    // Memory map the checkpoint file
    t->fd = open(checkpoint_path, O_RDONLY);
    if (t->fd == -1) { fprintf(stderr, "open failed\n"); exit(EXIT_FAILURE); }
    t->file_size = lseek(t->fd, 0, SEEK_END);
    t->data = (float*)mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(EXIT_FAILURE); }
    
    // Point to the start of the weights in the mapped memory
    float* weights_ptr = t->data + sizeof(Config) / sizeof(float);

    // Map the weights to the appropriate structures
    memory_map_weights(&t->weights, &t->config, weights_ptr, 1);

    // Allocate memory for the run state
    malloc_run_state(&t->state, &t->config);
}

/**
 * @brief Frees the Transformer architecture.
 * @param t The Transformer structure to be freed.
 */
void free_transformer(Transformer *t) {
    // Unmap the memory-mapped file if it exists
    if (t->data != NULL) {
        munmap(t->data, t->file_size);
        close(t->fd);
    }
    // Free the run state
    free_run_state(&t->state);
}

/**
 * @brief Performs the forward pass of the Transformer architecture.
 * @param transformer The Transformer structure.
 * @param token The token to be processed.
 * @param pos The position of the token in the sequence.
 * @return The logits of the token.
 */ 
float* forward(Transformer *transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * transformer->num_kv_heads) / p->n_heads;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // Copy the token embedding to GPU memory
    float* content_row = w->token_embedding + token * dim;
    CUDA_CHECK(cudaMemcpy(x, content_row, dim * sizeof(float), cudaMemcpyHostToDevice));

    // Process each layer of the transformer
    for(int l = 0; l < p->n_layers; l++) {
        // Layer normalization before self-attention
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // Compute query, key, and value
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // Apply rotary positional embedding
        RoPe_rotation(pos, s, dim, kv_dim, head_size);

        // Compute attention
        int loff = l * p->max_seq_len * kv_dim;
        grouped_query_attention(pos, p, s, kv_dim, transformer->num_kv_heads, head_size, loff);

        // Compute output projection
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // Residual connection
        accum(x, s->xb2, dim);

        // Layer normalization before feed-forward network
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Feed-forward network
        int ffn_dim = p->hidden_dim * 2; 
        swiglu(s, dim, ffn_dim); 

        // Output projection of feed-forward network
        matmul(s->xb, s->hb, w->w2 + l*dim*p->hidden_dim, p->hidden_dim, dim);

        // Residual connection
        accum(x, s->xb, dim);
    }

    // Final layer normalization
    rmsnorm(x, x, w->rms_final_weight, dim);

    // Compute logits
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);
    CUDA_CHECK(cudaMemcpy(s->logits, s->logits_gpu, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return s->logits;
}

/**
 * @brief Allocates the RunState buffers.
 * @param s The RunState structure to be allocated.
 * @param p The Config structure.
 */ 
void malloc_run_state(RunState *s, Config *p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    // Allocate memory for various tensors used in the forward pass
    CUDA_CHECK(cudaMalloc((void **) &s->x, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->xb, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->xb2, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->hb, p->hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->hb2, p->hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->q, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->key_cache, p->n_layers * p->max_seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->value_cache, p->n_layers * p->max_seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->att, p->n_heads * p->max_seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->logits_gpu, p->vocab_size * sizeof(float)));
    s->logits = (float *) calloc(p->vocab_size, sizeof(float));

    // Check if all allocations were successful
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits) {
        fprintf(stderr, "cudaMalloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Frees the RunState buffers.
 * @param s The RunState structure to be freed.
 */
void free_run_state(RunState *s) {
    // Free all allocated GPU memory
    CUDA_CHECK(cudaFree(s->x));
    CUDA_CHECK(cudaFree(s->xb));
    CUDA_CHECK(cudaFree(s->xb2));
    CUDA_CHECK(cudaFree(s->hb));
    CUDA_CHECK(cudaFree(s->hb2));
    CUDA_CHECK(cudaFree(s->q));
    CUDA_CHECK(cudaFree(s->att));
    CUDA_CHECK(cudaFree(s->logits_gpu));
    CUDA_CHECK(cudaFree(s->key_cache));
    CUDA_CHECK(cudaFree(s->value_cache));
    
    // Free CPU memory
    free(s->logits);
}

/**
 * @brief Maps the weights of the Transformer architecture.
 * @param w The TransformerWeights structure to be mapped.
 * @param p The Config structure.
 * @param ptr The pointer to the weights.
 * @param shared_weights Whether the weights are shared.
 */
void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    
    // Map token embeddings
    w->token_embedding = ptr;
    ptr += p->vocab_size * p->dim;
    
    // Map attention weights
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    
    // Map feed-forward network weights
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->b1 = ptr;
    ptr += n_layers * p->hidden_dim;
    w->b2 = ptr;
    ptr += n_layers * p->hidden_dim;
    
    // Map final layer normalization weights
    w->rms_final_weight = ptr;
    ptr += p->dim;
    
    // Skip rope frequencies (if present)
    ptr += p->max_seq_len * head_size / 2; 
    ptr += p->max_seq_len * head_size / 2;
    
    // Map classifier weights (may be shared with token embeddings)
    w->wcls = shared_weights ? w->token_embedding : ptr;
}
