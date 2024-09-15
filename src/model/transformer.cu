#include "transformer.cuh"

// Implementation of functions declared in transformer.cuh

void build_transformer(Transformer *t, char *checkpoint_path, int num_kv_heads) {
    // read in the Config
    FILE *file = fopen(checkpoint_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(EXIT_FAILURE); }
    if (fread(&t->config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    fclose(file);

    // set the number of key-value heads
    t->num_kv_heads = num_kv_heads;

    // memory map the Transformer weights into the data pointer
    t->fd = open(checkpoint_path, O_RDONLY);
    if (t->fd == -1) { fprintf(stderr, "open failed\n"); exit(EXIT_FAILURE); }
    t->file_size = lseek(t->fd, 0, SEEK_END);
    t->data = (float*)mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = t->data + sizeof(Config) / sizeof(float);

    // set all the pointers in the TransformerWeights struct
    memory_map_weights(&t->weights, &t->config, weights_ptr, 1);

    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
    // close the memory mapping
    if (t->data != NULL) {
        munmap(t->data, t->file_size);
        close(t->fd);
    }
    // free the RunState buffers
    free_run_state(&t->state);
}

float* forward(Transformer *transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * transformer->num_kv_heads) / p->n_heads;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding + token * dim;
    CUDA_CHECK(cudaMemcpy(x, content_row, dim * sizeof(float), cudaMemcpyHostToDevice));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        RoPe_rotation(pos, s, dim, kv_dim, head_size);

        // multihead attention. use grouped query attention
        int loff = l * p->max_seq_len * kv_dim;
        grouped_query_attention(pos, p, s, kv_dim, transformer->num_kv_heads, head_size, loff);

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // SwiGLU
        int ffn_dim = p->hidden_dim * 2; // Typically, SwiGLU uses 2/3 * 4 * dim for hidden_dim
        swiglu(s, dim, ffn_dim); 

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*p->hidden_dim, p->hidden_dim, dim);

        // residual connection
        accum(x, s->xb, dim);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);
    CUDA_CHECK(cudaMemcpy(s->logits, s->logits_gpu, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return s->logits;
}

void malloc_run_state(RunState *s, Config *p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
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
    // we calloc instead of malloc to keep valgrind happy
    s->logits = (float *) calloc(p->vocab_size, sizeof(float));

    // ensure all cudaMallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits) {
        fprintf(stderr, "cudaMalloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState *s) {
    CUDA_CHECK(cudaFree(s->x));
    CUDA_CHECK(cudaFree(s->xb));
    CUDA_CHECK(cudaFree(s->xb2));
    CUDA_CHECK(cudaFree(s->hb));
    CUDA_CHECK(cudaFree(s->hb2));
    CUDA_CHECK(cudaFree(s->q));
    CUDA_CHECK(cudaFree(s->att));
    CUDA_CHECK(cudaFree(s->logits_gpu));
    free(s->logits);
    CUDA_CHECK(cudaFree(s->key_cache));
    CUDA_CHECK(cudaFree(s->value_cache));
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding = ptr;
    ptr += p->vocab_size * p->dim;
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
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    // Remove this line:
    // w->w3 = ptr;
    // ptr += n_layers * p->dim * p->hidden_dim;
    w->b1 = ptr;
    ptr += n_layers * p->hidden_dim;
    w->b2 = ptr;
    ptr += n_layers * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->max_seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->max_seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding : ptr;
}
