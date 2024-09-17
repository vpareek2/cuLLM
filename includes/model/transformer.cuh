/**
 * This file contains the definition of the Transformer architecture and related functions.
 */

#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include "config.cuh"
#include "common.cuh"

// Transformer weights structure
typedef struct {
    float *token_embedding;
    float *rms_att_weight;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *rms_ffn_weight;
    float *w1;
    float *w2;
    float *b1;
    float *b2;
    float *rms_final_weight;
    float *wcls;
} TransformerWeights;

// Combine key and value caches into a single structure
typedef struct {
    float *key;
    float *value;
} KVCache;

// Update RunState structure
typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *att;
    float *logits_gpu;
    float *logits;
    KVCache *kv_cache;
} RunState;

// Transformer structure
typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float *data;
    ssize_t file_size;
    int num_kv_heads;
} Transformer;

void build_transformer(Transformer *t, char *checkpoint_path, int num_kv_heads);
void free_transformer(Transformer *t);
float* forward(Transformer *transformer, int token, int pos);

// Helper function declarations
void malloc_run_state(RunState *s, Config *p);
void free_run_state(RunState *s);
void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights);

#endif // TRANSFORMER_CUH
