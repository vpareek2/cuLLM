#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include "config.cuh"

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
    float *w3;
    float *rms_final_weight;
    float *wcls;
} TransformerWeights;

// Run state structure
typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits_gpu;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

// Transformer structure
typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float *data;
    ssize_t file_size;
} Transformer;

// Function declarations
void build_transformer(Transformer *t, char *checkpoint_path);
void free_transformer(Transformer *t);
float* forward(Transformer *transformer, int token, int pos);

// Helper function declarations
void malloc_run_state(RunState *s, Config *p);
void free_run_state(RunState *s);
void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights);

#endif // TRANSFORMER_CUH