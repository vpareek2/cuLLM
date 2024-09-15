#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "common.cuh"
#include "config.cuh"

// Function declarations
__global__ 
void grouped_query_attention_kernel(int pos, int seq_len, float *sq, float *satt, float *sxb, float *key_cache,
                                    float *value_cache, int kv_dim, int num_kv_heads, int head_size, int loff);

void grouped_query_attention(int pos, Config *p, RunState *s, int kv_dim, int num_kv_heads, int head_size, int loff);

#endif // ATTENTION_CUH
