#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "config.cuh"

// Function declarations
__global__ 
void multi_head_attention_kernel(int pos, int seq_len, float *sq, float *satt, float *sxb, float *key_cache,
                                            float *value_cache, int kv_dim, int kv_mul, int head_size, int loff);

void multi_head_attention(int pos, Config *p, RunState *s, int kv_dim, int kv_mul, int head_size, int loff);


#endif // ATTENTION_CUH

