/**
 * This file contains the kernel for the grouped query attention mechanism as used in the Llama 3.1 model architecture.
 * 
 * Grouped Query Attention (GQA) is an optimization of multi-head attention that reduces computational costs
 * by sharing key and value projections across multiple query heads. This allows for efficient scaling of
 * attention mechanisms in models while maintaining quality.
 */

#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "common.cuh"
#include "config.cuh"

__global__ 
void grouped_query_attention_kernel(int pos, int seq_len, float *sq, float *satt, float *sxb, float *key_cache,
                                    float *value_cache, int kv_dim, int num_kv_heads, int head_size, int loff);

void grouped_query_attention(int pos, Config *p, RunState *s, int kv_dim, int num_kv_heads, int head_size, int loff);

#endif // ATTENTION_CUH
