// This file contains the kernel implementation for the grouped query attention mechanism as used in the Llama 3.1 model architecture.
#include "attention.cuh"


/**
 * This kernel performs the grouped query attention mechanism.
 * 
 * @param pos The current position in the sequence.
 * @param seq_len The length of the sequence.
 * @param sq The query vectors for all heads.
 * @param satt The attention scores for all heads.
 * @param sxb The output of the attention mechanism for all heads.
 * @param key_cache The key cache for all heads.
 * @param value_cache The value cache for all heads.
 * @param kv_dim The dimension of the key and value vectors.
 * @param num_kv_heads The number of key and value heads.
 * @param head_size The size of each head.
 * @param loff The offset in the key and value caches.
*/
__global__ 
void grouped_query_attention_kernel(int pos, int seq_len, float *sq, float *satt, float *sxb, float *key_cache,
                                    float *value_cache, int kv_dim, int num_kv_heads, int head_size, int loff) {
    // Calculate the head index and the corresponding key-value head index
    int h = blockIdx.x;
    int kv_head = h / (gridDim.x / num_kv_heads);

    // Get pointers to the query and attention vectors for this head
    float *q = sq + h * head_size;
    float *att = satt + h * seq_len;

    // Step 1: Compute attention scores
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // Get the key vector for this position and key-value head
        float *k = key_cache + loff + t * kv_dim + kv_head * head_size;
        float score = 0.0f;
        // Compute dot product between query and key
        for (int i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        // Scale the score by the square root of head size (as per attention mechanism)
        score /= sqrtf(head_size);
        att[t] = score;
    }
    __syncthreads();

    // Step 2: Apply softmax to get attention weights
    softmax_gpu(att, pos + 1);
    __syncthreads();

    // Step 3: Compute weighted sum of values
    float *xb = sxb + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            // Get the value vector for this position and key-value head
            float *v = value_cache + loff + t * kv_dim + kv_head * head_size;
            float a = att[t];
            // Accumulate weighted value
            val += a * v[i];
        }
        // Store the result in the output vector
        xb[i] = val;
    }
}


/**
 * This function launches the grouped query attention kernel.
 * 
 * Grouped Query Attention (GQA) is an optimization where multiple query heads share the same key and value heads,
 * reducing computational and memory costs while maintaining model quality.
 * 
 * @param pos The current position in the sequence.
 * @param p The configuration parameters.
 * @param s The run state.
 * @param kv_dim The dimension of the key and value vectors.
 * @param num_kv_heads The number of key and value heads.
 * @param head_size The size of each head.
 * @param loff The offset in the key and value caches.
*/
void grouped_query_attention(int pos, Config *p, RunState *s, int kv_dim, int num_kv_heads, int head_size, int loff) {
    // Launch the kernel with one block per attention head and a large number of threads per block
    grouped_query_attention_kernel<<<p->n_heads, num_threads_large>>>(pos, p->max_seq_len, s->q, s->att, s->xb,
                                                                      s->key_cache, s->value_cache, kv_dim, num_kv_heads,
                                                                      head_size, loff);
}