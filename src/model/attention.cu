#include "attention.cuh"

__global__ 
void multi_head_attention_kernel(int pos, int seq_len, float *sq, float *satt, float *sxb, float *key_cache,
                                 float *value_cache, int kv_dim, int kv_mul, int head_size, int loff) {
    int h = blockIdx.x;
    // get the query vector for this head
    float *q = sq + h * head_size;
    // attention scores for this head
    float *att = satt + h * seq_len;
    // iterate over all timesteps, including the current one
    // In CUDA, each thread does a small portion of the calc
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
    }
    // above was this threads portion of the iteration. wait for all threads to finish
    __syncthreads();

    // softmax the scores to get attention weights, from 0...pos inclusively
    softmax_gpu(att, pos + 1);
    __syncthreads();

    // weighted sum of the values, store back into xb
    float *xb = sxb + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[t];
            val += a * v[i];
        }
        xb[i] = val;
    }
}

void multi_head_attention(int pos, Config *p, RunState *s, int kv_dim, int kv_mul, int head_size, int loff) {
    multi_head_attention_kernel<<<p->n_heads, num_threads_large>>>(pos, p->max_seq_len, s->q, s->att, s->xb,
                                                                   s->key_cache, s->value_cache, kv_dim, kv_mul,
                                                                   head_size, loff);
}
