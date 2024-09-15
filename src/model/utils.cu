#include "utils.cuh"

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