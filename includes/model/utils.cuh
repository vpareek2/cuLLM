#ifndef UTILS_CUH
#define UTILS_CUH

#include "common.cuh"

// Utility functions declarations
__host__ __device__ inline int divUp(int a, int b);

__host__ long time_in_ms();

// Sampler function declaration
int sample_argmax(float *probabilities, int n);

#endif // UTILS_CUH