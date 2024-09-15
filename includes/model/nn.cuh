/**
 * This file contains the neural network operations used in the transformer model.
 */

#ifndef NN_CUH
#define NN_CUH

#include "config.cuh"
#include "common.cuh"

// Constants
extern const int num_threads_large;

// cuBLAS handle
extern cublasHandle_t g_cublas_handle;

__global__ 
void rmsnorm_kernel(float *o, float *x, float *weight, int size, int elementsPerThread);
void rmsnorm(float *o, float *x, float *weight, int size);

__device__ 
void softmax_gpu(float *__restrict__ x, int size);

void matmul(float *xout, float *x, float *w, int n, int d);

__global__ 
void RoPE_rotation_kernel(int pos, float *sq, float *sk, int kv_dim, int head_size);
void RoPE_rotation(int pos, RunState *s, int dim, int kv_dim, int head_size);

__global__ 
void swiglu_kernel(float *out, float *x, float *w1, float *w2, float *b1, float *b2, int hidden_dim, int ffn_dim);
void swiglu(RunState *s, int hidden_dim, int ffn_dim);

__global__ 
void accum_kernel(float *a, float *b, int size);
void accum(float *a, float *b, int size);

// cuBLAS handle management
void create_cublas_handle();
void destroy_cublas_handle();

#endif // NN_CUH
