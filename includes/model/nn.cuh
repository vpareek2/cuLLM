#ifndef NN_CUH
#define NN_CUH

#include "config.cuh"
#include <cublas_v2.h>

// Constants
extern const int num_threads_large;

// cuBLAS handle
extern cublasHandle_t g_cublas_handle;


__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size, int elementsPerThread);
void rmsnorm(float *o, float *x, float *weight, int size);

__device__ void softmax_gpu(float *__restrict__ x, int size);

void matmul(float *xout, float *x, float *w, int n, int d);

__global__ void RoPE_rotation_kernel(int pos, float *sq, float *sk, int kv_dim, int head_size);
void RoPE_rotation(int pos, RunState *s, int dim, int kv_dim, int head_size);

__global__ void f_silu_elementwise_mul_w3_kernel(float *shb, float *shb2, int hidden_dim);
void f_silu_elementwise_mul_w3(RunState *s, int hidden_dim);

__global__ void accum_kernel(float *a, float *b, int size);
void accum(float *a, float *b, int size);

// cuBLAS handle management
void create_cublas_handle();
void destroy_cublas_handle();

#endif // NN_CUH
