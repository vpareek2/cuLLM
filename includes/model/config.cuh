#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>

// Configuration structure for the transformer model
typedef struct {
    int dim;            // D: Dimension of the model
    int hidden_dim;     // DD: Hidden dimension
    int n_layers;       // NL: Number of layers
    int n_heads;        // QHN: Number of query heads
    int n_kv_heads;     // KVHN: Number of key/value heads
    int vocab_size;     // VS: Vocabulary size
    int max_seq_len;    // M: Maximum sequence length
} Config;

// Constants
#define MAX_SEQ_LEN 1024
#define MAX_BATCH_SIZE 32

// CUDA error checking macro
#define CUDA_CHECK(val) { \
    if (val != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", val, cudaGetErrorString(val), __FILE__, __LINE__); \
        fflush(stderr); \
        exit(val); \
    } \
}

// Global cuBLAS handle
extern cublasHandle_t g_cublas_handle;

// Function declarations
void create_cublas_handle();
void destroy_cublas_handle();

// Utility functions
__host__ 
long time_in_ms();

__host__ 
__device__ 
void safe_printf(const char* format, ...);

// Memory allocation helpers
__host__ 
void* malloc_device(size_t size);

__host__ 
void* malloc_host(size_t size);

__host__ 
void free_device(void* ptr);

__host__ 
void free_host(void* ptr);

// Other utility macros or inline functions
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Thread configuration
const int num_threads_large = 1024;
const int num_threads_small = 64;

// Utility function for dividing work among threads
__host__ __device__ inline int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

#endif // CONFIG_CUH