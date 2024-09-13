#include "config.cuh"
#include <time.h>
#include <stdarg.h>

// Global cuBLAS handle implementation
cublasHandle_t g_cublas_handle = nullptr;

void create_cublas_handle() {
    cublasStatus_t stat = cublasCreate(&g_cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
}

void destroy_cublas_handle() {
    cublasStatus_t stat = cublasDestroy(g_cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS destruction failed\n");
        exit(EXIT_FAILURE);
    }
}

// Utility functions implementation
__host__ 
long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

__host__ 
__device__ 
void safe_printf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    #ifdef __CUDA_ARCH__
        vprintf(format, args);
    #else
        vfprintf(stdout, format, args);
    #endif
    va_end(args);
}

// Memory allocation helpers implementation
__host__ 
void* malloc_device(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

__host__ 
void* malloc_host(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

__host__ 
void free_device(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

__host__ 
void free_host(void* ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}
