#include "nn.cuh"

const int num_threads_large = 1024;

cublasHandle_t g_cublas_handle = nullptr;

__device__ int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size, int elementsPerThread) {
    // parallel reduction of sum of squares via CUB
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < size)
            ss += x[j] * x[j];
    }
    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);

    // serialization point to calculate normalization factor
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize and scale
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < size) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }
}

void rmsnorm(float *o, float *x, float *weight, int size) {
    int elementsPerThread = divUp(size, num_threads_large);
    rmsnorm_kernel<<<1, num_threads_large>>>(o, x, weight, size, elementsPerThread);
}

__device__ void softmax_gpu(float *__restrict__ x, int size) {
    int tid = threadIdx.x;
    int step = blockDim.x;

    float max_val = tid < size ? x[tid] : 0;
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int i = tid; i < size; i += step) {
        x[i] /= sum;
    }
}

void matmul(float *xout, float *x, float *w, int n, int d) {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemv(g_cublas_handle, CUBLAS_OP_T, n, d, &alpha, w, n, x, 1, &beta, xout, 1);
}

__global__ void RoPE_rotation_kernel(int pos, float *sq, float *sk, int kv_dim, int head_size) {
    int i = threadIdx.x * 2;
    int head_dim = i % head_size;
    float freq = 1.0f / powf(500000.0f, head_dim / (float) head_size); // 500,000 RoPE frequency hyperparameter as per llama3.1 paper
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1;
    for (int v = 0; v < rotn; v++) {
        float *vec = v == 0 ? sq : sk;
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

void RoPE_rotation(int pos, RunState *s, int dim, int kv_dim, int head_size) {
    RoPE_rotation_kernel<<<1, dim / 2>>>(pos, s->q, s->k, kv_dim, head_size);
}

__global__ void accum_kernel(float *a, float *b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] += b[i];
    }
}

void accum(float *a, float *b, int size) {
    int num_threads_small = 256; // You might want to define this as a constant
    accum_kernel<<<divUp(size, num_threads_small), num_threads_small>>>(a, b, size);
}

void create_cublas_handle() {
    cublasStatus_t stat = cublasCreate(&g_cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
}

void destroy_cublas_handle() {
    cublasStatus_t stat = cublasDestroy(g_cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS destruction failed\n");
        exit(EXIT_FAILURE);
    }
}

__device__ float swish(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swiglu_kernel(float *out, float *x, float *w1, float *w2, float *b1, float *b2, int hidden_dim, int ffn_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ffn_dim) {
        float val1 = 0.0f, val2 = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            val1 += x[j] * w1[j * ffn_dim + i];
            val2 += x[j] * w2[j * ffn_dim + i];
        }
        val1 += b1[i];
        val2 += b2[i];
        out[i] = swish(val1) * val2;
    }
}

void swiglu(RunState *s, int hidden_dim, int ffn_dim) {
    int num_threads = 256;
    int num_blocks = (ffn_dim + num_threads - 1) / num_threads;
    swiglu_kernel<<<num_blocks, num_threads>>>(s->hb, s->x, s->w1, s->w2, s->b1, s->b2, hidden_dim, ffn_dim);
}
