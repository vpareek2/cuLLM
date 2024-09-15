/**
 * This file contains utility functions for the transformer model.
 */

#include "utils.cuh"

/**
 * @brief Divides the first argument by the second and rounds up to the nearest integer, for CUDA kernel launches.
 * @param a The dividend.
 * @param b The divisor.
 * @return The smallest integer greater than or equal to the division of a by b.
 */
__host__ 
__device__ 
inline int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

/**
 * @brief Returns the current time in milliseconds.
 * @return The current time in milliseconds.
 */
__host__ 
long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

/**
 * @brief Samples an index from a probability distribution.
 * @param probabilities The probability distribution.
 * @param n The size of the probability distribution.
 * @return The index with the highest probability.
 */
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