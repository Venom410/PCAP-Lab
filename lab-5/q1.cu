#include <stdio.h>
#include"cuda_runtime.h"

#define N 3  // Length of vectors
#define THREADS_PER_BLOCK N

__global__ void addVectorsWithBlockSize(int *a, int *b, int *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void addVectorsWithNThreads(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int a[N] = {1, 3, 5};
    int b[N] = {2, 4, 6};
    int c[N];  // Host result vector
    int *d_a, *d_b, *d_c;   // Device vectors

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Part A: Using block size as N
    addVectorsWithBlockSize<<<1, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    // Part B: Using N threads
    addVectorsWithNThreads<<<1, N>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Print result
    printf("Result: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    return 0;
}
