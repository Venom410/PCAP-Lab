#include <stdio.h>
#include <math.h>

#define N 7
#define THREADS_PER_BLOCK 256

__global__ void calculateSine(float *angles, float *sineValues, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        sineValues[tid] = sinf(angles[tid]);
    }
}

int main() {
    float angles[N] = {0.0f, 0.2618f, 0.5236f, 0.7854f, 1.0472f, 1.3090f, 1.5708f};  // Initialize host array with angles in radians
    float sineValues[N];    // Host array to store sine values
    float *d_angles, *d_sineValues;  // Device arrays

    // Allocate memory on the device
    cudaMalloc((void **)&d_angles, N * sizeof(float));
    cudaMalloc((void **)&d_sineValues, N * sizeof(float));

    // Copy host array to device
    cudaMemcpy(d_angles, angles, N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel to calculate sine values
    calculateSine<<<numBlocks, THREADS_PER_BLOCK>>>(d_angles, d_sineValues, N);

    // Copy result back to host
    cudaMemcpy(sineValues, d_sineValues, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_angles);
    cudaFree(d_sineValues);

    // Print result
    printf("Angle (radians)\tSine Value\n");
    for (int i = 0; i < N; i++) {
        printf("%.4f\t\t%.4f\n", angles[i], sineValues[i]);
    }

    return 0;
}
