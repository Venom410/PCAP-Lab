#include <stdio.h>
#include <cuda_runtime.h>

#define KERNEL_SIZE 5
#define DATA_SIZE 10
#define THREADS_PER_BLOCK 256

__constant__ float d_kernel[KERNEL_SIZE];

__global__ void convolution1D(const float* input, float* output, int data_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    for (int i = 0; i < KERNEL_SIZE; i++) {
        int idx = tid + i - KERNEL_SIZE / 2;
        if (idx >= 0 && idx < data_size) {
            sum += input[idx] * d_kernel[i];
        }
    }

    output[tid] = sum;
}

int main() {
    float h_input[DATA_SIZE] = {1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
    float h_output[DATA_SIZE];
    float h_kernel[KERNEL_SIZE] = {0.1, 0.2, 0.4, 0.2, 0.1};

    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * sizeof(float));

    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, DATA_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output, DATA_SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (DATA_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    convolution1D<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, DATA_SIZE);

    cudaMemcpy(h_output, d_output, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    printf("Input Signal: ");
    for (int i = 0; i < DATA_SIZE; i++) {
        printf("%.1f ", h_input[i]);
    }
    
    printf("\n");
    printf("\n");

    printf("Kernel: ");
    for (int i = 0; i < KERNEL_SIZE; i++) {
        printf("%.1f ", h_kernel[i]);
    }
    
    printf("\n");
    printf("\n");
    
    printf("Output Signal: ");
    for (int i = 0; i < DATA_SIZE; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n");

    return 0;
}
