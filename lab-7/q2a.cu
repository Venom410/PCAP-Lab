#include <iostream>
#include <cuda_runtime.h>

const int N = 4;
const int M = 4;

// CUDA kernel for matrix multiplication (Each row of the resultant matrix computed by one thread)
__global__ void matrixMultiplyRow(int *a, int *b, int *c) {
    int row = blockIdx.x; // Each row computed by one thread
    int col = threadIdx.x;

    int result = 0;
    for (int i = 0; i < M; i++) {
        result += a[row * M + i] * b[i * M + col];
    }

    c[row * M + col] = result;
}

int main() {
    int a[N][N], b[N][M], c[N][M];
    int *dev_a, *dev_b, *dev_c;

    // Initialize matrices a and b
    std::cout << "Enter values for matrix A (" << N << "x" << N << "):" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cin >> a[i][j];
        }
    }

    std::cout << "Enter values for matrix B (" << N << "x" << M << "):" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cin >> b[i][j];
        }
    }

    // Allocate memory on the GPU
    cudaMalloc((void**)&dev_a, N * N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * M * sizeof(int));
    cudaMalloc((void**)&dev_c, N * M * sizeof(int));

    // Copy matrices a and b from host to device
    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * M * sizeof(int), cudaMemcpyHostToDevice);

    // Define the thread and block dimensions
    dim3 blockDim(M, 1);  // Each thread computes one element in a row
    dim3 gridDim(N, 1);   // Grid of size N x 1 (one thread per row)

    // Launch the CUDA kernel for matrix multiplication row-wise
    matrixMultiplyRow<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c);

    // Copy the result matrix c from device to host
    cudaMemcpy(c, dev_c, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Print the result matrix
    std::cout << "Resultant matrix C:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << c[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
