#include <iostream>
#include <cuda_runtime.h>

const int N = 4; // Number of rows in A
const int M = 4; // Number of columns in B

// CUDA kernel for matrix multiplication (Each element of the resultant matrix computed by one thread)
__global__ void matrixMultiplyElement(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int result = 0;
    for (int i = 0; i < N; i++) {
        result += a[row * N + i] * b[i * M + col];
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
    dim3 blockDim(M, N);  // Each thread computes one element
    dim3 gridDim(1, 1);   // Grid of size 1x1 (one thread)

    // Launch the CUDA kernel for matrix multiplication element-wise
    matrixMultiplyElement<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c);

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
