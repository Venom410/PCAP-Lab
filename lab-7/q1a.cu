#include <iostream>
#include <cuda_runtime.h>

const int N = 4; // Number of rows
const int M = 4; // Number of columns

// CUDA kernel for matrix addition
__global__ void matrixAdd(int *a, int *b, int *c) {
    int row = blockIdx.x; // Each row computed by one thread
    int col = threadIdx.x;
    
    int index = row * M + col;

    // Perform matrix addition for the element at (row, col)
    c[index] = a[index] + b[index];
}

int main() {
    int a[N][M], b[N][M], c[N][M];
    int *dev_a, *dev_b, *dev_c;

    std::cout << "Enter values for matrix A (" << N << "x" << M << "):" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
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
    cudaMalloc((void**)&dev_a, N * M * sizeof(int));
    cudaMalloc((void**)&dev_b, N * M * sizeof(int));
    cudaMalloc((void**)&dev_c, N * M * sizeof(int));

    // Copy matrices a and b from host to device
    cudaMemcpy(dev_a, a, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * M * sizeof(int), cudaMemcpyHostToDevice);

    // Define the thread and block dimensions
    dim3 blockDim(M, 1); // Each thread computes one row
    dim3 gridDim(N, 1);  // Grid of size N x 1 (one thread per row)

    // Launch the CUDA kernel with the specified grid and block dimensions
    matrixAdd<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c);

    // Copy the result matrix c from device to host
    cudaMemcpy(c, dev_c, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Resultant matrix C:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << c[i][j] << " ";
        }
        std::cout << std::endl;
    }

}
