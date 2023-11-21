#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define M 4 // Number of rows
#define N 4 // Number of columns

__global__ void replaceRows(int *matrix, int numRows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < numRows) {
        if (row != 0) { // Skip the first row (row 0)
            for (int col = 0; col < N; col++) {
                matrix[row * N + col] = pow(matrix[row * N + col], row + 1);
            }
        }
    }
}


int main() {
    int matrix[M][N] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};    int *d_matrix; // Device matrix
    int size = M * N * sizeof(int);
    cudaMalloc((void**)&d_matrix, size);

    cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(1, M);     // 1 block per row
    dim3 dimBlock(1, 1);   // 1 thread per element

    replaceRows<<<dimGrid, dimBlock>>>(d_matrix, M);

    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}