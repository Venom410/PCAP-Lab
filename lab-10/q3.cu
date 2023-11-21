#include <stdio.h>
#include <cuda_runtime.h>

__global__ void oddEvenSort(int *arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int phase, i, temp;

    for (phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) { // Even phase
            if (tid % 2 == 0 && tid < n - 1) {
                i = tid;
                if (arr[i] > arr[i + 1]) {
                    temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
            }
        } else { // Odd phase
            if (tid % 2 == 1 && tid < n - 1) {
                i = tid;
                if (arr[i] > arr[i + 1]) {
                    temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
            }
        }
        __syncthreads(); // Synchronize threads
    }
}

int main() {
    int n = 10;
    int arr[] = {2, 1, 4, 9, 5, 3, 6, 10, 8, 7};

    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    oddEvenSort<<<grid, block>>>(d_arr, n);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
