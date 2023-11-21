#include <stdio.h>

#define WIDTH 4
#define HEIGHT 4
#define MASK_WIDTH 3
#define MASK_HEIGHT 3

__global__ void convolution2D(float* input, float* mask, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int halfWidth = MASK_WIDTH / 2;
        int halfHeight = MASK_HEIGHT / 2;
        float sum = 0.0f;

        for (int i = -halfHeight; i <= halfHeight; i++) {
            for (int j = -halfWidth; j <= halfWidth; j++) {
                int offsetX = x + j;
                int offsetY = y + i;

                if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height) {
                    sum += input[offsetY * width + offsetX] * mask[(i + halfHeight) * MASK_WIDTH + (j + halfWidth)];
                }
            }
        }

        output[y * width + x] = sum;
    }
}

int main() {
    float input[WIDTH * HEIGHT];
    float mask[MASK_WIDTH * MASK_HEIGHT];
    float output[WIDTH * HEIGHT];


    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            input[i * WIDTH + j] = 1.0f; 
        }
    }

    for (int i = 0; i < MASK_HEIGHT; i++) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            mask[i * MASK_WIDTH + j] = 0.5f; 
        }
    }

    float *d_input, *d_mask, *d_output;
    cudaMalloc((void**)&d_input, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&d_mask, MASK_WIDTH * MASK_HEIGHT * sizeof(float));
    cudaMalloc((void**)&d_output, WIDTH * HEIGHT * sizeof(float));

    cudaMemcpy(d_input, input, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, MASK_WIDTH * MASK_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    convolution2D<<<gridDim, blockDim>>>(d_input, d_mask, d_output, WIDTH, HEIGHT);

    cudaMemcpy(output, d_output, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%f ", output[y * WIDTH + x]);
        }
        printf("\n");
    }

    return 0;
}
