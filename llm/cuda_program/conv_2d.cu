#include <cuda_runtime.h>
#include <iostream>
__global__ void convolution_2D(float* input, float* output, float* kernel, int inch, int outch, int ksize, int iH, int iW, int oH, int oW) {
    /*
     * Maps the output value at coordinate (ox, oy) to the memory location
     * output[(oy * oW + ox) * outch + outchid]
     */
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int outchid = blockIdx.z;

    if (ox < oW && oy < oH && outchid < outch) {
        float sum = 0.0f;
        for (int inch_id = 0; inch_id < inch; ++inch_id) {
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    int input_row = oy - ksize / 2 + ky;
                    int input_col = ox - ksize / 2 + kx;
                    if (input_row >= 0 && input_row < iH && input_col >= 0 && input_col < iW) {
                        sum += input[(input_row * iW + input_col) * inch + inch_id] *
                               kernel[(ky * ksize + kx) * inch * outch + outchid * inch + inch_id];
                    }
                }
            }
        }
        output[(oy * oW + ox) * outch + outchid] = sum;
    }
}

// Shared Mem
__global__ void matrixMulSharedMem(float* A, float* B, float* C, int M, int N, int K){
    __shared__ float sharedA[16][16];
    __shared__ float sharedB[16][16];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    for(int m = 0;m < 16; m += BLOCK_SIZE){
        sharedA[ty][tx] = A[(by * 16 + ty) * K + m + tx];
        sharedB[ty][tx] = B[(m+ ty) * N + bx * 16 + tx];
        __syncthreads();
    }
}

// Constant Mem
__constant__ float kernal[9];
float host_kernal[9] = {...};
cudaMemcpyToSymbol(kernal, host_kernal, 9*sizeof(float));

__global__ void convolution2D(float* input, float* output, int iH, int iW){
    // Calcul the thread inx
    ...

    float sum = 0.0f;
    for (int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
            sum += input[...] * kernal[j*3 + i];
        }
    }
    output[...] = sum;
}