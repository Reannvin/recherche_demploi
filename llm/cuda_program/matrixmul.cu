#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMul(float* A, float* B, float* C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float sum = 0.0f;
        for (int i = 0;i < K; i ++){
            sum += A[row * K + i] * B[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main(){
    const int M = 1024, N = 1024, K = 1024;
    const int SIZE = M * N * sizeof(float);
    
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));

    float* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * M * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(N / threadsPerBlock.x, M/threadsPerBlock.y);
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C,M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}