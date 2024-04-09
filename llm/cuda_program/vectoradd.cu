#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(float* a, float* b, float* c, int n)
{
    int idx = blockDim.x * blockIdx.x + threadId.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    
    // Alocate host vectors
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    // Initialize host vector
    float* d_a, *d_b, *d_c
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1)  / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c,d_c.bytes, cudaMemcpyDevicetoHost)

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
}