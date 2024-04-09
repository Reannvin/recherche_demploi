#include <cuda_runtime.h>
#include <iostream>

cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

float* d_a, *d_b, *d_c;
cudaMalloc(&d_a,...);
cudaMalloc(&d_b,...);
cudaMalloc(&d_c,...);

// Exec kernal in Stream1
kernal<<<...,0,stream1>>>(d_a, d_b, d_c);

// Exec transfer from CPU to GPU in Stream2
cudaMemcpyAsync(d_a,...,cudaMemcpyHostToDevice,stream2);

// Exec transfer from GPU to CPU in Stream1 after Stream2 done.
cudaMemcpyAsync(d_b.d_a,...,cudaMemcpyHostToDevice,stream1);

// Wait stream1 finished its kernal.
cudaStreamSynchronize(stream1);

