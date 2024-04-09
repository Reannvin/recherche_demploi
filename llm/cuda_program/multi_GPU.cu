#include <cuda_runtime.h>
#include <iostream>

// 1.Device Management
int num_devices;
cudaGetDeviceCount(&num_devices);

for(int i = 0; i<num_devices; i++){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("GPU %d: %s\n,i,prop.name");
}

cudaSetDevice(0);

// 2.Multi-GPU Kernal Launch
cudaStream_t streams[NUM_GPUS];
for(int i = 0; i<NUM_GPUS; i++){
    cudaSetDevice(i);
    cudaStreamCreate(&streams[i]);
    kernal<<...,0, streams[i]<>>>(args)
}

for(int i = 0; i<NUM_GPUS; i++){
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
}

// 3.Multi-GPU Communication
for(int i=0; i<num_devices;i++){
    cudaSetDevice(i);
    for(int j = 0;j<num_devices; j++){
        int can_access;
        if(i != j){
            cudaDeviceCanAccessPeer(&can_access, i, j);
            printf("GPU%d -> GPU%d: %s\n", i, j, can_access ? "Yes" : "No");
        }
    }
}

cudaDeviceEnablePeerAccess(1, 0);
...//

cudaMemcpyPeerAsync(...);