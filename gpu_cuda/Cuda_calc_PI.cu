#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define N 4096
#define BLOCK_SIZE 256

__device__ float f(float x) {
    return sqrtf(1 - x*x);
}

__global__ void aireTrapeze(float* tab, float intervalleSize){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        /*
        float a = i * intervalleSize;
        float b = (i + 1) * intervalleSize;
        tab[i] = 0.5f * (f(a) + f(b)) * intervalleSize;
        */
        tab[i] = f(i*intervalleSize) * intervalleSize;
    }
}

int main() {
    float *d_tab;
    float h_tab[N];

    cudaMalloc(&d_tab, N * sizeof(float));

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    aireTrapeze<<<numBlocks, BLOCK_SIZE>>>(d_tab, 1.0f/N);

    cudaMemcpy(h_tab, d_tab, N * sizeof(int), cudaMemcpyDeviceToHost);



    float total_area = 0.0;
    for (int i = 0; i < N; i++) {
        total_area += h_tab[i];
    }

    std::cout << total_area * 4 << std::endl;

    // Libérer la mémoire sur le device
    cudaFree(d_tab);
}