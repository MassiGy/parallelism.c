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


__global__ void sumVect(float* in, float *out, int esize ){ // esize = effective size
    __shared__ float cache[N];      // reserve N*sizeof(float) bytes that are shared across all threads of a block

    int offset = blockIdx.x * blockDim.x;
    int i = threadIdx.x;


    // copy in to cache (in parallel)
    if(i+offset < esize) {
        cache[i] = in[i+offset];
    }else {
        cache[i] = 0;
    }

    // reduce the cache by summing up (in parallel)
    int k = blockDim.x/2;
    while (k > 0){
        __syncthreads();
        if(i<k) {
            cache[i] = cache[i] + cache[k+i];
        }
        k = k/2;
    }

    // copy the result (first float slot of cache) to the out
    if(i == 0){ // this will make sure that only one thread writes the result (it can be any number within 0 and blockDim-1 included)
        out[blockIdx.x] = cache[0]; // we reduce in a foldr fashion, so the result will be in the left most slot
    }

}

int main() {
    float *d_tab;
    float h_tab[N];

    cudaMalloc(&d_tab, N * sizeof(float));

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    aireTrapeze<<<numBlocks, BLOCK_SIZE>>>(d_tab, 1.0f/N);

    /*
     * Here we get back the results from the GPU to do the sum on the CPU side
    cudaMemcpy(h_tab, d_tab, N * sizeof(int), cudaMemcpyDeviceToHost);

    float total_area = 0.0;
    for (int i = 0; i < N; i++) {
        total_area += h_tab[i];
    }

    std::cout << total_area * 4 << std::endl;
    */

    // now we will try to do the reduction of the res vect on the GPU side (using the sum just as before)
    int n = N;
    numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // these will be our args for the sumVect kernel
    float *in = d_tab;
    float *out; cudaMalloc(&out, N * sizeof(float ));

    float *temp;    // this will help us permute our args since the output of our i'th invokation of our kernel is the input of the i+1'th one
    while(n > 1){
        sumVect<<<numBlocks, BLOCK_SIZE>>>(in, out, n);
        n = numBlocks;
        numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        temp = in;
        in = out;
        out = temp;
    }

    // get the reduction result from the GPU memory to the Host memory and print it
    float res;
    cudaMemcpy(&res, in, sizeof(float), cudaMemcpyDeviceToHost);
    printf("PI: %f\n", res*4);


    // Libérer la mémoire sur le device
    cudaFree(d_tab);
    cudaFree(out);
}