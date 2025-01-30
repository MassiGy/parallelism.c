#include <cuda_runtime.h>
#include <iostream>
#include "util.h"
#include <cmath>

#define N 4096
#define BLOCK_SIZE 256

__global__ void multVects(float *res, float*a, float*b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i <N) {
        res[i] = a[i]*b[i];
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
    float h_fst_vect[N], h_snd_vect[N], h_res_vect[N];
    float *d_fst_vect, *d_snd_vect, *d_res_vect;

    // populate our data on the host memory
    for (int i = 0; i < N; ++i) {
        h_fst_vect[i] = h_snd_vect[i] = 1;
    }

    // allocate the memory for our GPU memory
    cudaMalloc(&d_fst_vect, N * sizeof(float));
    cudaMalloc(&d_snd_vect, N * sizeof(float));
    cudaMalloc(&d_res_vect, N * sizeof(float));

    // copy our in-host memory data to our GPU memory
    cudaMemcpy(d_fst_vect, h_fst_vect, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snd_vect, h_snd_vect, N * sizeof(float), cudaMemcpyHostToDevice);


    // offset guard & kernel invoke
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    multVects<<<numBlocks, BLOCK_SIZE>>>(d_res_vect, d_fst_vect, d_snd_vect);


    /*
    // gather the res from GPU to host memory
    cudaMemcpy(h_res_vect, d_res_vect, N * sizeof(float), cudaMemcpyDeviceToHost);

    // print the mult results
    for (int i = 0; i < N; ++i) {
        printf("res[%d]:%f\n", i, h_res_vect[i]);
    }
    */

    // start the reduction using the sum on GPU side
    int n = N;
    numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // we will use d_res_vect and d_fst_vect since they are already on the GPU (we will abuse them for something else)
    float *in  = d_res_vect;
    float *out = d_fst_vect;        // this one will be overwitten ( we can memset its content to 0)
    while(n > 1){
        sumVect<<<numBlocks, BLOCK_SIZE>>>(in, out, n);
        n = numBlocks;
        numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // permute the two args since the out of the previous sumVect invoke is the input of the next one
        float * temp = in;
        in = out;
        out = temp;
    }


    // get the reduced value (the result) from GPU memory to host memory
    float res;
    cudaMemcpy(&res, in, sizeof(float), cudaMemcpyDeviceToHost);    // the res will be in the input vector of the sumVect kernel (review the previous loop)

    printf("res:%f\n", res);


    // free the memory on the GPU memory
    cudaFree(d_fst_vect);
    cudaFree(d_snd_vect);
    cudaFree(d_res_vect);
}