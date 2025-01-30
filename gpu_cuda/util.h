#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>

void handle_cuda_err(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "%s in %s line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_CUDA_ERR(err) (handle_cuda_err((err), __FILE__, __LINE__))

#define START_TIMER(start) HANDLE_CUDA_ERR(cudaEventRecord((start), 0))

#define STOP_TIMER(start, stop, elapsed) do { \
	HANDLE_CUDA_ERR(cudaEventRecord((stop), 0)); \
	HANDLE_CUDA_ERR(cudaEventSynchronize((stop))); \
	HANDLE_CUDA_ERR(cudaEventElapsedTime(&(elapsed), (start), (stop))); \
} while(0)

#define HANDLE_NULL_ERR(ptr) do { \
	if ((ptr) == NULL) { \
		fprintf(stderr, "Pointer "#ptr" is NULL in %s line %d\n", __FILE__, __LINE__); \
		exit(EXIT_FAILURE); \
	}\
} while(0)

#endif

