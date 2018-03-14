#include<cuda.h>
#include "utils.h"

extern __device__ __host__ inline void swap(int a[], int i, int j);

__global__ void _ansC(int* data, int n){
  __shared__ int odd;
  __shared__ int i;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0){
    odd = 0;
    i = 0;
  }
  __syncthreads();

  while (true){
    if (i>=n)
      return;
    
    if (tid%2==odd)
      if (tid<n && data[tid]>data[tid+1])
	swap(data, tid, tid+1);

    __syncthreads();
    if (tid == 0){
      odd = !odd;
      i += 1;
    }
  }

}

void ansC(int* data, int nsize, int num_workers){
  int num_blocks = 1;
  int threads_per_block = num_workers;

  int* dd;
  cudaMalloc(&dd, nsize * sizeof(int));
  checkErrors("Failed to allocate");
  cudaMemcpy(dd, data, nsize * sizeof(int), cudaMemcpyHostToDevice);
  checkErrors("Failed to copy");

  _ansC <<<num_blocks, threads_per_block>>> (dd, nsize);
  cudaDeviceSynchronize();
  cudaMemcpy(data, dd, sizeof(int)*nsize, cudaMemcpyDeviceToHost);
}
