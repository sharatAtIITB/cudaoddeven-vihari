#include<cuda.h>
#include "utils.h"

extern __device__ __host__ inline void swap(int a[], int i, int j);

__global__ void _ansA(int* data, int n, int* sorted, int odd){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid%2 != odd)
    return;
  if (tid<n && data[tid]>data[tid+1]){
    swap(data, tid, tid+1);
    *sorted = 0;
  }
}

void ansA(int* data, int nsize, int num_blocks, int threads_per_block){
  int* dd;
  int is_sorted = 1;
  int *dsorted;
  cudaMalloc(&dsorted, 1 * sizeof(int));
  cudaMemset(dsorted, 1, sizeof(int));
  checkErrors("Failed to set");

  cudaMalloc(&dd, nsize * sizeof(int));
  checkErrors("Failed to allocate");
  cudaMemcpy(dd, data, nsize * sizeof(int), cudaMemcpyHostToDevice);
  checkErrors("Failed to copy");

  do{
    cudaMemset(dsorted, 1, sizeof(int));
    _ansA <<< num_blocks, threads_per_block>>> (dd, nsize, dsorted, 0);
    _ansA <<< num_blocks, threads_per_block>>> (dd, nsize, dsorted, 1);
    cudaMemcpy(&is_sorted, dsorted, sizeof(int), cudaMemcpyDeviceToHost);
  }while(is_sorted==0);
  cudaMemcpy(data, dd, sizeof(int)*nsize, cudaMemcpyDeviceToHost);
}
