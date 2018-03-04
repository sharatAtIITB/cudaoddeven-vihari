#include<cuda.h>
#include "utils.h"

#define ITEMS_PER_THREAD 1024

extern __device__ __host__ inline void swap(int a[], int i, int j);
extern __device__ __host__ void sort(int a[], int n);

__global__ void block_sort(int* data, int n){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int start = tid*ITEMS_PER_THREAD;
  int end = start + ITEMS_PER_THREAD;
  sort(data + start, end-start);
}

__device__ void exchange(int* data, int ai, int bi, int mins, int* sorted){
  int *a = data+ai;
  int *b = data+bi;
  for (int i=0;i<ITEMS_PER_THREAD;i++){
    if (mins==1 && a[i] > b[i]){
      swap(data, ai+i, bi+i);
      *sorted = 0;
    }
    if (mins==0 && a[i] < b[i]){
      swap(data, ai+i, bi+i);
      *sorted = 0;
    }
  }
}

__global__ void oe_merge(int* data, int n, int* sorted, int odd){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  
  int nbr;
  if (odd==0){
    if (tid%2==0)
      nbr = tid + 1;
    else
      nbr = tid - 1;
  }else{
    if (tid%2==0)
      nbr = tid-1;
    else nbr = tid + 1;
  }
  if (nbr<0 || nbr*ITEMS_PER_THREAD>n)
    return;
  if (nbr<tid)
    exchange(data, tid, nbr, 0, sorted);
  else
    exchange(data, tid, nbr, 1, sorted);
}

void ansE(int* data, int nsize, int num_workers){
  int num_blocks = 16;
  int threads_per_block = 16;
  nsize = 16*threads_per_block*ITEMS_PER_THREAD;

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

  block_sort<<<num_blocks, threads_per_block>>>(dd, nsize);
  printf("Done block sort...\n");
  do{
    cudaMemset(dsorted, 1, sizeof(int));
    printf("Merging...(1)\n");
    oe_merge <<< num_blocks, threads_per_block>>> (dd, nsize, dsorted, 0);
    checkErrors("Failed to invoke kernel");
    printf("Merging...(2)\n");
    oe_merge <<< num_blocks, threads_per_block>>> (dd, nsize, dsorted, 1);
    checkErrors("Failed to invoke kernel");
    printf("Reading back to host");
    cudaMemcpy(&is_sorted, dsorted, sizeof(int), cudaMemcpyDeviceToHost);
  }while(is_sorted==0);
  printf("Reading back to host, final");
  cudaMemcpy(data, dd, sizeof(int)*nsize, cudaMemcpyDeviceToHost);
}
