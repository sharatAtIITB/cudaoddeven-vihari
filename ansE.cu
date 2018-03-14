#include<cuda.h>
#include "utils.h"
#include "timer.h"

#define ITEMS_PER_THREAD 1024

extern __device__ __host__ inline void swap(int a[], int i, int j);
extern __device__ __host__ void sort(int a[], int n);
extern void Print_list(int a[], int n, const char* title);

__global__ void block_sort(int* data){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int start = tid*ITEMS_PER_THREAD;
  sort(data + start, ITEMS_PER_THREAD);
}

__device__ void exchange(int* data, int ai, int bi){
  int tmp[2*ITEMS_PER_THREAD]; 
  int i=0;
  int old_ai = ai;
  int a_n = 0, b_n = 0;
  
  while(a_n<ITEMS_PER_THREAD && b_n<ITEMS_PER_THREAD){
    if (data[ai]<data[bi]){
      tmp[i] = data[ai];
      ai += 1;
      a_n += 1;
    }
    else{
      tmp[i] = data[bi];
      bi += 1;
      b_n += 1;
    }
    i += 1;
  }
  if (a_n<ITEMS_PER_THREAD){
    for (int j=0;j<ITEMS_PER_THREAD-a_n;j+=1)
      tmp[i+j] = data[ai+j];
  }
  else if(b_n<ITEMS_PER_THREAD){
    for (int j=0;j<ITEMS_PER_THREAD-b_n;j+=1)
      tmp[i+j] = data[bi+j];
  }
  
  for (int k=0;k<2*ITEMS_PER_THREAD;k++)
    data[old_ai+k] = tmp[k];
}

__global__ void oe_merge(int* data, int n, int odd){
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
  if (nbr<0 || nbr*ITEMS_PER_THREAD>=n){
    return;
  }
  if (tid<nbr)
    exchange(data, tid*ITEMS_PER_THREAD, nbr*ITEMS_PER_THREAD);
  }

void ansE(int* data, int nsize, int num_workers){
  int num_blocks = 16;
  int threads_per_block = 1024;
  nsize = num_blocks*threads_per_block*ITEMS_PER_THREAD;

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

  block_sort<<<num_blocks, threads_per_block>>>(dd);
  printf("Done block sort...\n");

  cudaMemcpy(data, dd, sizeof(int)*nsize, cudaMemcpyDeviceToHost);
  int n = (int)(nsize/ITEMS_PER_THREAD);
  double start, end;
  GET_TIME(start);
  for(int i=0;i<n;i++){
    cudaMemset(dsorted, 1, sizeof(int));
    oe_merge <<< num_blocks, threads_per_block>>> (dd, nsize, 0);
    checkErrors("Failed to invoke kernel");
    oe_merge <<< num_blocks, threads_per_block>>> (dd, nsize, 1);
    checkErrors("Failed to invoke kernel");
    cudaMemcpy(&is_sorted, dsorted, sizeof(int), cudaMemcpyDeviceToHost);
    if (i%1000 == 1){
      GET_TIME(end);
      double eta = ((end-start)/i)*n;
      printf("%d/%d ETA: %f\n", i, n, eta);
    }
  }while(is_sorted==0);

  cudaMemcpy(data, dd, sizeof(int)*nsize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

/*int main(){
  int d[] = {3, 4, 5, 2, 3, 6};
  int s = 1;
  exchange(d, 0, 3, 1,  &s);
  for (int i=0;i<6;i++)
    printf("%d ", d[i]);
  printf("\n%d", s);
}
*/
