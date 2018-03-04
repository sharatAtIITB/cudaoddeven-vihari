#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include<stdio.h>
#include<cuda.h>

__device__ __host__ inline void swap(int a[], int i, int j){
  int temp = a[i];
  a[i] = a[j];
  a[j] = temp;
}

void checkErrors(const char* msg) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "%s\n", msg);
    fprintf(stderr, "Failed: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__device__ __host__ void sort(int* a, int n){
  int is_sorted = 1;
  do{
    is_sorted = 1;
    for(int i=0;i<n;i+=2)
      if (a[i]>a[i+1]){
	swap(a, i, i+1);
	is_sorted = 0;
      }
    for(int i=1;i<n;i+=2)
      if (a[i]>a[i+1]){
	swap(a, i, i+1);
	is_sorted = 0;
      }
  } while(is_sorted==0);
}

#endif
