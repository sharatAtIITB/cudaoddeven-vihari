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

#endif
