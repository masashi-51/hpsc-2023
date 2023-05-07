#include <cstdio>
#include <cstdlib>
#include <vector>

//first for loop
__global__ void init(int* bucket, int range) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= range) return;
  bucket[tid] = 0;
}

//second for loop
__global__ void cal(int* bucket, int* key, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;
  atomicAdd(&bucket[key[tid]], 1);
}

//third for loop
__global__ void sort(int* bucket, int* key, int range) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= range) return;
  int offset=0;
  for (int i=0; i<tid; i++) {
    offset += bucket[i];
  }
  for (int i=0; i<bucket[tid]; i++) {
    key[offset+i] = tid;
  }
}

int main() {
  int n = 200;
  int range = 10;
  int *key, *bucket;
  int M = 64;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  init<<<(range+M-1)/M,M>>>(bucket, range);
  cal<<<(n+M-1)/M,M>>>(bucket, key, n);
  sort<<<(range+M-1)/M,M>>>(bucket, key, range);
  cudaDeviceSynchronize();

/*
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
*/
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}

