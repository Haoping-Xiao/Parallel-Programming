/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <vector>



static inline void check(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << context << ": "
          << cudaGetErrorString(err) << std::endl;
      std::exit(EXIT_FAILURE);
  }
}

#define CHECK(x) check(x, #x)

template <class T>
void cuda_memcpy(T* target, const T* source, std::size_t num, cudaMemcpyKind direction) {
  CHECK(cudaMemcpy(target, source, num * sizeof(T), direction));
}

__global__ void correlate_gpu(int ny, int nx, const float*data, float *result){
  int i=threadIdx.x+blockIdx.x*blockDim.x;
  int j=threadIdx.y+blockIdx.y*blockDim.y;
  if(i>=ny || j>=ny) return;
  if (i>j){
    result[i*ny+j]=0;
    return;
  }
  double temp=0;
  for (int k=0; k<nx; ++k){
    temp+=data[i*nx+k]*data[j*nx+k];
  }
  result[i*ny+j]=temp;
}


static inline int divup(int a, int b) {
  return (a + b - 1)/b;
}

void correlate(int ny, int nx, const float *data, float *result) {

  //allocate memory & copy data to GPU
  
  float *dGPU=NULL;
  CHECK(cudaMalloc((void**)&dGPU,ny*nx*sizeof(float)));
  float *rGPU=NULL;
  CHECK(cudaMalloc((void**)&rGPU,ny*ny*sizeof(float)));


  // float *avg=new float[ny]{0};
  // float *normalized=new float[ny*nx]{0};
  // float *sqrtSqureSum=new float[ny]{0};



  std::vector<float> avg(ny,0);
  std::vector<float> normalized(ny*nx,0);
  std::vector<float> sqrtSqureSum(ny,0);

  for (int y=0; y<ny; ++y){
    double temp=0;
    for (int x=0; x<nx; ++x){
      temp+=data[y*nx+x];
    }
    avg[y]=temp/nx;
  }
  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      normalized[y*nx+x]=data[y*nx+x]-avg[y];
    }
  }
  // delete[] avg;
  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      sqrtSqureSum[y]+=pow(normalized[y*nx+x],2);
    }
    sqrtSqureSum[y]=sqrt(sqrtSqureSum[y]);
  }
  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      normalized[y*nx+x]/=sqrtSqureSum[y];
    }
  }
  // delete[] sqrtSqureSum;


  cuda_memcpy(dGPU,normalized.data(),ny*nx,cudaMemcpyHostToDevice);
  // CHECK(cudaMemcpy(dGPU,normalized.data(),ny*nx*sizeof(float),cudaMemcpyHostToDevice));
  dim3 dimBlock(16,16);
  dim3 dimGrid(divup(ny,dimBlock.x),divup(ny,dimBlock.y));
  correlate_gpu<<<dimGrid,dimBlock>>>(ny,nx,dGPU,rGPU);
  CHECK(cudaGetLastError());
  cuda_memcpy(result, rGPU, ny * ny, cudaMemcpyDeviceToHost);
  // CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dGPU));
  CHECK(cudaFree(rGPU));
  // delete[] normalized;
}
