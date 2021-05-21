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

  // params: 
  //         data : transposed padding data
  

__global__ void correlate_gpu(int ny, int nx, const float*data, float *result, int new_ny){
  const int nd=16;//         nd:    nd==blockDim.x==blockDim.y
                  //                compute nd*nd results each thread.

  int step=nd*nd;// each block will compute step*step results.
  int ia=threadIdx.x;
  int ja=threadIdx.y;
  int ic=blockIdx.x;
  int jc=blockIdx.y;

  // int i=threadIdx.x+blockIdx.x*blockDim.x;
  // int j=threadIdx.y+blockIdx.y*blockDim.y;

  // if(i>=ny || j>=ny) return;
  // if (i>j){
  //   result[i*ny+j]=0;
  //   return;
  // }

  float v[nd][nd];
  // double temp=0;
  for(int ib=0; ib<nd; ib++){
    for(int jb=0; jb<nd; jb++){
      v[ib][jb]=0;
    }
  }

  for (int k=0; k<nx; ++k){

    float x[nd];
    float y[nd];

    for(int ib=0; ib<nd; ib++){
      int i=ic*step+ib*nd+ia;
      x[ib]=data[k*new_ny +i];
    }

    for(int jb=0; jb<nd; jb++){
      int j=jc*step+jb*nd+ja;
      y[jb]=data[k*new_ny+j];
    }

    for(int ib=0; ib<nd; ib++){
      for(int jb=0; jb<nd; jb++){
        v[ib][jb]+=x[ib]*y[jb];
      }
    }
  }
  for(int ib=0; ib<nd; ib++){
    for(int jb=0; jb<nd; jb++){
      int i=ic*step+ib*nd+ia;
      int j=jc*step+jb*nd+ja;
      if(i<ny&&j<ny&&i<=j){
        result[ny*i+j]=v[ib][jb];
      }
    }
  }

  // result[i*ny+j]=temp;
}




__global__ void padding_transpose(int ny, int nx, const float*data, float* result, int new_ny){
  //result is padding and transpose data
  int ja=threadIdx.x;
  int i=blockIdx.y;

  for (int jb=0; jb<nx; jb+=blockDim.x){
    int j=jb+ja;
    if (j>=nx) break;
    float v=i<ny?data[i*ny+j]:0; //padding
    result[new_ny*j+i]=v; //transpose
  }
}






static inline int divup(int a, int b) {
  return (a + b - 1)/b;
}
static inline int roundup(int a, int b) {
  return divup(a, b) * b;
}
void correlate(int ny, int nx, const float *data, float *result) {


  // const int nd=16;//compute nd*nd results each thread. could not less than 
  const int block_size=16; //16*16 threads
  const int step=block_size*block_size; // each block will compute step*step results.

  int new_ny=roundup(ny,step);
  
  //allocate memory & copy data to GPU
  float *dGPU=NULL;
  CHECK(cudaMalloc((void**)&dGPU,ny*nx*sizeof(float)));

  float *padding=NULL;
  CHECK(cudaMalloc((void**)&padding,new_ny*nx*sizeof(float)));

  float *rGPU=NULL;
  CHECK(cudaMalloc((void**)&rGPU,ny*ny*sizeof(float)));


  // float *avg=new float[ny]{0};
  // float *normalized=new float[ny*nx]{0};
  // float *sqrtSqureSum=new float[ny]{0};



  std::vector<float> avg(ny,0);
  std::vector<float> normalized(ny*nx,0);
  std::vector<float> sqrtSqureSum(ny,0);
  std::vector<float> transposed(nx*new_ny,0);

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

  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      std::cout << normalized[y*nx+x] << " ";
    }
    std::cout<< std::endl ;
  }

  cuda_memcpy(dGPU,normalized.data(),ny*nx,cudaMemcpyHostToDevice);

  // Run kernel to padding and transpose
  {
    dim3 dimBlock(64,1);
    dim3 dimGrid(1,new_ny);
    padding_transpose<<<dimGrid,dimBlock>>>(ny,nx,dGPU,padding,new_ny);
    CHECK(cudaGetLastError());
  }


  cuda_memcpy(transposed.data(), padding, new_ny * nx, cudaMemcpyDeviceToHost);
  std::cout << new_ny<<std::endl;
  std::cout << transposed.size()<<std::endl;
  for(int x=0;x<nx;x++){
    for(int y=0;y<new_ny;y++){
      std::cout<< transposed[x*new_ny+y] << " ";
    }
    std::cout<< std::endl<< "----"<< std::endl;
  }
  
  // for (int x=0;x<nx;++x){
  //   for (int y=0; y<ny; ++y){
  //     transposed[x*ny+y]=normalized[y*nx+x];
  //   }
  // }

  // Run kernel to calculate cp
  {
    dim3 dimBlock(block_size,block_size);
    dim3 dimGrid(new_ny/step,new_ny/step);
    correlate_gpu<<<dimGrid,dimBlock>>>(ny,nx,padding,rGPU,new_ny);
    CHECK(cudaGetLastError());
  }
  


  
  
  cuda_memcpy(result, rGPU, ny * ny, cudaMemcpyDeviceToHost);
  // CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dGPU));
  CHECK(cudaFree(padding));
  CHECK(cudaFree(rGPU));
  // delete[] normalized;
}

