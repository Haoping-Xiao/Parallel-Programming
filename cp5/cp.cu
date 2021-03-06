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
  const int nd=8;//         nd:    nd==blockDim.x==blockDim.y
                  //                compute nd*nd results each thread.

  // int step=nd*nd;// each block will compute step*step results.
  int ia=threadIdx.x;
  int ja=threadIdx.y;
  int ic=blockIdx.x;
  int jc=blockIdx.y;
  // int i=ic*step+ib*nd+ia;
  // int j=jc*step+jb*nd+ja;
  // 0<=ia<=nd, 0<=ja<=nd
  // if ic>jc , then i>j
  if(ic>jc){ 
    for(int ib=0; ib<nd; ib++){
      for(int jb=0; jb<nd; jb++){
        int i=(ic*nd+ib)*nd+ia;
        int j=(jc*nd+jb)*nd+ja;
        if(i<ny&&j<ny){
          result[ny*i+j]=0;
        }
      }
    }
  }else{
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
      for(int ii=0; ii<nd; ii++){
        int i=(ic*nd+ii)*nd+ia;
        int j=(jc*nd+ii)*nd+ja;
        x[ii]=data[k*new_ny +i];
        y[ii]=data[k*new_ny +j];
      }



      for(int ib=0; ib<nd; ib++){
        for(int jb=0; jb<nd; jb++){
            v[ib][jb]+=x[ib]*y[jb];
        }
      }
    }
    for(int ib=0; ib<nd; ib++){
      for(int jb=0; jb<nd; jb++){
        int i=(ic*nd+ib)*nd+ia;
        int j=(jc*nd+jb)*nd+ja;
        if(i<ny&&j<ny){
          result[ny*i+j]=v[ib][jb];
        }
      }
    }
  }
  

  // result[i*ny+j]=temp;
}

__global__ void normalize(int ny, int nx, float*data, int step){
  int i = blockIdx.x;
  int j = threadIdx.x;
  int row=i*step+j;
  
  if (row<ny){
    // printf("row is %d \n", row);
    //for each row
    float temp=0, avg=0, sqrtSqureSum=0;
    for (int x=0; x<nx; ++x){
      temp+=data[row*nx+x];
    }
    avg=temp/nx;
    for (int x=0; x<nx; ++x){
      data[row*nx+x]=data[row*nx+x]-avg;
    }

    for (int x=0; x<nx; ++x){
      sqrtSqureSum+=powf(data[row*nx+x],2);
    }
    sqrtSqureSum=sqrtf(sqrtSqureSum);

    for (int x=0; x<nx; ++x){
      data[row*nx+x]/=sqrtSqureSum;
    }
  }
}


__global__ void padding_transpose(int ny, int nx, const float*data, float* result, int new_ny){
  //result is padding and transpose data
  int ja=threadIdx.x;
  int i=blockIdx.y;
  
  for (int jb=0; jb<nx; jb+=blockDim.x){
    int j=jb+ja;
    if (j>=nx) break;
    float v=i<ny?data[i*nx+j]:0.0; //padding
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
  const int block_size=8; //16*16 threads
  const int step=block_size*block_size; // each block will compute step*step results.

  int new_ny=roundup(ny,step);
  
  //allocate memory & copy data to GPU
  float *dGPU=NULL;
  CHECK(cudaMalloc((void**)&dGPU,ny*nx*sizeof(float)));

  float *padding=NULL;
  CHECK(cudaMalloc((void**)&padding,new_ny*nx*sizeof(float)));

  float *rGPU=NULL;
  CHECK(cudaMalloc((void**)&rGPU,ny*ny*sizeof(float)));

  cuda_memcpy(dGPU,data,ny*nx,cudaMemcpyHostToDevice);


  {
    normalize<<<divup(ny,step),step>>>(ny,nx,dGPU,step);
  }

  // Run kernel to padding and transpose
  {
    dim3 dimBlock(64,1);
    dim3 dimGrid(1,new_ny);
    padding_transpose<<<dimGrid,dimBlock>>>(ny,nx,dGPU,padding,new_ny);
    CHECK(cudaGetLastError());
  }


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

