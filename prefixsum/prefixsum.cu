#include <iostream>
template<typename T>
static inline void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

#define CHECK(x) check(x, #x,  __FILE__, __LINE__)



// template <class T>
// void cuda_memcpy(T* target, const T* source, std::size_t num, cudaMemcpyKind direction) {
// CHECK(cudaMemcpy(target, source, num * sizeof(T), direction));
// }


__global__ void prefixsum(unsigned int* mask, unsigned int* output,const int len, const unsigned int n ){
  // printf("checing len: %p",len);
  // printf("checing n: %p",n);


  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
  int step=len;
  int start=index*len+1;//exclusive
  if (start>n) return; //exclusive, could equal to n
  int end=start+step;
  output[start]=mask[start-1];
  for(unsigned int i=start+1;i<end&&i<n;i++){
    output[i]+=output[i-1]+mask[i-1];//exclusive, therefore mask[i-1]
  }
}


__global__ void serialsum_accrossthread(unsigned int* sum,const int len, const unsigned int n){
  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
  int step=len;
  // int offset=2*step-1;
  int offset=2*step;
  unsigned int start=step*blockDim.x*index+offset;
  unsigned int end=step*blockDim.x*(index+1)+1;
  for(unsigned int i=start;i<end && i<n; i+=step){
    sum[i]+=sum[i-step];
  }
}

__global__ void mergethread(unsigned int* sum,const int len, const unsigned int n){
  if (threadIdx.x==0) return;

  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
  int step=len;
  unsigned int start=index*step+1;//exclusive
  unsigned int end=start+step-1; // -1 is important, this position has been added in serial sum
  unsigned int base=sum[start-1];

  for(unsigned int i=start; i<end && i<n; i++){
    sum[i]+=base;
  }

}

// __global__ void serialsum_accrossblock(unsigned int* sum,const int len, const unsigned int n){
  

//   unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
//   int step=len*blockDim.x;
//   // int offset=2*step-1;
//   int offset=2*step;
//   unsigned int start= blockDim.x*step*index+offset;
//   unsigned int end= blockDim.x*step*(index+1);
//   for(unsigned int i=start; i<end && i<n; i+=step){
//     sum[i]+=sum[i-step];
//   }
// }


void serialsum_accrossblock(unsigned int* sum,const int len, const unsigned int n, const int block_size){
  int step=len*block_size;//each block has step number
  int start=2*step;

  for(unsigned int i=start; i<n; i+=step){
    sum[i]+=sum[i-step];
  }

}


__global__ void mergeblock(unsigned int* sum,const int len, const unsigned int n){
  

  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index==0) return;  //the first block is not needed to merge

  int step=len*blockDim.x;
  
  int start=index*step+1; //exclusive
  // int end=start+step;
  int end=start+step-1;// -1 is important, this position has been added in serial sum
  // int base=sum[blockIdx.x*len*blockDim.x-1];//last element at last block
  int base=sum[start-1];//last element at last block
  for(int i=start; i<end && i<n; i++){
    sum[i]+=base;
  }
}

static inline int divup(int a, int b) {
  return (a + b - 1)/b;
}

int main(){

  const unsigned int n=100000; //100000 number
  unsigned int data[n];
  unsigned int result[n];
  unsigned int inter_sum[n];

  unsigned int inter_result[n];
  unsigned int *cal_result=new unsigned int [n];

  
  for (unsigned int i=0; i<n; i++){
    data[i]=i;
  }
  for (unsigned int i=0; i<n; i++){
    cal_result[i]=i;
  }
  


  
  for (long long i=0; i<n; i++){
    result[i]=(i-1)*i/2;
  }


  std::cout<< "data preparation done"<<std::endl;
  const int block_size=64;//64 threads per block;
  const int len=1000; // add 1000 prefix sum per thread; 

  unsigned int *d_in=NULL;
  
  CHECK(cudaMalloc((void**)&d_in,n*sizeof(unsigned int)));
  unsigned int *d_sum=NULL;
  CHECK(cudaMalloc((void**)&d_sum,n*sizeof(unsigned int)));
  CHECK(cudaMemset(d_sum,0,n*sizeof(unsigned int)));
  CHECK(cudaMemcpy(d_in,data,n * sizeof(unsigned int), cudaMemcpyHostToDevice));
  // cuda_memcpy(d_in,data,n,cudaMemcpyHostToDevice);

  // std::cout<< divup(n,block_size*len) <<std::endl;
  // for (long long i=64001; i<65001; i++){
  //   inter_result[i]=(64000+i-1)*(i-64000)/2;
  // }

  prefixsum<<<divup(n,block_size*len),block_size>>>(d_in,d_sum,len,n);
  // CHECK(cudaMemcpy(cal_result, d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  // for (int i=64001; i<65001; i++){
  //   if(inter_result[i]!=cal_result[i]){
  //     std::cout<<"i: "<< i <<"error!"<<std::endl;
  //     std::cout<< inter_result[i] << "vs" << cal_result[i] <<std::endl;
  //     break;
  //     // return 0;
  //   }
  // }
  // std::cout<<"pass here"<<std::endl;


  CHECK(cudaGetLastError());
  long long start=64001;
  // int end=start+1000;
  unsigned int end=100000;
  // std::cout<< end*end<<std::endl;
  for (unsigned int i=start; i<end; i++){
    // int index=i-64000;
    inter_result[i]=((start-1+i-1)*(i-start+1))/2;
  }

  start=1;
  end=start+64000;
  for (long long i=start; i<end; i++){
    // int index=i-64000;
    
    inter_result[i]=(i-1)*(i-start+1)/2;
  }
  // CHECK(cudaMemcpy(cal_result, d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  // for (int i=65000; i<66000; i++){
  //   if(inter_result[i]!=cal_result[i]){
  //     std::cout<<"i: "<< i <<"error!"<<std::endl;
  //     break;
  //     // return 0;
  //   }
  // }
  // std::cout<<"pass"<<std::endl;
  

  // for (unsigned int i=65000; i<66000; i++){
  //   inter_result[i]=(65000+i)*(i-65000+1)/2;
  // }
    // for (long long i=64001; i<65001; i++){
  //   inter_result[i]=(64000+i-1)*(i-64000)/2;
  // }
  // for (unsigned int i=1001; i<2001; i++){
  //   inter_result[i]=(1000+i-1)*(i-1000)/2;
  // }
  // inter_result[2000]+=result[1000];

  serialsum_accrossthread<<<divup(n,block_size*len*block_size),block_size>>>(d_sum,len,n);
  CHECK(cudaGetLastError());
  
  // CHECK(cudaMemcpy(cal_result, d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  // for(int i=1001; i<2001; i++){
  //   if(inter_result[i]!=cal_result[i]){
  //     std::cout<<"first: i: "<< i << " " << cal_result[i] <<"error!"<<std::endl;
  //     break;
  //     // return 0;
  //   }
  // }
  // std::cout<<"pass"<<std::endl;
  // std::cout << "pass first one"<<std::endl;
  // for (int i=64000; i<65000; i++){
  //   if(inter_result[i]!=cal_result[i]){
  //     std::cout<<"i: "<< i << " " << cal_result[i] <<"error!"<<std::endl;
  //     break;
  //     // return 0;
  //   }
  // }
  // inter_result[65999]+=inter_result[64999];
  // for (int i=65000; i<66000; i++){
  //   if(inter_result[i]!=cal_result[i]){
  //     std::cout<<"i: "<< i << " " << cal_result[i] <<"error!"<<std::endl;
  //     break;
  //     // return 0;
  //   }
  // }

  mergethread<<<divup(n,block_size*len),block_size>>>(d_sum,len,n);
  CHECK(cudaGetLastError());


  // CHECK(cudaMemcpy(inter_result, d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  //serial sum
  CHECK(cudaMemcpy(inter_sum, d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  for (int i=64001; i<100000; i++){
    if(inter_result[i]!=inter_sum[i]){
      std::cout<<"i: "<< i <<"error!"<<std::endl;
      std::cout<< inter_result[i] << "vs" << inter_sum[i] <<std::endl;
      break;
      // return 0;
    }
  }
  std::cout<<"pass here 1"<<std::endl;

  serialsum_accrossblock(inter_sum, len, n, block_size);
  CHECK(cudaMemcpy(d_sum, inter_sum,n * sizeof(unsigned int), cudaMemcpyHostToDevice));

  // CHECK(cudaMemcpy(cal_result, d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  for (int i=1; i<100000; i++){
      if(inter_result[i]!=inter_sum[i]){
        std::cout<<"i: "<< i <<"error!"<<std::endl;
        std::cout<< inter_result[i] << "vs" << inter_sum[i] <<std::endl;
        break;
        // return 0;
    }
  }
  std::cout<<"pass here"<<std::endl;
  
  
  // serialsum_accrossblock<<<divup(n,block_size*len*block_size*block_size) ,block_size>>>(d_sum,len,n);
  // CHECK(cudaGetLastError());
  // CHECK(cudaMemcpy(cal_result, d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  // for (int i=0; i<100000; i++){
  //     if(inter_result[i]!=cal_result[i]){
  //       std::cout<<"i: "<< i <<"error!"<<std::endl;
  //       std::cout<< inter_result[i] << "vs" << cal_result[i] <<std::endl;
  //       break;
  //       // return 0;
  //   }
  // }
  // for (unsigned int i=64000; i<100000; i++){
  //   inter_result[i]+=inter_result[63999];
  // }

  // std::cout<< divup(n,block_size*len) << std::endl;
  mergeblock<<<divup(n,block_size*len*block_size) ,block_size>>>(d_sum,len,n);
  CHECK(cudaGetLastError());
  


  // cuda_memcpy(cal_result, d_sum, n, cudaMemcpyDeviceToHost);
  CHECK(cudaMemcpy(cal_result, d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_sum));

  //compare

  unsigned int i;
  for (i=0; i<n; i++){
    if(result[i]!=cal_result[i]){
      std::cout<<"i: "<< i <<"error!"<<std::endl;
      std::cout<<result[i]<<"vs"<<cal_result[i]<<std::endl;
      break;
    }
  }
  if(i==n){
    std::cout<<"correct"<<std::endl;
  }
  return 0;
}