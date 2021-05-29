#include <algorithm>
#include <iostream>
#include <vector>




typedef unsigned long long data_t;

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

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
  }


  __global__ void getMask(data_t *d_in, unsigned int *d_out, const int len, const unsigned int n, data_t bit_shift, unsigned int One) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    data_t bit = 0;
    data_t one=1;
    data_t shift=one<<bit_shift;
    unsigned int start=index*len;
    if (start>=n) return;
    unsigned int end=start+len;
    for(unsigned int i=start;i<end && i<n; i++ ){
      bit=d_in[i]&shift;
      bit = (bit > 0) ? 1 : 0;
      d_out[i] = (One ? bit : 1 - bit);
    }
}

__global__ void getIndex(unsigned int *d_index, unsigned int *d_sum, unsigned int *d_mask, const int len, const unsigned int n,
    unsigned int total_pre) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    unsigned int start=index*len;
  
    if (start>=n || total_pre==n) return;
    
    unsigned int end=start+len;
    for (unsigned int i=start; i<end && i<n; i++){
      if(d_mask[i]==1){
        d_index[i]=total_pre+d_sum[i];
      }
    }
    // if (index < n) {
    //     if (d_mask[index] == 1) {
    //         d_index[index] = total_pre + d_sum[index];
    //     }
    // }
}
// scatter<<<divup(n,block_size*len),block_size>>>(d_in, d_index, d_out, len, n);
__global__ void scatter(data_t *d_in, unsigned int *d_index, data_t *d_out, const int len, const unsigned int n) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    unsigned int start=index*len;
    if (start>=n) return;
    unsigned int end=start+len;

    for(unsigned int i=start;i<end && i<n; i++ ){
      d_out[d_index[i]]=d_in[i];
    }
    // if (index < n) {
    //     d_out[d_index[index]] = d_in[index];
    // }
}


__global__ void prefixsum(unsigned int* mask, unsigned int* output,const int len, const unsigned int n ){
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

void serialsum_accrossblock(unsigned int* sum,const int len, const unsigned int n, const int block_size){
  int step=len*block_size;//each block has step number
  int start=2*step;
  for(unsigned int i=start; i<n; i+=step){
    sum[i]+=sum[i-step];
  }
}
// __global__ void serialsum_accrossblock(unsigned int* sum,const int len, const unsigned int n){
  

//   unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
//   int step=len*blockDim.x;
//   int offset=2*step-1;
//   unsigned int start= blockDim.x*step*index+offset;
//   unsigned int end= blockDim.x*step*(index+1);
//   for(unsigned int i=start; i<end && i<n; i+=step){
//     sum[i]+=sum[i-step];
//   }
// }

__global__ void mergeblock(unsigned int* sum,const int len, const unsigned int n){
  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index==0) return;  //the first block is not needed to merge

  int step=len*blockDim.x;
  
  int start=index*step+1; //exclusive
  int end=start+step-1;// -1 is important, this position has been added in serial sum
  // int base=sum[blockIdx.x*len*blockDim.x-1];//last element at last block
  int base=sum[start-1];//last element at last block
  for(int i=start; i<end && i<n; i++){
    sum[i]+=base;
  }
}

void psort(int n, data_t *data) {
  if(n<=0) return;
  // FIXME: Implement a more efficient parallel sorting algorithm for the GPU.

  const int block_size=64;//64 threads per block;
  const int len=1000; // add 1000 prefix sum per thread; 

  data_t *d_temp;
  data_t *d_in=NULL;
  CHECK(cudaMalloc((void**)&d_in,n*sizeof(data_t)));

  data_t *d_out_long=NULL;
  CHECK(cudaMalloc((void**)&d_out_long,n*sizeof(data_t)));
  unsigned int *d_out=NULL;
  CHECK(cudaMalloc((void**)&d_out,n*sizeof(unsigned int)));
  unsigned int *d_sum=NULL;
  CHECK(cudaMalloc((void**)&d_sum,n*sizeof(unsigned int)));
  unsigned int *d_index=NULL;
  CHECK(cudaMalloc((void**)&d_index,n*sizeof(unsigned int)));

  std::vector<unsigned int> inter_sum(n);
  // unsigned int inter_sum[n];

  cuda_memcpy(d_in,data,n,cudaMemcpyHostToDevice);

  data_t bits=sizeof(data_t)*8;



  unsigned int total_zeros, mask_last;

  for(data_t i=0; i<bits; i++){
      // get mask for 0 and store in d_out
      // getMask<<<dimGrid, dimBlock>>>(d_in, d_out, n, i, 0);
      CHECK(cudaMemset(d_sum,0,n*sizeof(unsigned int)));
      getMask<<<divup(n,block_size*len),block_size>>>(d_in, d_out, len, n, i, 0);
      // std::cout<<"out"<<std::endl;
      // CHECK(cudaMemcpy(index,d_out, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // for(int j=0; j<n; j++){
      //   std::cout<< index[j] << " " ;
      // }
      // std::cout<< std::endl;

      CHECK(cudaGetLastError());
      //inclusive prefix sum
      
      prefixsum<<<divup(n,block_size*len),block_size>>>(d_out,d_sum,len,n);
      CHECK(cudaGetLastError());
      serialsum_accrossthread<<<divup(n,block_size*len*block_size),block_size>>>(d_sum,len,n);
      CHECK(cudaGetLastError());
      mergethread<<<divup(n,block_size*len),block_size>>>(d_sum,len,n);
      CHECK(cudaGetLastError());
      CHECK(cudaMemcpy(inter_sum.data(), d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      serialsum_accrossblock(inter_sum.data(), len, n, block_size);
      CHECK(cudaMemcpy(d_sum, inter_sum.data(),n * sizeof(unsigned int), cudaMemcpyHostToDevice));
      // serialsum_accrossblock<<<divup(n,block_size*len*block_size*block_size) ,block_size>>>(d_sum,len,n);
      // CHECK(cudaGetLastError());
      mergeblock<<<divup(n,block_size*len),block_size>>>(d_sum,len,n);
      CHECK(cudaGetLastError());

      CHECK(cudaMemcpy(&total_zeros, d_sum+n-1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(&mask_last, d_out+n-1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      total_zeros+=(mask_last==1)?1:0;
      // std::cout<< "zeros" << total_zeros<< std::endl;
      // std::cout<<"sum1"<<std::endl;
      // CHECK(cudaMemcpy(index,d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // for(int j=0; j<n; j++){
      //   std::cout<< index[j] << " " ;
      // }
      // std::cout<< std::endl;

      getIndex<<<divup(n,block_size*len),block_size>>>(d_index, d_sum, d_out, len, n, 0);
      
      CHECK(cudaGetLastError());
      // get mask for 1 and store in d_out
      getMask<<<divup(n,block_size*len),block_size>>>(d_in, d_out, len, n, i, 1);

      // std::cout<<"out"<<std::endl;
      // CHECK(cudaMemcpy(index,d_out, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // for(int j=0; j<n; j++){
      //   std::cout<< index[j] << " " ;
      // }
      // std::cout<< std::endl;

      CHECK(cudaGetLastError());
      //inclusive prefix sum
      CHECK(cudaMemset(d_sum,0,n*sizeof(unsigned int)));
      prefixsum<<<divup(n,block_size*len),block_size>>>(d_out,d_sum,len,n);
      CHECK(cudaGetLastError());
      serialsum_accrossthread<<<divup(n,block_size*len*block_size),block_size>>>(d_sum,len,n);
      CHECK(cudaGetLastError());
      mergethread<<<divup(n,block_size*len),block_size>>>(d_sum,len,n);
      CHECK(cudaGetLastError());

      CHECK(cudaMemcpy(inter_sum.data() , d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      serialsum_accrossblock(inter_sum.data(), len, n, block_size);
      CHECK(cudaMemcpy(d_sum, inter_sum.data(),n * sizeof(unsigned int), cudaMemcpyHostToDevice));
      // serialsum_accrossblock<<<divup(n,block_size*len*block_size*block_size) ,block_size>>>(d_sum,len,n);
      // CHECK(cudaGetLastError());
      mergeblock<<<divup(n,block_size*len),block_size>>>(d_sum,len,n);
      CHECK(cudaGetLastError());
      // std::cout<<"sum2"<<std::endl;
      // CHECK(cudaMemcpy(index,d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // for(int j=0; j<n; j++){
      //   std::cout<< index[j] << " " ;
      // }
      // std::cout<< std::endl;
      
      getIndex<<<divup(n,block_size*len),block_size>>>(d_index, d_sum, d_out, len, n, total_zeros);
      CHECK(cudaGetLastError());
      // std::cout<<"index"<<std::endl;
      // CHECK(cudaMemcpy(index,d_index, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // for(int j=0; j<n; j++){
      //   std::cout<< index[j] << " " ;
      // }
      // std::cout<< std::endl;

      scatter<<<divup(n,block_size*len),block_size>>>(d_in, d_index, d_out_long, len, n);
      // CHECK(cudaMemcpy(cal_result,d_out_long, n * sizeof(data_t), cudaMemcpyDeviceToHost));
      // for(int j=0; j<n; j++){
      //   std::cout<< cal_result[j] << " " ;
      // }
      // std::cout<< std::endl;
      CHECK(cudaGetLastError());
      //must swap pointers
      d_temp = d_in;
      d_in = d_out_long;
      d_out_long = d_temp;
  }

  cuda_memcpy(data, d_in, n, cudaMemcpyDeviceToHost);
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out_long));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_sum));
  CHECK(cudaFree(d_index));
  // std::sort(data, data + n);
}


int main(){

  const data_t n=10000000; //100000 number
  std::vector<data_t> data(n);
  // data_t * data= new data_t[n];
  // data_t data[n];
  // data_t result[n];
  // data_t inter_result[n];
  // data_t *cal_result=new data_t [n];

  
  for (data_t i=0; i<n; i++){
    data[i]=i;
  }

  // for (data_t i=0; i<n; i++){
  //   result[i]=i;
  // }
  psort(n, data.data());


  // for(data_t i=0; i<n; i++){
  //   if(result[i]==data[i]){
  //     std::cout<<i<<"vs"<<data[i]<<std::endl;
  //   }
      
  // }
  // data_t i;
  // for (i=0; i<n; i++){
  //   if(result[i]!=data[i]){
  //     std::cout<<"i: "<< i <<"error!"<<std::endl;
  //     std::cout<<result[i]<<"vs"<<data[i]<<std::endl;
  //     break;
  //   }
  // }
  // if(i==n){
  //   std::cout<<"correct"<<std::endl;
  // }

  return 0;

}