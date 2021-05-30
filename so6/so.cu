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

const int MAX_CONSTANT = 16*1024;
__constant__ unsigned int const_sum[MAX_CONSTANT];


template <class T>
void cuda_memcpy(T* target, const T* source, std::size_t num, cudaMemcpyKind direction) {
CHECK(cudaMemcpy(target, source, num * sizeof(T), direction));
}

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
  }

// get the 0 bit of each number by bit_shift
// example: number : 10001, bit_shit: 1, One: 1,
// 
// it means check if the second bit is 1 or not.
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

__global__ void getIndex(unsigned int *d_index, unsigned int *d_sum, unsigned int* d_mask, const int len, const unsigned int n,
    unsigned int total_pre) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    unsigned int start=index*len;
  
    if (start>=n) return;
    

    unsigned int end=start+len;
    for (unsigned int i=start; i<end && i<n; i++){
      d_index[i]=d_mask[i]?d_sum[i]:i-d_sum[i]+total_pre;
      if(d_index[i]>=n){
        printf(" d_sum[i] : %d, total_pre : %d, d_mask[i] : %d \n", d_sum[i], total_pre, d_mask[i]);
      }
      // if(d_mask[i]==1){
      //   d_index[i]=total_pre+d_sum[i];
      // }
    }
}

__global__ void scatter(data_t *d_in, unsigned int *d_index, data_t *d_out, const int len, const unsigned int n) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    unsigned int start=index*len;
    if (start>=n) return;
    unsigned int end=start+len;

    for(unsigned int i=start;i<end && i<n; i++ ){
      d_out[d_index[i]]=d_in[i];
    }
}






// pay attention that blockDim.x must be power of 2
__global__ void blellochScan(unsigned int *out, unsigned int *in,
                              unsigned int *sum, unsigned int inputSize) {
  __shared__ unsigned int temp[2 * 256];
  unsigned int start = blockIdx.x * blockDim.x << 1;
  unsigned int tx = threadIdx.x;
  unsigned int index = 0;
  temp[tx] = (start + tx < inputSize)? in[start+tx]:0;

  temp[tx+blockDim.x] = (start + tx + blockDim.x < inputSize)? in[start + tx + blockDim.x] : 0;

  // Blelloch Scan
  __syncthreads();
  // reduction step
  unsigned int stride = 1;
  while (stride <= blockDim.x) {
    index = (tx + 1) * (stride << 1) - 1;
    if (index < (blockDim.x << 1)) {
      temp[index] += temp[index - stride];
    }
    stride <<= 1;
    __syncthreads();
  }
  // first store the reduction sum in sum array
  // make it zero since it is exclusive scan
  if (tx == 0) {
    // sum array contains the prefix sum of each
    // 2*blockDim blocks of element.
    if (sum != NULL) {
      sum[blockIdx.x] = temp[(blockDim.x << 1) - 1];
    }
    temp[(blockDim.x << 1) - 1] = 0;
  }
  // wait for thread zero to write
  __syncthreads();
  // post scan step
  stride = blockDim.x;
  index = 0;
  unsigned int var = 0;
  while (stride > 0) {
    index = ((stride << 1) * (tx + 1)) - 1;
    if (index < (blockDim.x << 1)) {
      var = temp[index];
      temp[index] += temp[index - stride];
      temp[index - stride] = var;
    }
    stride >>= 1;
    __syncthreads();
  }

  // now write the temp array to output
  if (start + tx < inputSize) {
    out[start + tx] = temp[tx];
  }
  if (start + tx + blockDim.x < inputSize) {
    out[start + tx + blockDim.x] = temp[tx + blockDim.x];
  }
}

/* 
sum out the blocks' accumulated sums to each element
*/
__global__ void mergeScanBlocks(unsigned int *sum, unsigned int *output,
                                unsigned int opSize) {
  unsigned int index = (blockDim.x * blockIdx.x << 1) + threadIdx.x;
  if (index < opSize) {
    // output[index] += sum[blockIdx.x];
    output[index] += (opSize > MAX_CONSTANT)? sum[blockIdx.x]:const_sum[blockIdx.x];
    // output[index] += tex1Dfetch(tex_sum, blockIdx.x);
  }
  if (index + blockDim.x < opSize) {
    // output[index + blockDim.x] += sum[blockIdx.x];
    output[index + blockDim.x] += (opSize > MAX_CONSTANT)? sum[blockIdx.x]:const_sum[blockIdx.x];
    // output[index + blockDim.x] += tex1Dfetch(tex_sum, blockIdx.x);
  }
}

/* 
api for exclusiveScan
*/
void exclusiveScan(unsigned int *out, unsigned int *in, unsigned int in_size, unsigned int block_size) {
  unsigned int numBlocks1 = in_size / block_size;
  if (in_size % block_size) numBlocks1++;
  unsigned int numBlocks2 = numBlocks1 / 2;
  if (numBlocks1 % 2) numBlocks2++;
  dim3 dimThreadBlock;
  dimThreadBlock.x = block_size;
  dimThreadBlock.y = 1;
  dimThreadBlock.z = 1;
  dim3 dimGrid;
  dimGrid.x = numBlocks2;
  dimGrid.y = 1;
  dimGrid.z = 1;

  unsigned int *d_sumArr = NULL;
  if (in_size > (2 * block_size)) {
    // we need the sum auxilarry  array only if nuFmblocks2 > 1
    CHECK(cudaMalloc((void **)&d_sumArr, numBlocks2 * sizeof(unsigned int)));
  }
  blellochScan<<<dimGrid, dimThreadBlock>>>(out, in, d_sumArr, in_size);

  if (in_size <= (2 * block_size)) {
    // out has proper exclusive scan. just return
    CHECK(cudaDeviceSynchronize());
    return;
  } else {
    // now we need to perform exclusive scan on the auxilliary sum array
    unsigned int *d_sumArr_scan;
    CHECK(cudaMalloc((void **)&d_sumArr_scan, numBlocks2 * sizeof(unsigned int)));
    exclusiveScan(d_sumArr_scan, d_sumArr, numBlocks2, block_size);
    // d_sumArr_scan now contains the exclusive scan op of individual blocks
    // now just do a one-one addition of blocks
    // cudaBindTexture(0, tex_sum, d_sumArr_scan, numBlocks2 * sizeof(unsigned int));
    if(numBlocks2 <= MAX_CONSTANT) {
      CHECK(cudaMemcpyToSymbol(const_sum, d_sumArr_scan, numBlocks2 * sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice));
    }
    mergeScanBlocks<<<dimGrid, dimThreadBlock>>>(d_sumArr_scan, out, in_size);
    // cudaUnbindTexture(tex_sum);
    
    cudaFree(d_sumArr);
    cudaFree(d_sumArr_scan);
  }
}




// idea to do exclusive prefix is similar to my ppc course https://www.youtube.com/watch?v=HVhCtl96gUs
// I will use y,z,s to specify which step I am in.
// in particular, I split the whole array into multiple smaller array. each small array has [len] numbers
// Thread level y: each thread will do addition sequentially. threads are working independently, dealing with [len] numbers.
// Thread level z: each threads in the same block will do sequentially. threads are working independently, dealing with one block.
// Thread level s: each thread will add the result from its previous thread. threads are working independently, dealing with [len] numbers.
// Block level y: this will get prefix sum in block level. 
// Block level z: only one block and one thread are used here, do addition sequentially.
// Block level s: each threads will add the result from its previous block.
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

// void serialsum_accrossblock(unsigned int* sum,const int len, const unsigned int n, const int block_size){
//   int step=len*block_size;//each block has step number
//   int start=2*step;
//   for(unsigned int i=start; i<n; i+=step){
//     sum[i]+=sum[i-step];
//   }
// }



__global__ void serialsum_accrossblock(unsigned int* sum,const int len, const unsigned int n, const int block_size){
  //only one block and one thread
  int step=len*block_size;//each block has step number
  int start=2*step;
  for(unsigned int i=start; i<n; i+=step){
    sum[i]+=sum[i-step];
  }
}

// __global__ void mergeblock(unsigned int* sum,const int len, const unsigned int n){
//   unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
//   if (index==0) return;  //the first block is not needed to merge
//   int step=len*blockDim.x;
//   int start=index*step+1; //exclusive
//   int end=start+step-1;// -1 is important, this position has been added in serial sum
//   int base=sum[start-1];//last element at last block
//   for(int i=start; i<end && i<n; i++){
//     sum[i]+=base;
//   }
// }

__global__ void mergeblock(unsigned int* sum,const int len, const unsigned int n){
  if (blockIdx.x==0) return;//the first block is not needed to merge
  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
  int step=len;
  unsigned int base_index=blockIdx.x*step*blockDim.x;
  unsigned int base=sum[base_index];
  int start=index*step; //only the first thread in block should excluded the first element
  int end=start+step;
  start=(start==base_index)?start+1:start;
  
  // int base=sum[start-1];//last element at last block
  
  for(int i=start; i<end && i<n; i++){
    sum[i]+=base;
  }
}

void psort(int n, data_t *data) {
  if(n<=0) return;
  // FIXME: Implement a more efficient parallel sorting algorithm for the GPU.

  const int block_size=256;//64 threads per block;
  const int len=2000; // add 1000 prefix sum per thread; 

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

  // std::vector<unsigned int> inter_sum(n);
  // unsigned int inter_sum[n];

  cuda_memcpy(d_in,data,n,cudaMemcpyHostToDevice);

  data_t bits=sizeof(data_t)*8;

  // unsigned int out[n];
  // unsigned int sum[n];
  unsigned int total_zeros, mask_last;
  //one pass here
  for(data_t i=0; i<bits; i++){
      CHECK(cudaMemset(d_sum,0,n*sizeof(unsigned int)));
      getMask<<<divup(n,block_size*len),block_size>>>(d_in, d_out, len, n, i, 0);
      CHECK(cudaGetLastError());
      // CHECK(cudaMemcpy(out, d_out, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // std::cout<<"out "<<std::endl;
      // for(int j=0;j<n;j++){
      //   std::cout<<out[j]<<" ";
      // }
      // std::cout<<std::endl;


      // //inclusive prefix sum
      // prefixsum<<<divup(n,block_size*len),block_size>>>(d_out,d_sum,len,n);
      // CHECK(cudaGetLastError());
      // serialsum_accrossthread<<<divup(n,block_size*len*block_size),block_size>>>(d_sum,len,n);
      // CHECK(cudaGetLastError());
      // mergethread<<<divup(n,block_size*len),block_size>>>(d_sum,len,n);
      // CHECK(cudaGetLastError());
      // serialsum_accrossblock<<<1,1>>>(d_sum, len, n, block_size);
      // CHECK(cudaGetLastError());
      // // CHECK(cudaMemcpy(inter_sum.data(), d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // // serialsum_accrossblock(inter_sum.data(), len, n, block_size);
      // // CHECK(cudaMemcpy(d_sum, inter_sum.data(),n * sizeof(unsigned int), cudaMemcpyHostToDevice));
      // // CHECK(cudaGetLastError());
      // mergeblock<<<divup(n,block_size*len),block_size>>>(d_sum,len,n);
      // CHECK(cudaGetLastError());


      exclusiveScan(d_sum, d_out, n, block_size);
      // CHECK(cudaMemcpy(sum, d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // std::cout<<"sum "<<std::endl;
      // for(int j=0;j<n;j++){
      //   std::cout<<sum[j]<<" ";
      // }
      // std::cout<<std::endl;
      CHECK(cudaMemcpy(&total_zeros, d_sum+n-1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(&mask_last, d_out+n-1, sizeof(unsigned int), cudaMemcpyDeviceToHost));

      total_zeros+=(mask_last==1)?1:0;

      getIndex<<<divup(n,block_size*len),block_size>>>(d_index, d_sum, d_out, len, n, total_zeros);
      // std::cout<<"index "<<std::endl;
      // CHECK(cudaMemcpy(sum, d_index, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // for(int j=0;j<n;j++){
      //   std::cout<<sum[j]<<" ";
      // }
      // std::cout<<std::endl;
      CHECK(cudaGetLastError());

      // // get mask for 1 and store in d_out
      // getMask<<<divup(n,block_size*len),block_size>>>(d_in, d_out, len, n, i, 1);

      // CHECK(cudaGetLastError());
      // //inclusive prefix sum
      // CHECK(cudaMemset(d_sum,0,n*sizeof(unsigned int)));
      // prefixsum<<<divup(n,block_size*len),block_size>>>(d_out,d_sum,len,n);
      // CHECK(cudaGetLastError());
      // serialsum_accrossthread<<<divup(n,block_size*len*block_size),block_size>>>(d_sum,len,n);
      // CHECK(cudaGetLastError());
      // mergethread<<<divup(n,block_size*len),block_size>>>(d_sum,len,n);
      // CHECK(cudaGetLastError());

      // // CHECK(cudaMemcpy(inter_sum.data() , d_sum, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // // serialsum_accrossblock(inter_sum.data(), len, n, block_size);
      // // CHECK(cudaMemcpy(d_sum, inter_sum.data(),n * sizeof(unsigned int), cudaMemcpyHostToDevice));

      // serialsum_accrossblock<<<1,1>>>(d_sum, len, n, block_size);
      // CHECK(cudaGetLastError());
      // mergeblock<<<divup(n,block_size*len),block_size>>>(d_sum,len,n);
      // CHECK(cudaGetLastError());

      
      // getIndex<<<divup(n,block_size*len),block_size>>>(d_index, d_sum, d_out, len, n, total_zeros);
      // CHECK(cudaGetLastError());


      scatter<<<divup(n,block_size*len),block_size>>>(d_in, d_index, d_out_long, len, n);

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