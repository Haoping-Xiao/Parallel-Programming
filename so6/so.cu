#include <algorithm>
#include <iostream>
#include <vector>
#include <time.h>

#define MAX_BLOCK_SZ 128
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

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


__global__
void gpu_add_block_sums(data_t* const d_out,
    const data_t* const d_in,
    data_t* const d_block_sums,
    const size_t numElems)
{
    data_t d_block_sum_val = d_block_sums[blockIdx.x];

    // Simple implementation's performance is not significantly (if at all)
    //  better than previous verbose implementation
    data_t cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (cpy_idx < numElems)
    {
        d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
        if (cpy_idx + blockDim.x < numElems)
            d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
    }
}


__global__
void gpu_prescan(data_t* const d_out,
    const data_t* const d_in,
    data_t* const d_block_sums,
    const data_t len,
    const data_t shmem_sz,
    const data_t max_elems_per_block)
{
    // Allocated on invocation
    extern __shared__ data_t s_out[];

    int thid = threadIdx.x;
    int ai = thid;
    int bi = thid + blockDim.x;

    // Zero out the shared memory
    // Helpful especially when input size is not power of two
    s_out[thid] = 0;
    s_out[thid + blockDim.x] = 0;
    // If CONFLICT_FREE_OFFSET is used, shared memory size
    //  must be a 2 * blockDim.x + blockDim.x/num_banks
    s_out[thid + blockDim.x + (blockDim.x >> LOG_NUM_BANKS)] = 0;
    
    __syncthreads();
    
    // Copy d_in to shared memory
    // Note that d_in's elements are scattered into shared memory
    //  in light of avoiding bank conflicts
    data_t cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
    if (cpy_idx < len)
    {
        s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
        if (cpy_idx + blockDim.x < len)
            s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
    }

    // For both upsweep and downsweep:
    // Sequential indices with conflict free padding
    //  Amount of padding = target index / num banks
    //  This "shifts" the target indices by one every multiple
    //   of the num banks
    // offset controls the stride and starting index of 
    //  target elems at every iteration
    // d just controls which threads are active
    // Sweeps are pivoted on the last element of shared memory

    // Upsweep/Reduce step
    int offset = 1;
    for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_out[bi] += s_out[ai];
        }
        offset <<= 1;
    }

    // Save the total sum on the global block sums array
    // Then clear the last element on the shared memory
    if (thid == 0) 
    { 
        d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 
            + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
        s_out[max_elems_per_block - 1 
            + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
    }

    // Downsweep step
    for (int d = 1; d < max_elems_per_block; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();

        if (thid < d)
        {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            data_t temp = s_out[ai];
            s_out[ai] = s_out[bi];
            s_out[bi] += temp;
        }
    }
    __syncthreads();

    // Copy contents of shared memory to global memory
    if (cpy_idx < len)
    {
        d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
        if (cpy_idx + blockDim.x < len)
            d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
    }
}
 
void sum_scan_blelloch(data_t* const d_out,
    const data_t* const d_in,
    const size_t numElems)
{
    // Zero out d_out
    CHECK(cudaMemset(d_out, 0, numElems * sizeof(data_t)));

    // Set up number of threads and blocks
    
    data_t block_sz = MAX_BLOCK_SZ / 2;
    data_t max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

    // If input size is not power of two, the remainder will still need a whole block
    // Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
    //data_t grid_sz = (data_t) std::ceil((double) numElems / (double) max_elems_per_block);
    // UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically  
    //  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
    data_t grid_sz = numElems / max_elems_per_block;
    // Take advantage of the fact that integer division drops the decimals
    if (numElems % max_elems_per_block != 0) 
        grid_sz += 1;

    // Conflict free padding requires that shared memory be more than 2 * block_sz
    data_t shmem_sz = max_elems_per_block + ((max_elems_per_block) >> LOG_NUM_BANKS);

    // Allocate memory for array of total sums produced by each block
    // Array length must be the same as number of blocks
    data_t* d_block_sums;
    CHECK(cudaMalloc(&d_block_sums, sizeof(data_t) * grid_sz));
    CHECK(cudaMemset(d_block_sums, 0, sizeof(data_t) * grid_sz));

    // Sum scan data allocated to each block
    //gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(data_t) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
    gpu_prescan<<<grid_sz, block_sz, sizeof(data_t) * shmem_sz>>>(d_out, 
                                                                    d_in, 
                                                                    d_block_sums, 
                                                                    numElems, 
                                                                    shmem_sz,
                                                                    max_elems_per_block);

    // Sum scan total sums produced by each block
    // Use basic implementation if number of total sums is <= 2 * block_sz
    //  (This requires only one block to do the scan)
    if (grid_sz <= max_elems_per_block)
    {
        data_t* d_dummy_blocks_sums;
        CHECK(cudaMalloc(&d_dummy_blocks_sums, sizeof(data_t)));
        CHECK(cudaMemset(d_dummy_blocks_sums, 0, sizeof(data_t)));
        //gpu_sum_scan_blelloch<<<1, block_sz, sizeof(data_t) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
        gpu_prescan<<<1, block_sz, sizeof(data_t) * shmem_sz>>>(d_block_sums, 
                                                                    d_block_sums, 
                                                                    d_dummy_blocks_sums, 
                                                                    grid_sz, 
                                                                    shmem_sz,
                                                                    max_elems_per_block);
        CHECK(cudaFree(d_dummy_blocks_sums));
    }
    // Else, recurse on this same function as you'll need the full-blown scan
    //  for the block sums
    else
    {
        data_t* d_in_block_sums;
        CHECK(cudaMalloc(&d_in_block_sums, sizeof(data_t) * grid_sz));
        CHECK(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(data_t) * grid_sz, cudaMemcpyDeviceToDevice));
        sum_scan_blelloch(d_block_sums, d_in_block_sums, grid_sz);
        CHECK(cudaFree(d_in_block_sums));
    }
    
    // Add each block's total sum to its scan output
    // in order to get the final, global scanned array
    gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

    CHECK(cudaFree(d_block_sums));
}




__global__ void gpu_radix_sort_local(data_t* d_out_sorted,
  data_t* d_prefix_sums,
  data_t* d_block_sums,
  data_t input_shift_width,
  data_t* d_in,
  data_t d_in_len,
  data_t max_elems_per_block)
{
  // need shared memory array for:
  // - block's share of the input data (local sort will be put here too)
  // - mask outputs
  // - scanned mask outputs
  // - merged scaned mask outputs ("local prefix sum")
  // - local sums of scanned mask outputs
  // - scanned local sums of scanned mask outputs

  // for all radix combinations:
  //  build mask output for current radix combination
  //  scan mask ouput
  //  store needed value from current prefix sum array to merged prefix sum array
  //  store total sum of mask output (obtained from scan) to global block sum array
  // calculate local sorted address from local prefix sum and scanned mask output's total sums
  // shuffle input block according to calculated local sorted addresses
  // shuffle local prefix sums according to calculated local sorted addresses
  // copy locally sorted array back to global memory
  // copy local prefix sum array back to global memory

  extern __shared__ data_t shmem[];
  data_t* s_data = shmem;
  // s_mask_out[] will be scanned in place
  data_t s_mask_out_len = max_elems_per_block + 1;
  data_t* s_mask_out = &s_data[max_elems_per_block];
  data_t* s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
  data_t* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
  data_t* s_scan_mask_out_sums = &s_mask_out_sums[4];

  data_t thid = threadIdx.x;

  // Copy block's portion of global input data to shared memory
  data_t cpy_idx = max_elems_per_block * blockIdx.x + thid;
  if (cpy_idx < d_in_len)
      s_data[thid] = d_in[cpy_idx];
  else
      s_data[thid] = 0;

  __syncthreads();

  // To extract the correct 2 bits, we first shift the number
  //  to the right until the correct 2 bits are in the 2 LSBs,
  //  then mask on the number with 11 (3) to remove the bits
  //  on the left
  data_t t_data = s_data[thid];
  data_t t_2bit_extract = (t_data >> input_shift_width) & 3;

  for (data_t i = 0; i < 4; ++i)
  {
      // Zero out s_mask_out
      s_mask_out[thid] = 0;
      if (thid == 0)
          s_mask_out[s_mask_out_len - 1] = 0;

      __syncthreads();

      // build bit mask output(0 to 3, same as input data in paper)
      bool val_equals_i = false;
      if (cpy_idx < d_in_len)
      {
          val_equals_i = t_2bit_extract == i;
          s_mask_out[thid] = val_equals_i;
      }
      __syncthreads();

      // Scan mask outputs (Hillis-Steele)
      int partner = 0;
      data_t sum = 0;
      data_t max_steps = (data_t) log2f(max_elems_per_block);
      for (data_t d = 0; d < max_steps; d++) {
          partner = thid - (1 << d);
          if (partner >= 0) {
              sum = s_mask_out[thid] + s_mask_out[partner];
          }
          else {
              sum = s_mask_out[thid];
          }
          __syncthreads();
          s_mask_out[thid] = sum;
          __syncthreads();
      }

      // Shift elements to produce the same effect as exclusive scan
      data_t cpy_val = 0;
      cpy_val = s_mask_out[thid];
      __syncthreads();
      s_mask_out[thid + 1] = cpy_val;
      __syncthreads();

      if (thid == 0)
      {
          // Zero out first element to produce the same effect as exclusive scan
          s_mask_out[0] = 0;
          data_t total_sum = s_mask_out[s_mask_out_len - 1];
          s_mask_out_sums[i] = total_sum;
          d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
      }
      __syncthreads();

      if (val_equals_i && (cpy_idx < d_in_len))
      {
          s_merged_scan_mask_out[thid] = s_mask_out[thid];
      }

      __syncthreads();
  }//end loop here. complete local prefix sum

  // Scan mask output sums
  // Just do a naive scan since the array is really small
  if (thid == 0)
  {
      data_t run_sum = 0;
      for (data_t i = 0; i < 4; ++i)
      {
          s_scan_mask_out_sums[i] = run_sum;
          run_sum += s_mask_out_sums[i];
      }
  }// use s_scan_mask_out_sums for local shuffle, get index for each input data(0~3)

  __syncthreads();

  if (cpy_idx < d_in_len)
  {
      // Calculate the new indices of the input elements for sorting
      data_t t_prefix_sum = s_merged_scan_mask_out[thid];
      data_t new_pos = t_prefix_sum + s_scan_mask_out_sums[t_2bit_extract];
      
      __syncthreads();

      // Shuffle the block's input elements to actually sort them
      // Do this step for greater global memory transfer coalescing
      //  in next step
      s_data[new_pos] = t_data; //0~3
      s_merged_scan_mask_out[new_pos] = t_prefix_sum;
      
      __syncthreads();

      // Copy block - wise prefix sum results to global memory
      // Copy block-wise sort results to global 
      d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thid];
      d_out_sorted[cpy_idx] = s_data[thid];
  }
}

__global__ void gpu_glbl_shuffle(data_t* d_out,
  data_t* d_in,
  data_t* d_scan_block_sums,
  data_t* d_prefix_sums,
  data_t input_shift_width,
  data_t d_in_len,
  data_t max_elems_per_block)
{
  // d_scan_block_sums is prefix block sum
  // d_prefix_sums is local prefix sum

  data_t thid = threadIdx.x;
  data_t cpy_idx = max_elems_per_block * blockIdx.x + thid;

  if (cpy_idx < d_in_len)
  {
      data_t t_data = d_in[cpy_idx];
      data_t t_2bit_extract = (t_data >> input_shift_width) & 3;
      data_t t_prefix_sum = d_prefix_sums[cpy_idx];
      data_t data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x]
          + t_prefix_sum; // max pos is less than 100,000,000 in our test case, data_t is sufficient
      __syncthreads();
      d_out[data_glbl_pos] = t_data;
  }
}




__global__ void order_checking_local(data_t* d_in, data_t* d_out, data_t d_in_len, data_t max_elems_per_block)
{

    extern __shared__ data_t shmem[]; //dynamic shared memory 
                                      //https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
                                      //its length is (max_elems_per_block+ 1)+ max_elems_per_block
                                      // 1 is the first element in the next block, the last max_elems_per_block is comparison result
    
    data_t* s_data=shmem;
    data_t* s_comparison=&s_data[max_elems_per_block+1];


    data_t thid=threadIdx.x;// from 0 to max_elems_per_block-1

     // Copy block's portion of global input data to shared memory
     // one thread for one number
    data_t cpy_idx = max_elems_per_block * blockIdx.x + thid;

    
    
    if(cpy_idx<d_in_len)
        s_data[thid]=d_in[cpy_idx];
    else
        s_data[thid]=d_in[d_in_len-1]+(cpy_idx-d_in_len);//padding ensure is greater or equal to the previous one
    
    if(thid==0){
        data_t next_cpy_idx= max_elems_per_block *(blockIdx.x+1);//the first element in the next block
        if(next_cpy_idx<d_in_len)
            s_data[max_elems_per_block]=d_in[next_cpy_idx];
        else
            s_data[max_elems_per_block]=d_in[d_in_len-1]+(next_cpy_idx-d_in_len);//padding ensure is greater or equal to the previous one
    }
    //Wait for all threads to finish reading
    __syncthreads();

    //Perform order checking
    s_comparison[thid]=s_data[thid]>s_data[thid+1];
    //Wait for all threads to finish reading
    __syncthreads();

    //Perform reduction sum
    //Scan comparison result (Hillis-Steele)
    int partner=0;
    data_t sum=0;
    data_t max_steps = (data_t) log2f(max_elems_per_block);

    for (data_t d = 0; d < max_steps; d++) {
        partner = thid - (1 << d);
        if (partner >= 0) {
            sum = s_comparison[thid] + s_comparison[partner];
        }
        else {
            sum = s_comparison[thid];
        }
        __syncthreads();
        s_comparison[thid] = sum;
        __syncthreads();
    }
    //the last element of s_comparison is an exclusive sum
    //if 0, then the current block is sorted
    d_out[blockIdx.x]=s_comparison[max_elems_per_block-1];
}


bool partial_order_checking(data_t* d_in, data_t d_in_len)
{   //checing if d_in from 0 to d_in_len-1 is sorted

    const int block_size=MAX_BLOCK_SZ;//64 threads per block;
    const int len=block_size; // max_elems_per_block
    const int grid_size=divup(d_in_len,len);
    data_t shmem_sz =(len+len+1)*sizeof(data_t);//(max_elems_per_block+ 1)+ max_elems_per_block

    data_t *d_out=NULL;
    CHECK(cudaMalloc((void**)&d_out,grid_size*sizeof(data_t)));// each block will yeild one result, sorted or not.

    order_checking_local<<<grid_size, block_size, shmem_sz>>>(d_in, d_out, d_in_len, len);

    std::vector <data_t> d_out_cpu(grid_size);
    cuda_memcpy(d_out_cpu.data(), d_out, grid_size, cudaMemcpyDeviceToHost);

    //The all_of algorithm has the added benefit that it may exit early
    //if one of the elements isn't 0 and save some unnecessary checking.
    //if all elements are 0, then sorted
    bool sorted = std::all_of(d_out_cpu.begin(), d_out_cpu.end(), [](data_t i) { return i==0; });
    CHECK(cudaFree(d_out));
    return sorted;
}

bool order_checking(data_t* d_in, data_t d_in_len)
{
    //do the first half checking
    data_t half=d_in_len/2;
    if(partial_order_checking(d_in,half))
        return partial_order_checking(d_in+half-1,d_in_len-half+1);
    else
        return false;
}


void psort(int n, data_t *data) {
  if(n<=0) return;
  // FIXME: Implement a more efficient parallel sorting algorithm for the GPU.

  const int block_size=MAX_BLOCK_SZ;//64 threads per block;
  const int len=block_size; // max_elems_per_block
  const int grid_size=divup(n,len);

  data_t *d_in=NULL;
  CHECK(cudaMalloc((void**)&d_in,n*sizeof(data_t)));
  cuda_memcpy(d_in,data,n,cudaMemcpyHostToDevice);


  data_t *d_out=NULL;
  CHECK(cudaMalloc((void**)&d_out,n*sizeof(data_t)));

  data_t* d_prefix_sums; //local prefix sum

  CHECK(cudaMalloc((void**)&d_prefix_sums,n*sizeof(data_t)));
  CHECK(cudaMemset(d_prefix_sums, 0, n*sizeof(data_t)));



  data_t* d_block_sums; //block sum in the paper
  data_t d_block_sums_len = 4 * grid_size; // 4-way split
  CHECK(cudaMalloc(&d_block_sums, sizeof(data_t) * d_block_sums_len));
  CHECK(cudaMemset(d_block_sums, 0, sizeof(data_t) * d_block_sums_len));


  data_t* d_scan_block_sums;//prefix block sum in the paper
  CHECK(cudaMalloc(&d_scan_block_sums, sizeof(data_t) * d_block_sums_len)); 
  CHECK(cudaMemset(d_scan_block_sums, 0, sizeof(data_t) * d_block_sums_len));

  data_t s_data_len = len;
  data_t s_mask_out_len = len + 1;
  data_t s_merged_scan_mask_out_len = len;
  data_t s_mask_out_sums_len = 4; // 4-way split
  data_t s_scan_mask_out_sums_len = 4;
  data_t shmem_sz = (s_data_len 
                          + s_mask_out_len
                          + s_merged_scan_mask_out_len
                          + s_mask_out_sums_len
                          + s_scan_mask_out_sums_len)
                          * sizeof(data_t);//share memory size
  
  
//   clock_t cpu_startTime;
  for (data_t shift_width = 0; shift_width <= sizeof(data_t)*8 ; shift_width += 2)
  {     
      
    //   cpu_startTime=clock();
      if(order_checking(d_in,n))
      {
        //early stop if sorted, based on experiments, oder checking is indeed inexpensive.
        //but it usually save 1 iteration only in our test case.
        std::cout<<"order checking save "<< (64-shift_width)/2+1 <<" iteration" << std::endl; 
        std::cout<< "early stopping..." <<std::endl;
        break;
      }
    //   std::cout<< "order checking spend: "<<(clock()-cpu_startTime)/CLOCKS_PER_SEC <<"s"<<std::endl;
      gpu_radix_sort_local<<<grid_size, block_size, shmem_sz>>>(d_out, 
                                                                d_prefix_sums, 
                                                                d_block_sums, 
                                                                shift_width, 
                                                                d_in, 
                                                                n, 
                                                                len);
      
      // scan global block sum array
      // prefix block sum in the paper
      sum_scan_blelloch(d_scan_block_sums, d_block_sums, d_block_sums_len);

      // scatter/shuffle block-wise sorted array to final positions
      gpu_glbl_shuffle<<<grid_size, block_size>>>(d_in, 
                                                  d_out, 
                                                  d_scan_block_sums, 
                                                  d_prefix_sums, 
                                                  shift_width, 
                                                  n, 
                                                  len);
  }


  cuda_memcpy(data, d_in, n, cudaMemcpyDeviceToHost);

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_scan_block_sums));
  CHECK(cudaFree(d_block_sums));
  CHECK(cudaFree(d_prefix_sums));
  // std::sort(data, data + n);
}