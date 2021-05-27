#include <algorithm>
#include <omp.h>
#include <iostream>
#include <vector>
#include <random>

typedef unsigned long long data_t;

static inline void swap(data_t* a, data_t* b)
{
	data_t t = *a;
	*a = *b;
	*b = t;
}


void quickSort(data_t arr[], int low, int high)
{

    
    int cutoff=10000;
    int interval=high-low;
    

	if (interval>=cutoff)
	{
        constexpr int samples_num=11;
        int split=interval/samples_num;
        int left=low, right=high, i=low;
        std::vector<data_t> samples;
        for (int i=0; i<samples_num; i++){
            samples.push_back(arr[low+split*i]);
        }

        const auto median_it = samples.begin() + samples.size() / 2;
        std::nth_element(samples.begin(), median_it , samples.end());
        data_t pivot = *median_it;


        while(i<=right){
            if(arr[i]<pivot){
                swap(&arr[i],&arr[left]);
                left++;
                i++;
            }else if(arr[i]>pivot){
                swap(&arr[i],&arr[right]);
                right--;
            }else{
                i++;
            }
        }
        #pragma omp taskgroup
        {
            #pragma omp task untied mergeable
            quickSort(arr,low, left-1);
            #pragma omp task untied mergeable
            quickSort(arr, right+1, high);
            // #pragma omp taskyield
        }
		
	}else{
        std::sort(arr+low,arr+high+1);
    }
}



void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of quicksort.
    // int p = omp_get_max_threads();
    #pragma omp parallel
    #pragma omp single
    {
        quickSort(data,0,n-1);
    }
}
