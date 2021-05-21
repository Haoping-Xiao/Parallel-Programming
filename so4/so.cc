#include <algorithm>

typedef unsigned long long data_t;



void TopDownMerge(data_t* A, int iBegin, int iMiddle, int iEnd, data_t* B)
{
    int i = iBegin;
    int j = iMiddle;
    // While there are elements in the left or right runs...
    // #pragma omp parallel for
    for (int k = iBegin; k < iEnd; k++) {
        // If left run head exists and is <= existing right run head.
        if (i < iMiddle && (j >= iEnd || A[i] <= A[j])) {
            B[k] = A[i];
            i = i + 1;
        } else {
            B[k] = A[j];
            j = j + 1;
        }
    }
}


void TopDownSplitMerge(data_t* B, int iBegin,int iEnd, data_t* A){
    if(iEnd - iBegin >=2000){// if run size >=32
        // split the run longer than 1 item into halves
        int iMiddle = (iEnd + iBegin) / 2;              // iMiddle = mid point
        #pragma omp taskgroup
        {
            // recursively sort both runs from array A[] into B[]
            #pragma omp task shared(A,B) untied if(iEnd - iBegin >= (1<<8))
            TopDownSplitMerge(A, iBegin,  iMiddle, B);  // sort the left  run
            #pragma omp task shared(A,B) untied if(iEnd - iBegin >= (1<<8))
            TopDownSplitMerge(A, iMiddle,    iEnd, B);  // sort the right run
            #pragma omp taskyield
        }
        // merge the resulting runs from array B[] into A[]
        TopDownMerge(B, iBegin, iMiddle, iEnd, A);
    }else{
        std::sort(A+iBegin,A+iEnd);
    }           
    
}




void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of merge sort.
    // std::sort(data, data + n);

    data_t* copy =new data_t[n];
    // std::vector<data_t> copy(n);

    #pragma omp parallel for
    for(int i=0;i<n;i++){
        copy[i]=data[i];
    }

    #pragma omp parallel
    #pragma omp single nowait
    TopDownSplitMerge(copy,0,n,data);// sort data from B[] into A[]
    // TopDownSplitMerge(data,0,n);
    delete[] copy;
}
