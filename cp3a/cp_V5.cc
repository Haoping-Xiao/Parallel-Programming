/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <math.h>
#include <vector>
#include <cstdlib>
#include <x86intrin.h>
#include <iostream>
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 0b0101); }
static inline double4_t swap2(double4_t x) { return _mm256_permute4x64_pd(x, 0b01001110); }

static inline double4_t pow4(double4_t vv, const double n) {
    double4_t v;
    for (int i = 0; i < 4; ++i) {
        v[i] = pow(vv[i], n);
    }
    return v;
}

static inline double sum(double4_t vv) {
    double temp=0;
    for (int i = 0; i < 4; ++i) {
        temp+=vv[i];
    }
    return temp;
}

// static inline void check(double4_t v){
//   for (int i = 0; i < 4; ++i) {
//     std::cout<<v[i]<<" ";
//   }
//   std::cout<<std::endl;
// }

void correlate(int ny, int nx, const float *data, float *result) {
  constexpr int parallel=4;
  int new_nx=(nx+parallel-1)/parallel;
  int new_ny=(ny+parallel-1)/parallel;
  double4_t* padding_data=double4_alloc(ny*new_nx);
  std::vector<double> avg(ny,0);
  std::vector<double> sqrtSqureSum(ny,0);

  double4_t* normalized=double4_alloc(ny*new_nx);
  double4_t* column_normalized=double4_alloc(nx*new_ny);


  #pragma omp parallel for collapse(2)
  for (int y=0; y<ny; ++y){
    for (int x=0; x<new_nx; ++x){
      for (int k=0; k<parallel;++k){
        int old_x=y*nx+x*parallel+k;
        padding_data[y*new_nx+x][k]=(x*parallel+k)<nx?data[old_x]:0;
      }
    }
  }


  #pragma omp parallel for schedule(static,1)
  for (int y=0; y<ny; ++y){
    double4_t temp={0,0,0,0};
    for (int x=0; x<new_nx; x++){
        temp+=padding_data[y*new_nx+x];
    }
    avg[y]=sum(temp)/nx;
  }


  #pragma omp parallel for schedule(static,1)
  for (int y=0; y<ny; ++y){
    for (int x=0; x<new_nx-1; ++x){
        normalized[y*new_nx+x]=padding_data[y*new_nx+x]-avg[y];
    }
    for (int i=0; i<parallel; ++i){
      int x=new_nx-1;
      int index=y*new_nx+x;
      normalized[index][i]=(x*parallel+i)<nx? padding_data[index][i]-avg[y]:padding_data[index][i];
    }
  }

  

  std::free(padding_data);

  #pragma omp parallel for schedule(static,1)
  for (int y=0; y<ny; ++y){
    double4_t temp={0,0,0,0};
    for (int x=0; x<new_nx; x++){
      temp+=pow4(normalized[y*new_nx+x],2);
    }

    sqrtSqureSum[y]=sqrt(sum(temp));

  }

  #pragma omp parallel for collapse(2) schedule(static,1)
  for (int y=0; y<ny; ++y){
    for (int x=0; x<new_nx; ++x){
      normalized[y*new_nx+x]/=sqrtSqureSum[y];
    }
  }


  // #pragma omp parallel for collapse(3) schedule(static,1)
  // for(int x=0; x<nx; ++x){
  //   for(int y=0; y<new_ny; ++y){
  //     for(int k=0; k<parallel; ++k){
  //       if(y*parallel+k<ny ){
  //         transpose_normalized[x*new_ny+y][k]=normalized[(y*parallel+k)*new_nx+x/4][x%4];
  //       }else{
  //         transpose_normalized[x*new_ny+y][k]=0;
  //       }
  //     }
  //   }
  // }
  #pragma omp parallel for collapse(2) schedule(static,1)
  for (int y=0; y<new_ny;++y){
    for (int x=0; x<nx; ++x){
      for (int k=0; k<parallel; ++k){
        if(y*parallel+k<ny){
          column_normalized[y*nx+x][k]=normalized[(y*parallel+k)*new_nx+x/4][x%4];
        }else{
          column_normalized[y*nx+x][k]=0;
        }
      }
    }
  }

  // for (int y=0; y<ny; ++y){
  //   for (int x=0; x<new_nx; ++x){
  //     check(normalized[y*new_nx+x]);
  //   }
  // }
  
  // for(int x=0; x<nx; ++x){
  //   for(int y=0; y<new_ny; ++y){
  //     check(column_normalized[x*new_ny+y]);
  //   }
  // }


  std::free(normalized);





  #pragma omp parallel for schedule(static,1)
  for (int ia=0; ia<new_ny; ia++){
    for (int ja=ia; ja<new_ny; ja++){
      
      double4_t temp1={0,0,0,0};
      double4_t temp2={0,0,0,0};
      double4_t temp3={0,0,0,0};
      double4_t temp4={0,0,0,0};
      // double4_t temp5={0,0,0,0};
      // double4_t temp6={0,0,0,0};
      // double4_t temp7={0,0,0,0};
      // double4_t temp8={0,0,0,0};
      double4_t temp9={0,0,0,0};
      double4_t temp10={0,0,0,0};
      double4_t temp11={0,0,0,0};
      double4_t temp12={0,0,0,0};
      for (int k=0; k<nx; k++){
        double4_t v1=column_normalized[ia*nx+k];
        double4_t v2=column_normalized[ja*nx+k];
        double4_t v11=swap1(v1);
        double4_t v12=swap2(v1);
        double4_t v21=swap1(v2);
        // double4_t v22=swap2(v2);
        temp1+= v1*v1;
        temp2+= v1*v11;
        temp3+= v1*v12;
        temp4+= v11*v12;
        // temp5+= v2*v2;
        // temp6+= v2*v21;
        // temp7+= v2*v22;
        // temp8+= v21*v22;
        temp9+= v1*v2;
        temp10+= v1*v21;
        temp11+= v12*v2;
        temp12+= v12*v21;
      }
      temp3=swap2(temp3);
      temp4=swap2(temp4);
      // temp7=swap2(temp7);
      // temp8=swap2(temp8);
      temp11=swap2(temp11);
      temp12=swap2(temp12);
      double4_t temp[]={ temp9, temp10, temp11, temp12};
      double4_t temp_[]={temp1, temp2, temp3, temp4 };
      // double4_t temp__[]={temp5, temp6, temp7, temp8};
      for(int ib=0; ib<parallel; ib++){
        for(int jb=0; jb<parallel; jb++){
          int i=ia*parallel+ib;
          int j=ja*parallel+jb;
          int i1=ia*parallel+jb;
          int j1=ia*parallel+(ib^jb);
          if(i<ny && j<ny && j>=i){
            result[i*ny+j]=temp[ib^jb][ib];
          }
          if((i1!=i || j1!=j) && i1<ny && j1<ny && j1>=i1){
            result[i1*ny+j1]=temp_[ib][jb];
          }
        }
      }
      // for(int ib=0; ib<parallel; ib++){
      //   for(int jb=0; jb<parallel; jb++){
      //     int i=ia*parallel+jb;
      //     int j=ia*parallel+(ib^jb);
      //     if(i<ny && j<ny && j>=i){
      //       result[i*ny+j]=temp_[ib][jb];
      //     }
      //   }
      // }


    }


  }
  
  std::free(column_normalized);
}