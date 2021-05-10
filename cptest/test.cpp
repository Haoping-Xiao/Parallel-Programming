/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <algorithm>
#include <math.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <immintrin.h>
constexpr int parallel=8;



typedef double double8_t __attribute__ ((vector_size (parallel * sizeof(double))));

constexpr double8_t d8zero{
  0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0
};

static double8_t* double8_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double8_t), sizeof(double8_t) * n)) {
        throw std::bad_alloc();
    }
    return (double8_t*)tmp;
}


static inline double8_t pow8(double8_t vv, const double n) {
    double8_t v;
    for (int i = 0; i < parallel; ++i) {
        v[i] = pow(vv[i], n);
    }
    return v;
}

static inline double sum(double8_t vv) {
    double temp=0;
    for (int i = 0; i < parallel; ++i) {
        temp+=vv[i];
    }
    return temp;
}

static inline void check(double8_t v){
  for (int i = 0; i < parallel; ++i) {
    std::cout<<v[i]<<" ";
  }
  std::cout<<std::endl;
}

void correlate(int ny, int nx, const float *data, float *result) {
  //block size
  constexpr int nd=8; 
  //vector per row
  int new_nx=(nx+parallel-1)/parallel;
  //how many blocks of rows
  int nc=(ny+nd-1)/nd;
  //number of rows after padding
  int new_ny=nc*nd;
  //250 vector per shorter row (2000 elements) if needed
  int shorter= new_nx>1 ?1:new_nx;
  //how many slices we got
  int slice=(new_nx+shorter-1)/shorter;

  std::vector<std::tuple<int,int,int>> rows(nc*nc);

  double8_t* padding_data=double8_alloc(new_ny*new_nx);

  std::vector<double> avg(ny,0);
  std::vector<double> sqrtSqureSum(ny,0);

  double8_t* normalized=double8_alloc(new_ny*new_nx);

  #pragma omp parallel for schedule(static,2)
  for (int y=0; y<ny; ++y){
    for (int x=0; x<new_nx; ++x){
      for (int k=0; k<parallel;++k){
        int old_x=y*nx+x*parallel+k;
        int check=data[old_x];
        int check_bool=(x*parallel+k)<nx;
        padding_data[y*new_nx+x][k]=(x*parallel+k)<nx?data[old_x]:0;
      }
    }
  }

  // for (int y=0; y<new_ny; ++y){
  //   for (int x=0; x<new_nx; ++x){
  //     check(padding_data[y*new_nx+x]);
  //   }
  // }
  #pragma omp parallel for schedule(static,1)
  for (int y=0; y<ny; ++y){
    double8_t temp={0};
    for (int x=0; x<new_nx; x++){
        temp+=padding_data[y*new_nx+x];
    }
    avg[y]=sum(temp)/nx;
  }


  #pragma omp parallel for schedule(static,1)
  for (int y=0; y<ny; ++y){
    int x=0;
    for (x=0; x<new_nx-1; ++x){
        normalized[y*new_nx+x]=padding_data[y*new_nx+x]-avg[y];
    }
    for (int i=0; i<parallel; ++i){
      normalized[y*new_nx+x][i]=(x*parallel+i)<nx? padding_data[y*new_nx+x][i]-avg[y]:padding_data[y*new_nx+x][i];
    }
  }

  for (int y=0; y<new_ny; ++y){
    for (int x=0; x<new_nx; ++x){
      check(normalized[y*new_nx+x]);
    }
  }
  std::free(padding_data);

  #pragma omp parallel for schedule(static,1)
  for (int y=0; y<ny; ++y){
    double8_t temp={0};
    for (int x=0; x<new_nx; x++){
      temp+=pow8(normalized[y*new_nx+x],2);
    }
    sqrtSqureSum[y]=sqrt(sum(temp));
  }

  #pragma omp parallel for collapse(2) schedule(static,1)
  for (int y=0; y<ny; ++y){
    for (int x=0; x<new_nx; ++x){
      //vector / scalar == each element / scalar
      normalized[y*new_nx+x]/=sqrtSqureSum[y];
    }
  }
    for (int y=0; y<new_ny; ++y){
    for (int x=0; x<new_nx; ++x){
      check(normalized[y*new_nx+x]);
    }
  }

  
  #pragma omp parallel for collapse(2)
  for (int ia = 0; ia < nc; ++ia) {
      for (int ja = 0; ja < nc; ++ja) {
          int ija = _pdep_u32(ia, 0x55555555) | _pdep_u32(ja, 0xAAAAAAAA);
          rows[ia*nc + ja] = std::make_tuple(ija, ia, ja);
      }
  }
  std::sort(rows.begin(), rows.end());

  double result_temp[ny*ny]={0};
  // #pragma omp parallel for schedule(static,1)
  for (int s=0; s<slice; s++){
    for(auto row:rows){
      int i=std::get<1>(row);
      int j=std::get<2>(row);
      double8_t vv[nd][nd];
      for (int id=0; id<nd; ++id){
        for(int jd=0; jd<nd; ++jd){
          vv[id][jd]=d8zero;
        }
      }    
      for (int k=s*shorter; k<new_nx && k<(s+1)*shorter; ++k){
        double8_t y[nd];
        double8_t x[nd];
        //read 8+8 rows
        for (int id=0; id<nd; ++id){
          y[id]=normalized[(i*nd+id)*new_nx+k];
          x[id]=normalized[(j*nd+id)*new_nx+k];
        }
        for (int id=0; id<nd; ++id){
          for(int jd=0; jd<nd; ++jd){
            vv[id][jd]+=y[id]*x[jd];
          }
        }
      }

      for (int id=0; id<nd; ++id){
        for(int jd=0; jd<nd; ++jd){
          int ie=i*nd+id;
          int je=j*nd+jd;
          if(ie<ny && je<ny && ie<=je){
            result_temp[ie*ny+je]+=sum(vv[id][jd]);
          }
        }
      }
    }
  }

  #pragma omp parallel for collapse(2)
  for (int i=0; i<ny; i++){
    for (int j=0; j<ny; j++){
      result[i*ny+j]=result_temp[i*ny+j];
    }
  }
  std::free(normalized);
}



int main(){

  // int ny=2;
  // int nx=10;
  // // float data[]={+0.81472367, +0.90579194,
  // // +0.45150527, +0.49610928}; //result: 1 1 0 1

  // float data[]={-1.0, 1.0, 1.0, -1.0, -1.0,1.0, -1.0, -1.0, 1.0, 1.0,
  //               -1.0, 1.0, 1.0, -1.0, -1.0,1.0, -1.0, -1.0, 1.0, 1.0,};
  // float result[4];
  // correlate(ny, nx, data, result);
  // std::cout<<"result"<<std::endl;


  // for (int i=0;i<4;i++){

  //   std::cout<<result[i]<<" ";
  // }

int nc=20;
std::vector<std::tuple<int,int,int>> rows((1+nc)*nc/2);

// #pragma omp parallel for
  for (int ia = 0; ia < nc; ++ia) {
      for (int ja = ia; ja < nc; ++ja) {
          int ija = _pdep_u32(ia, 0x55555555) | _pdep_u32(ja, 0xAAAAAAAA);
          int p= ia*nc-(ia-1)*ia/2;
          std::cout<<p+ja-ia<<std::endl;
          rows[p+(ja-ia)]=std::make_tuple(ija, ia, ja) ;
      }
  }
  std::sort(rows.begin(), rows.end());
  int i=0;
  std::cout<<"result"<<std::endl;
  for(auto row:rows){
    
    std::cout<< i << ","<<std::get<0>(row) << ":"
              << std::get<1>(row) << ","
              << std::get<2>(row) << std::endl;
    i++;
  }

  // double4_t test={0,1,2,3};
  // double4_t sw1=swap1(test);
  // double4_t sw2=swap2(test);
  // for (int i=0;i<4;i++){
  //     std::cout<<test[i]<<" ";
  //   }
  //   std::cout<<"result"<<std::endl;
  //   for (int i=0;i<4;i++){
  //     std::cout<<sw1[i]<<" ";
  //   }
  //   std::cout<<"result"<<std::endl;
  //   for (int i=0;i<4;i++){
  //     std::cout<<sw2[i]<<" ";
  //   }
  
  return 0;
}