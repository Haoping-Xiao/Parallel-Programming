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
#include <iostream>
#include <tuple>
#include <immintrin.h>
#include <algorithm>
constexpr int parallel=16;



typedef float float16_t __attribute__ ((vector_size (parallel * sizeof(float))));

constexpr float16_t f16zero{
  0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0
};

static float16_t* float16_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(float16_t), sizeof(float16_t) * n)) {
        throw std::bad_alloc();
    }
    return (float16_t*)tmp;
}


static inline float16_t pow16(float16_t vv, const double n) {
    float16_t v;
    for (int i = 0; i < parallel; ++i) {
        v[i] = pow(vv[i], n);
    }
    return v;
}

static inline double sum(float16_t vv) {
    double temp=0;
    for (int i = 0; i < parallel; ++i) {
        temp+=vv[i];
    }
    return temp;
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
  //max vector per shorter row
  //1MB / 4 blocks * row in each block * vector_size * double bytes
  int nn=1*1024*1024/(4*nd*parallel*8)-5;
  //how many vector per shorter row if needed
  int shorter= new_nx>nn ?nn:new_nx;
  //how many slices we got
  int slice=(new_nx+shorter-1)/shorter;


  // std::vector<std::tuple<int,int,int>> rows((1+nc)*nc/2);

  float16_t* padding_data=float16_alloc(new_ny*new_nx);

  std::vector<float> avg(ny,0);
  std::vector<float> sqrtSqureSum(ny,0);

  float16_t* normalized=float16_alloc(new_ny*new_nx);

  #pragma omp parallel for schedule(static,2)
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
    float16_t temp={0};
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

  std::free(padding_data);

  #pragma omp parallel for schedule(static,1)
  for (int y=0; y<ny; ++y){
    float16_t temp={0};
    for (int x=0; x<new_nx; x++){
      temp+=pow16(normalized[y*new_nx+x],2);
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

  // std::vector<double> result_temp(ny*ny,0);
  for (int s=0; s<slice; s++){
    #pragma omp parallel for schedule(static,1)
    for (int i=0; i<nc; i++){
      for (int j=i; j<nc; j++){
        float16_t vv[nd][nd];
        for (int id=0; id<nd; ++id){
          for(int jd=0; jd<nd; ++jd){
            vv[id][jd]=f16zero;
          }
        }    
        for (int k=s*shorter; k<new_nx && k<(s+1)*shorter; ++k){
          float16_t y[nd];
          float16_t x[nd];
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
              result[ie*ny+je]+=sum(vv[id][jd]);
            }
          }
        }
      }
    }
  }

  // #pragma omp parallel for
  // for (int i=0; i<ny; i++){
  //   for (int j=i; j<ny; j++){
  //     result[i*ny+j]=result_temp[i*ny+j];
  //   }
  // }
  std::free(normalized);
}

