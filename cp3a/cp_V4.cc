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



void correlate(int ny, int nx, const float *data, float *result) {
  //block size
  constexpr int nd=8; 
  //vector per row
  int new_nx=(nx+parallel-1)/parallel;
  //how many blocks of rows
  int nc=(ny+nd-1)/nd;
  //number of rows after padding
  int new_ny=nc*nd;

  double8_t* padding_data=double8_alloc(new_ny*new_nx);

  std::vector<double> avg(ny,0);
  std::vector<double> sqrtSqureSum(ny,0);

  double8_t* normalized=double8_alloc(new_ny*new_nx);

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



  #pragma omp parallel for schedule(static,1)
  for (int i=0; i<nc; ++i){
    for (int j=i; j<nc; ++j){
      double8_t vv[nd][nd];
      for (int id=0; id<nd; ++id){
        for(int jd=0; jd<nd; ++jd){
          vv[id][jd]=d8zero;
        }
      }
      for (int k=0; k<new_nx; ++k){
        double8_t y[nd];
        double8_t x[nd];
        //read 8+8
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
            result[ie*ny+je]=sum(vv[id][jd]);
          }
        }
      }
    }
  }
  std::free(normalized);
}
