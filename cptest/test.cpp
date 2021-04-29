#include <math.h>
#include <vector>
#include <iostream>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}


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

static inline void check(double4_t v){
  for (int i = 0; i < 4; ++i) {
    std::cout<<v[i]<<" ";
  }
  std::cout<<std::endl;
}

void correlate(int ny, int nx, const float *data, float *result) {
  constexpr int parallel=4;
  //vector per row
  int new_nx=(nx+parallel-1)/parallel;

  double4_t* padding_data=double4_alloc(ny*new_nx);
  std::vector<double> avg(ny,0);
  std::vector<double> sqrtSqureSum(ny,0);
  // double4_t* avg=double4_alloc(ny);
  // double4_t* sqrtSqureSum=double4_alloc(ny);
  double4_t* normalized=double4_alloc(ny*new_nx);

  for (int y=0; y<ny; ++y){
    for (int x=0; x<new_nx; ++x){
      for (int k=0; k<parallel;++k){
        int old_x=y*nx+x*parallel+k;
        padding_data[y*new_nx+x][k]=(x*parallel+k)<nx?data[old_x]:0;
      }
    }
  }

  // for (int i=0;i<ny*new_nx;i++){
  //   check(padding_data[i]);
  // }

  for (int y=0; y<ny; ++y){
    double4_t temp={0,0,0,0};
    // std::vector<double> temp(parallel,0);
    for (int x=0; x<new_nx; x++){
        temp+=padding_data[y*new_nx+x];
    }
    avg[y]=sum(temp)/nx;
  }
  //   for (int i=0;i<ny;i++){
  //   std::cout<<avg[i]<<" ";
  // }


  for (int y=0; y<ny; ++y){
    int x=0;
    for (x=0; x<new_nx-1; ++x){
      //vector - scalar == each element - scalar
        normalized[y*new_nx+x]=padding_data[y*new_nx+x]-avg[y];
    }
    for (int i=0; i<parallel; ++i){
      normalized[y*new_nx+x][i]=(x*parallel+i)<nx? padding_data[y*new_nx+x][i]-avg[y]:padding_data[y*new_nx+x][i];
    }
  }

  // std::cout<<"nor"<<std::endl;
  //  for (int i=0;i<ny*new_nx;i++){
  //   check(normalized[i]);
  // } 
  std::free(padding_data);

  for (int y=0; y<ny; ++y){
    double4_t temp={0,0,0,0};
    for (int x=0; x<new_nx; x++){
      temp+=pow4(normalized[y*new_nx+x],2);
    }


    // double temp_sum=sum(temp);
    sqrtSqureSum[y]=sqrt(sum(temp));
    // for (int i=0; i<parallel; ++i){
    //     sqrtSqureSum[y][i]=sqrt(temp_sum);
    // }
  }

  for (int y=0; y<ny; ++y){
    for (int x=0; x<new_nx; ++x){
      normalized[y*new_nx+x]/=sqrtSqureSum[y];
    }
  }

  // std::free(sqrtSqureSum);
  for (int i=0; i<ny; ++i){
    for (int j=i; j<ny; ++j){
      double4_t temp={0,0,0,0};
      for (int k=0; k<new_nx; ++k){
        temp+=normalized[i*new_nx+k]*normalized[j*new_nx+k];
      }
      result[j+i*ny]=sum(temp);
    }
  }
  std::free(normalized);
}

int main(){

  int ny=2;
  int nx=2;
  float data[]={+0.81472367, +0.90579194,
  +0.45150527, +0.49610928}; //result: 1 1 0 1

  // float data[]={-1.0, 1.0, -1.0, 1.0};
  float result[4];
  correlate(ny, nx, data, result);
  std::cout<<"result"<<std::endl;


  for (int i=0;i<4;i++){

    std::cout<<result[i]<<" ";
  }

  
  return 0;
}