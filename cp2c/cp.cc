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

void correlate(int ny, int nx, const float *data, float *result) {
  constexpr int parallel=4;
  //vector per row
  int new_nx=(nx+parallel-1)/parallel;

  double4_t* padding_data=double4_alloc(ny*new_nx);
  std::vector<double> avg(ny,0);
  std::vector<double> sqrtSqureSum(ny,0);

  double4_t* normalized=double4_alloc(ny*new_nx);

  for (int y=0; y<ny; ++y){
    for (int x=0; x<new_nx; ++x){
      for (int k=0; k<parallel;++k){
        int old_x=y*nx+x*parallel+k;
        padding_data[y*new_nx+x][k]=(x*parallel+k)<nx?data[old_x]:0;
      }
    }
  }


  for (int y=0; y<ny; ++y){
    double4_t temp={0};
    for (int x=0; x<new_nx; x++){
        temp+=padding_data[y*new_nx+x];
    }
    avg[y]=sum(temp)/nx;
  }


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

  std::free(padding_data);

  for (int y=0; y<ny; ++y){
    double4_t temp={0};
    for (int x=0; x<new_nx; x++){
      temp+=pow4(normalized[y*new_nx+x],2);
    }

    sqrtSqureSum[y]=sqrt(sum(temp));

  }

  for (int y=0; y<ny; ++y){
    for (int x=0; x<new_nx; ++x){
      //vector / scalar == each element / scalar
      normalized[y*new_nx+x]/=sqrtSqureSum[y];
    }
  }

  for (int i=0; i<ny; ++i){
    for (int j=i; j<ny; ++j){
      double4_t temp={0};
      for (int k=0; k<new_nx; ++k){
        temp+=normalized[i*new_nx+k]*normalized[j*new_nx+k];
      }
      result[j+i*ny]=sum(temp);
    }
  }
  std::free(normalized);
}
