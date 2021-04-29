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

void correlate(int ny, int nx, const float *data, float *result) {
  std::vector<double> avg(ny,0);
  std::vector<double> normalized(ny*nx,0);
  std::vector<double> sqrtSqureSum(ny,0);

  #pragma omp parallel for schedule(static,1)
  for (int y=0; y<ny; ++y){
    double temp=0;
    for (int x=0; x<nx; ++x){
      temp+=data[y*nx+x];
    }
    avg[y]=temp/nx;
  }

  #pragma omp parallel for collapse(2) schedule(static,1)
  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      normalized[y*nx+x]=data[y*nx+x]-avg[y];
    }
  }

  #pragma omp parallel for schedule(static,1)
  for (int y=0; y<ny; ++y){
    double temp=0;
    for (int x=0; x<nx; ++x){
      temp+=pow(normalized[y*nx+x],2);
    }
    sqrtSqureSum[y]=sqrt(temp);
  }

  #pragma omp parallel for collapse(2) schedule(static,1)
  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      normalized[y*nx+x]/=sqrtSqureSum[y];
    }
  }

  #pragma omp parallel for collapse(2) schedule(static,1)
  for (int i=0; i<ny; ++i){
    for (int j=0; j<ny; ++j){
      if(j<=i){
        double temp=0;
        for (int k=0; k<nx; ++k){
          temp+=normalized[i*nx+k]*normalized[j*nx+k];
        }
        result[i+j*ny]=temp;
      }else continue;
    }
  }
}
