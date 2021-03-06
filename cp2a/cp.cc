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
  std::vector<double> sqrtSqureSum(ny,0);

  const int parallel=8;
  int new_nx = ((nx+parallel-1)/parallel)*parallel;
  
  std::vector<double> padding_data(ny*new_nx,0);
  std::vector<double> normalized(ny*new_nx,0);

  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
     padding_data[y*new_nx+x]=data[y*nx+x];
    }
  }

  for (int y=0; y<ny; ++y){
    std::vector<double> temp(parallel,0);
    for (int x=0; x<new_nx; x=x+parallel){
      for (int i=0; i<parallel; ++i){
        temp[i]+=padding_data[y*new_nx+x+i];
      }
    }
    for (int i=0; i<parallel; ++i){
        avg[y]+=temp[i];
    }
    avg[y]/=nx;
  }

  
  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      normalized[y*new_nx+x]=padding_data[y*new_nx+x]-avg[y];
    }
  }

  
  for (int y=0; y<ny; ++y){
    std::vector<double> temp(parallel,0);
    for (int x=0; x<new_nx; x=x+parallel){
      for (int i=0; i<parallel; ++i){
        temp[i]+=pow(normalized[y*new_nx+x+i],2);
      } 
    }
    for (int i=0; i<parallel; ++i){
        sqrtSqureSum[y]+=temp[i];
    }
    sqrtSqureSum[y]=sqrt(sqrtSqureSum[y]);
  }


  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      normalized[y*new_nx+x]/=sqrtSqureSum[y];
    }
  }

  for (int i=0; i<ny; ++i){
    for (int j=0; j<ny; ++j){
      if(j<=i){
        std::vector<double> temp(parallel,0);
        for (int k=0; k<new_nx; k=k+parallel){
          for (int p=0; p<parallel; ++p){
            temp[p]+=normalized[i*new_nx+k+p]*normalized[j*new_nx+k+p];
          }
        }
        double temp_sum=0;
        for (int k=0;k<parallel;k++){
          temp_sum+=temp[k];
        }
        result[i+j*ny]=temp_sum;
      }else continue;
    }
  }
}
