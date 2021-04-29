#include <math.h>
#include <vector>
#include <iostream>
void correlate(int ny, int nx, const float *data, float *result) {
  
  std::vector<double> avg(ny,0);
  std::vector<double> sqrtSqureSum(ny,0);

  const int parallel=4;
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

  // for (double &i: avg){
  //   std::cout<< i <<std::endl;
  // }
  
  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      normalized[y*new_nx+x]=padding_data[y*new_nx+x]-avg[y];
    }
  }
  // for (double &i: normalized){
  //   std::cout<< i <<std::endl;
  // }
  
  for (int y=0; y<ny; ++y){
    std::vector<double> temp(parallel,0);
    for (int x=0; x<new_nx; x=x+parallel){
      for (int i=0; i<parallel; ++i){
        temp[i]+=pow(normalized[y*new_nx+x+i],2);
      } 
      // sqrtSqureSum[y]+=pow(normalized[y*nx+x],2);
    }
    for (int i=0; i<parallel; ++i){
        sqrtSqureSum[y]+=temp[i];
    }
    sqrtSqureSum[y]=sqrt(sqrtSqureSum[y]);
  }
  // for (double &i: sqrtSqureSum){
  //   std::cout<< i <<std::endl;
  // }  


  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      normalized[y*new_nx+x]/=sqrtSqureSum[y];
    }
  }

  for (int i=0; i<ny; ++i){
    for (int j=0; j<ny; ++j){
      if(j<=i){
        std::vector<double> temp(parallel,0);
        // double temp=0;
        for (int k=0; k<new_nx; k=k+parallel){
          for (int p=0; p<parallel; ++p){
            temp[p]+=normalized[i*new_nx+k+p]*normalized[j*new_nx+k+p];
          }
        }
        result[i+j*ny]=temp[0]+temp[1]+temp[2]+temp[3];
      }else continue;
    }
  }
}


int main(){

  int ny=2;
  int nx=2;
  float data[]={-1.0, 1.0, -1.0, 1.0};
  float result[4];
  correlate(ny, nx, data, result);
  std::cout<<"result"<<std::endl;
  for (int i=0;i<4;i++){
    std::cout<<result[i]<<" ";
  }
  return 0;
}