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

// void correlate(int ny, int nx, const float *data, float *result) {
//   // initialize avg array with 0
//   double *avg = new double[ny]();
//   for (int y=0; y<ny; ++y){
//     double temp=0;
//     for (int x=0; x<nx; ++x){
//       temp+=data[y*nx+x];
//     }
//     avg[y]=temp/nx;
//   }


//   for (int i=0; i<ny; ++i){
//       for (int j=0; j<ny; ++j){
//         if(j<=i){
//           double numerator=0;
//           double denominator1=0;
//           double denominator2=0;
//           for(int k=0; k<nx;++k){
//             numerator+=(data[i*nx+k]-avg[i])*(data[j*nx+k]-avg[j]);
//             denominator1+=pow(data[i*nx+k]-avg[i],2);
//             denominator2+=pow(data[j*nx+k]-avg[j],2);
//           }
//           result[i+j*ny]=numerator/(sqrt(denominator1)*sqrt(denominator2));
//         }else continue;
//       }
//     }
//   // release memory
//   delete []avg;
// }

void correlate(int ny, int nx, const float *data, float *result) {
  // initialize avg array with 0
  double *avg = new double[ny]();
  std::vector<std::vector<double>> normalized(ny,std::vector<double>(nx,0));
  std::vector<double> sqrtSqureSum(ny,0);

  for (int y=0; y<ny; ++y){
    double temp=0;
    for (int x=0; x<nx; ++x){
      temp+=data[y*nx+x];
    }
    avg[y]=temp/nx;
  }
  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      normalized[y][x]=data[y*nx+x]-avg[y];
    }
  }

  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      sqrtSqureSum[y]+=pow(normalized[y][x],2);
    }
    sqrtSqureSum[y]=sqrt(sqrtSqureSum[y]);
  }

  for (int y=0; y<ny; ++y){
    for (int x=0; x<nx; ++x){
      normalized[y][x]/=sqrtSqureSum[y];
    }
  }

  for (int i=0; i<ny; ++i){
    for (int j=0; j<ny; ++j){
      if(j<=i){
        double temp=0;
        for (int k=0; k<nx; ++k){
          temp+=normalized[i][k]*normalized[j][k];
        }
        result[i+j*ny]=temp;
      }else continue;

    }
  }
  // release memory
  delete []avg;
}