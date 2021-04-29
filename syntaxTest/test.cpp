

#include <iostream>
#include <vector>
#include <algorithm>

void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {

  for(int y=0; y<ny; y++){
    for(int x=0; x<nx; x++){
      std::vector<double> v;
      for (int delta_y=-hy; delta_y<=hy; delta_y++){
        if(y+delta_y>=0 && y+delta_y<ny){
          for (int delta_x=-hx; delta_x<=hx; delta_x++){
            if(x+delta_x>=0 && x+delta_x<nx){
              double temp=in[(x+delta_x)+nx*(y+delta_y)];
              v.push_back(in[(x+delta_x)+nx*(y+delta_y)]);
            }
          }
        }
      }
      for (auto n : v) {
        std::cout << n << ", ";
      }
      
      if(v.size()%2!=0){
        std::nth_element(v.begin(),v.begin()+v.size()/2, v.end());
        out[x+nx*y]=v[v.size()/2];
      }else{
        std::nth_element(v.begin(),v.begin()+v.size()/2, v.end());
        double second=v[v.size()/2];
        std::nth_element(v.begin(),v.begin()+v.size()/2-1, v.end());
        out[x+nx*y]=(v[v.size()/2-1]+second)/2.0;
      }
      std::cout<< std::endl;
    }
  }
}

int main(){
  int ny=5;
  int nx=5;
  int hy=1;
  int hx=3;
  double x[5][6]={};
  // double *avg = new double[ny]();
  const float in[]={+0.81472367, +0.13547695, +0.90579188, +0.83500856, +0.12698680,
  +0.96886772, +0.91337585, +0.22103399, +0.63235921, +0.30816704,
  +0.09754038, +0.54722059, +0.27849817, +0.18838197, +0.54688150,
  +0.99288130, +0.95750678, +0.99646127, +0.96488851, +0.96769488,
  +0.15761304, +0.72583896, +0.97059274, +0.98110968, +0.95716691};
  float out[25];
  mf(ny,nx,hy,hx,in,out);
  
  return 0;
}