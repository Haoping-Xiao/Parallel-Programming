/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

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
              v.push_back(in[(x+delta_x)+nx*(y+delta_y)]);
            }
          }
        }
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
      
    }
  }
}
