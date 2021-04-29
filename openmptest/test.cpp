#include <iostream>

int main(){

  #pragma omp parallel for
  for (int i=0; i<2; i++){
    int temp=0;
    
    for (int j=0; j<10; j++){
      temp+=1;
      std::cout<<temp<<std::endl;
    }
    
  }
  return 0;
}