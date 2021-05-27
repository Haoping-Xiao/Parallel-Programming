
#include <vector>
#include <limits>
#include<iostream>
#include <tuple>
#include <algorithm>
struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

constexpr int parallel=4;

typedef double double4_t __attribute__ ((vector_size (parallel * sizeof(double))));

static inline double sum3(double4_t vv) {
    double temp=0;
    for (int i = 0; i < 3; ++i) {
        temp+=vv[i];
    }
    return temp;
}
constexpr double4_t d4zero{
  0.0, 0.0, 0.0, 0.0
};

static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}



/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    double4_t* padding_data=double4_alloc(ny*nx);
    // padding 1 row and 1 column
    double4_t* sum_data=double4_alloc((ny+1)*(nx+1));

    int new_nx=nx+1;
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    // double inf=std::numeric_limits<double>::infinity();
    // double best_error=std::numeric_limits<double>::infinity();
    #pragma omp parallel for collapse(2)
    for(int y=0; y<ny; y++){
        for(int x=0;x<nx;x++){
            padding_data[x+nx*y][0]=data[3*x+3*nx*y];
            padding_data[x+nx*y][1]=data[1+3*x+3*nx*y];
            padding_data[x+nx*y][2]=data[2+3*x+3*nx*y];
            padding_data[x+nx*y][3]=0;
        }
    }
    #pragma omp parallel for
    for(int x=0;x<=nx;x++){
        //y=0
        sum_data[x]=d4zero;
    }
    #pragma omp parallel for 
    for(int y=0;y<=ny;y++){
        //x=0
        sum_data[new_nx*y]=d4zero;
    }

    for(int y1=1; y1<=ny; y1++){
        double4_t temp={0,0,0,0};
        for(int x1=1;x1<=nx;x1++){
            int x=x1-1;
            int y=y1-1;
            temp+=padding_data[x+nx*y];
            sum_data[x1+new_nx*y1]=sum_data[x1+new_nx*y]+temp;
        }
    }


    std::free(padding_data);

    double4_t sum_all=sum_data[nx+(nx+1)*ny];
    double all=nx*ny;
    // errors: error+anchor_y0+anchor_x0+anchor_y1+anchor_x1
    std::vector<std::tuple<double,int,int,int,int>> errors(ny*nx,{0,0,0,0,0});

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int height=1; height<=ny; height++){
        for(int width=1; width<=nx; width++){
            int in=width*height;
            int out=all-in;
            double reverse_in=1.0/in;
            double reverse_out=1.0/out;
            double reverse_sum=reverse_in+reverse_out;
            double4_t temp=reverse_out*sum_all;
             for(int offset_y=0; offset_y<=(ny-height);offset_y++){
                for(int offset_x=0; offset_x<=(nx-width);offset_x++){
                    int anchor_y1=offset_y+height;
                    int anchor_x1=offset_x+width;
                    // int anchor_y0=offset_y;
                    // int anchor_x0=offset_x;
                    int y1x=new_nx*anchor_y1;
                    int yx=new_nx*offset_y;
                    double4_t sum_in=sum_data[anchor_x1+y1x]-sum_data[offset_x+y1x]
                                    -sum_data[anchor_x1+ yx]+sum_data[offset_x+ yx];
                    // double4_t sum_out=sum_all-sum_in;
                    // simplifed sum of square errors: remove const term + Horner's method
                    double4_t cost=temp*sum_all+sum_in*(-2.0*temp+sum_in*reverse_sum);
                    double total_cost=sum3(cost);
                    int index=width-1+nx*(height-1);
                    if(total_cost > std::get<0>(errors[index])){
                        errors[index]=std::make_tuple(total_cost, offset_y,offset_x,anchor_y1,anchor_x1);
                    }
                }
            }
        }
    }
    auto maximum=std::max_element(errors.begin(),errors.end());
    result.y0=std::get<1>(*maximum);
    result.x0=std::get<2>(*maximum);
    result.y1=std::get<3>(*maximum);
    result.x1=std::get<4>(*maximum);

    double4_t inner=sum_data[result.x1+new_nx*result.y1]-sum_data[result.x0+new_nx*result.y1]
                   -sum_data[result.x1+new_nx*result.y0]+sum_data[result.x0+new_nx*result.y0];
    std::free(sum_data);
    double4_t outer=sum_all-inner;
    double in=(result.y1-result.y0)*(result.x1-result.x0);
    double out=all-in;
    for(int i=0; i<3; i++){
        result.inner[i]=inner[i]/in;
        result.outer[i]=outer[i]/out;
    }
    
    return result;
}
