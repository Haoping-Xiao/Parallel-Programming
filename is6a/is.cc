
#include <vector>
#include<iostream>
#include <cmath>
#include <immintrin.h>
struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

constexpr int parallel=8;// 8 windows at one scan

// typedef double double4_t __attribute__ ((vector_size (parallel * sizeof(double))));
typedef float float8_t __attribute__((vector_size(parallel * sizeof(float))));


static inline float max8(float8_t vec){
    float v1= vec[0]<=vec[1] ? vec[1] : vec[0]; // any comparison with Nan is false, Nan is alway at the right padding, and vec[0] is always not Nan 
    float v2= vec[2]<=vec[3] ? vec[3] : vec[2]; 
    float v3= vec[4]<=vec[5] ? vec[5] : vec[4]; 
    float v4= vec[6]<=vec[7] ? vec[7] : vec[6]; 
    float v12= v1<=v2 ? v2 :v1; 
    float v34= v3<=v4 ? v4 :v3; 
    return v12<v34?v34:v12;
}

// constexpr double4_t d4zero{
//   0.0, 0.0, 0.0, 0.0
// };
constexpr float8_t f8zero{
  0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0
};




/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {

    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    const int new_nx=nx+parallel; // padding additional 8 columns. one af the left, seven at the right
    std::vector<float> sum_data((ny+1)*new_nx);

    #pragma omp parallel for
    for(int x=0;x<=nx;x++){
        //y=0
        sum_data[x]=0;
    }
    #pragma omp parallel for 
    for(int y=0;y<=ny;y++){
        //x=0
        sum_data[new_nx*y]=0;
    }

    for(int y1=1; y1<=ny; y1++){
        float temp=0;
        for(int x1=1;x1<=nx;x1++){
            int x=x1-1;
            int y=y1-1;
            temp+=data[x*3+3*nx*y]; //c1=c2=c3, so c1 is enough.
            sum_data[x1+new_nx*y1]=sum_data[x1+new_nx*y]+temp;
        }
    }

    #pragma omp parallel for
    for(int y1=0; y1<=ny; y1++){
        for(int x1=nx+1; x1<new_nx; x1++){
            sum_data[x1+new_nx*y1]=std::sqrt(-1);//padding seven Nan values on the right.
        }
    }

    float sum_all=sum_data[nx+new_nx*ny];
        
    float err_best=0;
    int w=0, h=0; // record window position and size when reach optimality.
    int all=nx*ny; //the number of pixels

    #pragma omp parallel
    {

        float err_=0; //best err per thread
        float8_t err_8w=f8zero;// 8 windows' errors
        float err_w =0; // best error across 8 windows
        int w_=0, h_=0; //record best window size

        #pragma omp for schedule(dynamic)
        for(int height=1; height<=ny; height++){
            for(int offset_y=0; offset_y<=(ny-height);offset_y++){
                for(int width=1; width<=nx; width++){
                    int in=width*height; //the number of pixels inside window
                    int out=all-in;
                    float reverse_in=1.0/in;
                    float reverse_out=1.0/out;
                    float reverse_sum=reverse_in+reverse_out;
                    float temp=reverse_out*sum_all;
                    for(int offset_x=0; offset_x<=(nx-width);offset_x+=parallel){
                        int anchor_y1=offset_y+height;
                        int anchor_x1=offset_x+width;

                        int y1x=new_nx*anchor_y1;
                        int yx=new_nx*offset_y;

                        __m256 x1y1 = _mm256_loadu_ps(&sum_data[anchor_x1+y1x]);//load 8 values into vector, this will result in consecutive 8 windows
                        __m256 x0y1 = _mm256_loadu_ps(&sum_data[offset_x+y1x]);
                        __m256 x1y0 = _mm256_loadu_ps(&sum_data[anchor_x1+ yx]);
                        __m256 x0y0 = _mm256_loadu_ps(&sum_data[offset_x+ yx]);
                        
                        float8_t sum_in=x1y1-x0y1-x1y0+x0y0;
                        float8_t cost=temp*sum_all+sum_in*(float(-2.0)*temp+sum_in*reverse_sum);
                        err_8w=cost>err_8w?cost:err_8w; // any calculation with Nan will get Nan, any comparison with Nan will get false
                    }
                    err_w=max8(err_8w);
                    if(err_w>err_){
                        
                        err_ = err_w;// in this thread, this window size got best error
                        w_ = width;
                        h_ = height;
                    }
                }
            }
        }
        // each thread finish their calculation, now try to update shared global data
        #pragma omp critical
        {
            if(err_>err_best){
                err_best=err_;
                w=w_;
                h=h_;
            }
        }
    }

    int in=w*h;
    int out=all-in;
    float reverse_in=1.0/in;
    float reverse_out=1.0/out;
    float reverse_sum=reverse_in+reverse_out;
    float temp=reverse_out*sum_all;

    #pragma omp for collapse(2) 
    for(int offset_y=0; offset_y<=(ny-h);offset_y++){
        for(int offset_x=0; offset_x<=(nx-w);offset_x++){
            int anchor_y1=offset_y+h;
            int anchor_x1=offset_x+w;

            int y1x=new_nx*anchor_y1;
            int yx=new_nx*offset_y;

            float sum_in=sum_data[anchor_x1+y1x]-sum_data[offset_x+y1x]
                        -sum_data[anchor_x1+ yx]+sum_data[offset_x+ yx];
            float cost=temp*sum_all+sum_in*(float(-2.0)*temp+sum_in*reverse_sum);
            #pragma omp critical
            {
                if(cost>=err_best){
                    result.x0=offset_x;
                    result.y0=offset_y;
                    result.x1=anchor_x1;
                    result.y1=anchor_y1;
                }
            }
        }
    }

    

    float sum_in=sum_data[result.x1+new_nx*result.y1]-sum_data[result.x0+new_nx*result.y1]
                -sum_data[result.x1+new_nx*result.y0]+sum_data[result.x0+new_nx*result.y0];

    float sum_out=sum_all-sum_in;
    for(int i=0; i<3; i++){
        result.inner[i]=sum_in/in;
        result.outer[i]=sum_out/out;
    }
    
    return result;
}
