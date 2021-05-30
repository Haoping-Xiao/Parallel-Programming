#include <vector>
#include <algorithm>
#include<iostream>
struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

#define CHECK(x) check(x, #x)


__global__ void scan(int ny, int nx, float* sum_data, float* errors, int* coordinates){
    //same as Is4, fix window size and scan different position
    int width = threadIdx.x + blockIdx.x * blockDim.x + 1; 
    int height = threadIdx.y + blockIdx.y * blockDim.y + 1; 
    
    if(width>nx || height>ny) return;
    int new_nx=nx+1;
    float err_=0;
    int x0=0, y0=0, x1=0, y1=0; 

    float sum_all=sum_data[nx+ny*new_nx];
    int all=nx*ny;
    int in=width*height; //the number of pixels inside window
    int out=all-in;
    float reverse_in=1.0/in;
    float reverse_out=1.0/out;
    float reverse_sum=reverse_in+reverse_out;
    float temp=reverse_out*sum_all;

    for(int offset_y=0; offset_y<=(ny-height);offset_y++){
        for(int offset_x=0; offset_x<=(nx-width);offset_x++){
            int anchor_y1=offset_y+height;
            int anchor_x1=offset_x+width;
            int y1x=new_nx*anchor_y1;
            int yx=new_nx*offset_y;
            float sum_in=sum_data[anchor_x1+y1x]-sum_data[offset_x+y1x]
                        -sum_data[anchor_x1+ yx]+sum_data[offset_x+ yx];
            float cost=temp*sum_all+sum_in*(float(-2.0)*temp+sum_in*reverse_sum);
            if(cost>err_){
                err_=cost;
                x0=offset_x;
                y0=offset_y;
                x1=anchor_x1;
                y1=anchor_y1;
            }
        }
    }
    //same as tuple in Is4
    int w=width-1;
    int h=height-1;
    int i=w+nx*h;
    errors[i]=err_;
    coordinates[4*i+0]=y0;
    coordinates[4*i+1]=x0;
    coordinates[4*i+2]=y1;
    coordinates[4*i+3]=x1;
}

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};


    const int new_nx=nx+1;
    std::vector<float> sum_data((ny+1)*new_nx);//padding 1 column at the left and 1 row at the top
    std::vector<float> err(ny*nx);
    std::vector<int> coord(4*ny*nx);

    for(int x=0;x<=nx;x++){
        //y=0
        sum_data[x]=0;
    }

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


    float sum_all=sum_data[nx+new_nx*ny];
    int all=nx*ny; //the number of pixels


    float* cuda_sum_data=NULL;
    float* errors=NULL;
    int* coordinates=NULL; //same as tuple in is4, pointer to tuple seems hard to pass to gpu.

    CHECK(cudaMalloc((void**)&cuda_sum_data, (ny+1)*new_nx*sizeof(float)));
    CHECK(cudaMalloc((void**)&errors, all*sizeof(float)));
    CHECK(cudaMalloc((void**)&coordinates, 4*all*sizeof(int)));
    CHECK(cudaMemcpy(cuda_sum_data, sum_data.data(), (ny+1)*new_nx*sizeof(float), cudaMemcpyHostToDevice));


    dim3 dimBlock(32,32); 
    dim3 dimGrid(divup(nx, 32), divup(ny, 32));
    scan<<<dimGrid, dimBlock>>>(ny,nx,cuda_sum_data,errors,coordinates);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(err.data(), errors, all*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(coord.data(), coordinates, 4*all*sizeof(int), cudaMemcpyDeviceToHost));


    auto maximum=std::max_element(err.begin(), err.end());
    int i=std::distance(err.begin(), maximum);
    result.y0=coord[4*i+0];
    result.x0=coord[4*i+1];
    result.y1=coord[4*i+2];
    result.x1=coord[4*i+3];

    float inner=sum_data[result.x1+new_nx*result.y1]-sum_data[result.x0+new_nx*result.y1]
               -sum_data[result.x1+new_nx*result.y0]+sum_data[result.x0+new_nx*result.y0];
    float outer=sum_all-inner;
    int in=(result.y1-result.y0)*(result.x1-result.x0);
    int out=all-in;

    for(int i=0; i<3; i++){
        result.inner[i]=inner/in;
        result.outer[i]=outer/out;
    }

    CHECK(cudaFree(cuda_sum_data));
    CHECK(cudaFree(errors));
    CHECK(cudaFree(coordinates));
    return result;
}
