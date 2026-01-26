#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "utils.cuh"


__global__ void lut_k(const unsigned char* input, unsigned char* output, const unsigned char* lut_table, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x; //globalne indeksy
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tId = threadIdx.x + threadIdx.y * blockDim.x; // lokalny indeks w bloku

    __shared__ unsigned char shared_lut[256]; // współdzielona pamiec

    if (tId < 256){
        shared_lut[tId] = lut_table[tId]; //z każdego watku jedna wartosc do pamieci współdzielonej, bo lut_table w ram karty (wolne) wiec wrzucamy do współdzielonej
    }

   __syncthreads();

    //Boundry check
    if (x < width && y < height){ 
        int idx = y * width + x;
        unsigned char old_pixel_val = input[idx];
        unsigned char new_pixel_val = shared_lut[old_pixel_val];
        output[idx] = new_pixel_val;
    }

}

torch::Tensor lut(torch::Tensor img, torch::Tensor lut_table){
    TORCH_CHECK(img.device().type() == torch::kCUDA);
    TORCH_CHECK(img.dtype() == torch::kByte);
    TORCH_CHECK(lut_table.device().type() == torch::kCUDA);
    TORCH_CHECK(lut_table.dtype() == torch::kByte);

    img = img.contiguous();
    lut_table = lut_table.contiguous();

    const auto height = img.size(0);
    const auto width = img.size(1);

    dim3 blockDim(16, 16);

    int grid_x = ((width + blockDim.x - 1)/blockDim.x);
    int grid_y = ((height + blockDim.y - 1)/blockDim.y);

    dim3 gridDim(grid_x, grid_y);

    auto result = torch::empty({height, width}, img.options());
    lut_k<<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        lut_table.data_ptr<unsigned char>(),
        width,
        height
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}