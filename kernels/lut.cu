#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "utils.cuh"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>


__global__ void lut_k(const unsigned char* input, unsigned char* output, const unsigned char* lut_table, int width, int height, int channels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tId = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ unsigned char shared_lut[256];

    if (tId < 256){
        shared_lut[tId] = lut_table[tId];
    }

    __syncthreads();

    if (x < width && y < height){ 
        int start_idx = (y * width + x)*channels;
        for (int i = 0; i < channels; ++i){
            int col_idx = start_idx + i;
            unsigned char origin_val = input[col_idx];
            output[col_idx] = shared_lut[origin_val];
        }
    }
}


using Tensor = torch::Tensor;

Tensor lut(Tensor img, Tensor lut_table){

    TORCH_CHECK(img.device().type() == torch::kCUDA);
    TORCH_CHECK(img.dtype() == torch::kByte);
    TORCH_CHECK(lut_table.device().type() == torch::kCUDA);
    TORCH_CHECK(lut_table.dtype() == torch::kByte);

    img = img.contiguous();
    lut_table = lut_table.contiguous();


    auto lut_copy = torch::empty_like(lut_table);

    thrust::device_ptr<unsigned char> src(lut_table.data_ptr<unsigned char>());
    thrust::device_ptr<unsigned char> dst(lut_copy.data_ptr<unsigned char>());

    thrust::copy(
        thrust::device,
        src,
        src + lut_table.numel(),
        dst
    );

    const int height = img.size(0);
    const int width = img.size(1);

    int channels = 1;
    if (img.dim() == 3) {
        channels = img.size(2);
    }

    dim3 blockDim(16, 16);

    auto ceil_div = [](int a, int b) {
        return (a + b - 1) / b;
    };

    int grid_x = ceil_div(width, blockDim.x);
    int grid_y = ceil_div(height, blockDim.y);

    dim3 gridDim(grid_x, grid_y);

    auto result = torch::empty_like(img);
    lut_k<<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        lut_copy.data_ptr<unsigned char>(),   // <- używamy kopii LUT
        width,
        height,
        channels
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return result;
}