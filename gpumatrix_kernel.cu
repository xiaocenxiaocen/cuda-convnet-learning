#include "gpumatrix.cuh"

#include "cuda_common.cuh"
#include "cuda_common_api.h"

template<typename T>
__global__ void kTranspose(T * id, T * od, const int width, const int height)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int ltidx = threadIdx.x;
	unsigned int ltidy = threadIdx.y;
	unsigned int inputIdx = ty * width + tx;
	__shared__ T smem[BlockYTransp][BlockXTransp];
	
	if(tx >= width || ty >= height) return;
	
	smem[ltidy][ltidx] = id[inputIdx];
	__syncthreads();
	
	tx = threadIdx.y + blockIdx.x * blockDim.x;
	ty = threadIdx.x + blockIdx.y * blockDim.y;
	unsigned int outputIdx = tx * height + ty;

	od[outputIdx] = smem[ltidx][ltidy];
}
