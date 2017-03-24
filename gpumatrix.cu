/* 
 * NOTE: in file <memory.cuh>, about memory manager
 */
/*
 * Memory manager to use for GPU memory allocations.
 *
 * CUDAMemoryManager: Default Nvidia memory manager; just calls cudaMalloc / cudaFree.
 *                    Allocating and freeing memory is slow.
 * FastMemoryManager: A GPU memory manager with very fast (constant time)
 *                    alloc / free, but possibly more wasteful of memory.
 */

/*
 * Memory manager to use for host memory allocations.
 *
 * CUDAHostMemoryManager: Default Nvidia memory manager; just calls cudaHostAlloc / cudaFreeHost.
 *                        Allocating and freeing memory is slow.
 * FastHostMemoryManager: A host memory manager with very fast (constant time)
 *                        alloc / free, but possibly more wasteful of memory.
 */
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <time.h>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <pthread.h>
#include <complex>
//extern "C" {
//#include "lapacke.h"
//#include "lapacke_mangling.h"
//}
#include <mkl.h>

#include <mmintrin.h> 	// MMX
#include <xmmintrin.h>	// SSE
//#include <smmintrin.h>
//#include <nmmintrin.h>
#include <emmintrin.h>	// SSE2
#include <immintrin.h>	// AVX
//#include <intrin.h>

#include "cuda_common_api.h"
#include "cuda_common.cuh"
#include "gpumatrix.cuh"

using namespace std;
using namespace cv;

//inline int i_div_up(const int & divisor, const int & dividend)
//{
//	return ( divisor % dividend ) ? ( divisor / dividend + 1) : ( divisor / dividend );
//}

///***************************************/
///* Kernel                              */
///***************************************/
//const unsigned int BlockYTransp = 32;
//const unsigned int BlockXTransp = 32;
//template<typename T>
//__global__ void kTranspose(T * id, T * od, const int width, const int height, const int istride, const int ostride)
//{
//	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
//	unsigned int ltidx = threadIdx.x;
//	unsigned int ltidy = threadIdx.y;
//	unsigned int inputIdx = ty * istride + tx;
//	__shared__ T smem[BlockYTransp][BlockXTransp];
//	
//	if(tx >= width || ty >= height) return;
//	
//	smem[ltidy][ltidx] = id[inputIdx];
//	__syncthreads();
//	
//	tx = threadIdx.y + blockIdx.x * blockDim.x;
//	ty = threadIdx.x + blockIdx.y * blockDim.y;
//	unsigned int outputIdx = tx * ostride + ty;
//
//	od[outputIdx] = smem[ltidx][ltidy];
//}
///***************************************/
///* end Kernel                          */
///***************************************/
//
//template<typename T>
//GPUMatrix<T> GPUMatrix<T>::transpose()
//{
//	int numRows_trans = numCols;
//	int numCols_trans = numRows;
//	GPUMatrix<T> trans(numRows_trans, numCols_trans, padding);
//	dim3 threads(BlockXTransp, BlockYTransp, 1);
//	dim3 blocks(i_div_up(numCols, BlockXTransp), i_div_up(numRows, BlockYTransp), 1);
//	kTranspose<T><<<threads, blocks>>>(getDevPtr(), trans.getDevPtr(), numCols, numRows, stride, trans.stride);
//	return trans;	
//}
//
//template<typename T>
//void GPUMatrix<T>::init() {
//	numCols = 0;
//	numRows = 0;
//	numElts = 0;
//	stride = 0;
//	padding = 0;
//	d_ptr = NULL;
//	refCount = NULL;
//}
//
//template<typename T>
//T* GPUMatrix<T>::getDevPtr() const {
//	return d_ptr;
//}
//
//template<typename T>
//int GPUMatrix<T>::getRefCount() const {
//	if(refCount == NULL) return 0;
//	return *refCount;
//}
//
//template<typename T>
//void GPUMatrix<T>::copyToHost(const T * h_ptr, const int h_stride) const
//{
//	cutilSafeCall( cudaMemcpy2D(h_ptr, h_stride * sizeof(T), getDevPtr(), stride * sizeof(T), numCols * sizeof(T), numRows, cudaMemcpyDeviceToHost) );
//}
//
//template<typename T>
//void GPUMatrix<T>::copyFromHost(const T * h_ptr, const int h_stride)
//{
//	cutilSafeCall( cudaMemcpy2D(getDevPtr(), stride * sizeof(T), h_ptr, numCols * sizeof(T), numCols * sizeof(T), numRows, cudaMemcpyHostToDevice) );
//}

#ifdef TEST
//int main(int argc, char * argv[])
//{
//	get_device_property(0);	
//	int numCols = 512;
//	int numRows = 512;
//	GPUMatrix<uint> mat1;
//	GPUMatrix<uint> mat2(numRows, numCols);	
//	printf("INFO: stride = %d\n", mat2.stride);
//	GPUMatrix<uint> mat3(mat2);
//	GPUMatrix<uint> mat4;
//	mat4 = mat3;
//	printf("INFO: ref count = %d\n", mat4.getRefCount());
//	GPUMatrix<uint> mat5 = mat4.transpose();
//	printf("INFO: ref count = %d\n", mat5.getRefCount());
//	return 0;
//}
#endif
