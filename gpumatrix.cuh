#ifndef GPUMATRIX_CUH
#define GPUMATRIX_CUH

#include "cuda_common.cuh"

inline int i_div_up(const int & divisor, const int & dividend)
{
	return ( divisor % dividend ) ? ( divisor / dividend + 1) : ( divisor / dividend );
}

typedef unsigned int uint;

template<typename T>
class GPUMatrix {
public:
	void init();
	GPUMatrix() {
		init();
	}
	GPUMatrix(int _numRows, int _numCols, int _padding = 0) {
		init();
		numRows = _numRows;
		numCols = _numCols;
		padding = _padding;
		assert(numRows >= 1 && numCols >= 1 && padding >= 0);
		numElts = numRows * numCols;
//		cutilSafeCall( cudaMalloc((void**)&d_ptr, sizeof(T) * (numCols + padding) * numRows) );
		cutilSafeCall( cudaMallocPitch((void**)&d_ptr, reinterpret_cast<size_t*>(&stride), static_cast<size_t>((numCols + padding) * sizeof(T)), static_cast<size_t>(numRows)) );
		stride /= sizeof(T);
		refCount = new int(1);	
	}
	GPUMatrix(const GPUMatrix& like) {
		init();
		numRows = like.numRows;
		numCols = like.numCols;
		numElts = like.numElts;
		padding = like.padding;
		stride = like.stride;
		d_ptr = like.d_ptr;
		refCount = like.refCount;
		if(refCount != NULL) *refCount += 1;
	}
	GPUMatrix& operator=(const GPUMatrix& like) {
		numRows = like.numRows;
		numCols = like.numCols;
		numElts = like.numElts;
		padding = like.padding;
		stride = like.stride;
		if(d_ptr != like.d_ptr) {
			if(d_ptr != NULL) {
				*refCount -= 1;
				if(*refCount == 0) cutilSafeCall( cudaFree(d_ptr) );
			}
			d_ptr = like.d_ptr;
			refCount = like.refCount;
			if(refCount != NULL) *refCount += 1;
		}
		return *this;
	}
//	GPUMatrix(GPUMatrix&& temp) {
//		init();
//		numRows = temp.numRows;
//		numCols = temp.numCols;
//		numElts = temp.numElts;
//		padding = temp.padding;
//		stride = temp.stride;
//		d_ptr = temp.d_ptr;
//		refCount = temp.refCount;
//		temp.init();
//	}
//	GPUMatrix& operator=(GPUMatrix&& temp) {
//		numRows = temp.numRows;
//		numCols = temp.numCols;
//		numElts = temp.numElts;
//		padding = temp.padding;
//		stride = temp.stride;
//		if(d_ptr != temp.d_ptr) {
//			if(d_ptr != NULL) {
//				*refCount -= 1;
//				if(*refCount == 0) cutilSafeCall( cudaFree(d_ptr) );
//			}
//			d_ptr = temp.d_ptr;
//			refCount = temp.refCount;
//			if(refCount != NULL) *refCount += 1;
//		}
//		temp.init();
//		return *this;		
//	}
	~GPUMatrix() {
		if(refCount != NULL) {
			if(*refCount > 1) *refCount -= 1;
			else if(*refCount == 1) {
				delete refCount;
				cutilSafeCall( cudaFree(d_ptr) );
			}
		}		
	}
	T * getDevPtr() const;
	int getRefCount() const;
	GPUMatrix transpose();
	void copyToHost(const T * h_ptr, const int h_stride) const;
	void copyFromHost(const T * h_ptr, const int h_stride);	
public:
	int numCols, numRows;
	int numElts;
	int padding;
	int stride;
protected:
	T * d_ptr;
	int * refCount;	
};

/***************************************/
/* Kernel                              */
/***************************************/
const unsigned int BlockYTransp = 16;
const unsigned int BlockXTransp = 16;
template<typename T>
__global__ void kTranspose(T * id, T * od, const int width, const int height, const int istride, const int ostride)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int ltidx = threadIdx.x;
	unsigned int ltidy = threadIdx.y;
	unsigned int inputIdx = ty * istride + tx;
	__shared__ T smem[BlockYTransp][BlockXTransp];
	
	if(tx < width && ty < height) {	
		smem[ltidy][ltidx] = id[inputIdx];
	}
	__syncthreads();
	
	tx = threadIdx.y + blockIdx.x * blockDim.x;
	ty = threadIdx.x + blockIdx.y * blockDim.y;
	unsigned int outputIdx = tx * ostride + ty;

	if(tx < width && ty < height) {
		od[outputIdx] = smem[ltidx][ltidy];
	}
}
/***************************************/
/* end Kernel                          */
/***************************************/

template<typename T>
GPUMatrix<T> GPUMatrix<T>::transpose()
{
	int numRows_trans = numCols;
	int numCols_trans = numRows;
	GPUMatrix<T> trans(numRows_trans, numCols_trans, padding);
	dim3 threads(BlockXTransp, BlockYTransp, 1);
	dim3 blocks(i_div_up(numCols, BlockXTransp), i_div_up(numRows, BlockYTransp), 1);
	kTranspose<T><<<blocks, threads>>>(getDevPtr(), trans.getDevPtr(), numCols, numRows, stride, trans.stride);
	return trans;	
}

template<typename T>
void GPUMatrix<T>::init() {
	numCols = 0;
	numRows = 0;
	numElts = 0;
	stride = 0;
	padding = 0;
	d_ptr = NULL;
	refCount = NULL;
}

template<typename T>
T* GPUMatrix<T>::getDevPtr() const {
	return d_ptr;
}

template<typename T>
int GPUMatrix<T>::getRefCount() const {
	if(refCount == NULL) return 0;
	return *refCount;
}

template<typename T>
void GPUMatrix<T>::copyToHost(const T * h_ptr, const int h_stride) const
{
	cutilSafeCall( cudaMemcpy2D((void*)h_ptr, h_stride * sizeof(T), getDevPtr(), stride * sizeof(T), numCols * sizeof(T), numRows, cudaMemcpyDeviceToHost) );
}

template<typename T>
void GPUMatrix<T>::copyFromHost(const T * h_ptr, const int h_stride)
{
	cutilSafeCall( cudaMemcpy2D(getDevPtr(), stride * sizeof(T), h_ptr, numCols * sizeof(T), numCols * sizeof(T), numRows, cudaMemcpyHostToDevice) );
}

#endif
