#ifndef _CUDA_COMMON_CUH_
#define _CUDA_COMMON_CUH_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

inline void __cudaSafeCall( cudaError err, const char * file, const int line)
{
	if( err != cudaSuccess ) {
		fprintf(stderr, "ERROR: cudaSafeCall() Runtime API error in file <%s>, line %i : %s.\n",
			file, line, cudaGetErrorString( err ) );
		exit(-1);
	}
}

inline void __check_cuda_error(const char * msg, const char * file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ) {
		fprintf(stderr, "ERROR: %s Runtime API error in file <%s>, line %i : %s.\n",
			msg, file, line - 1, cudaGetErrorString( err ) );
		exit(EXIT_FAILURE);
	}
}

#define check_cuda_error(msg) __check_cuda_error  (msg, __FILE__, __LINE__)
#define cutilSafeCall(err) __cudaSafeCall         (err, __FILE__, __LINE__)

inline int cudaGetMaxGflopsDeviceId()
{
	int device_count = 0;
	cudaGetDeviceCount( &device_count );
	
	cudaDeviceProp device_properties;
	int max_gflops_device = 0;
	int max_gflops = 0;
	
	int current_device = 0;
	cudaGetDeviceProperties( &device_properties, current_device );
	max_gflops = device_properties.multiProcessorCount * device_properties.clockRate;
	++current_device;
	
	while( current_device < device_count ) {
		cudaGetDeviceProperties( &device_properties, current_device );
		int gflops = device_properties.multiProcessorCount * device_properties.clockRate;
		int predictor = gflops > max_gflops;
		max_gflops = predictor ? gflops : max_gflops;
		max_gflops_device = predictor ? current_device : max_gflops_device;
		++current_device;
	}

	return max_gflops_device;
}

#endif
