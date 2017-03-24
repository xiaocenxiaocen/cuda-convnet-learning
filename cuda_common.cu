#include "cuda_common.cuh"
#ifdef VERBOSE
#define verbose 1
#else
#define verbose 0
#endif

extern "C"
{

void checkCudaMem(const char * msg, const char * file, const int line, int num_gpus)
{
	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ) {
		printf("CUDA MEM ERROR: %s: in file <%s>, line %i :%s.\n", msg, file, line, cudaGetErrorString(err));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

void checkCudaError(const char * msg, const char * file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ) {
		printf("CUDA ERROR: %s: in file <%s>, line %i :%s.\n", msg, file, line, cudaGetErrorString(err));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

void cuda_threads_sync(void)
{
	cutilSafeCall( cudaThreadSynchronize() );
	if(verbose) printf("INFO: threads syncthronized\n");
	return;
}

void cuda_thread_exit(void)
{
	cutilSafeCall( cudaThreadExit() );
	if(verbose) printf("INFO: thread exit\n");
	return;
}

// note: the address that num point to can not be modified, but the content of *num can be modified. be careful
void get_device_count(int * const num)
{
	cutilSafeCall( cudaGetDeviceCount(num) );
	if(verbose) printf("INFO: device count = %d\n", *num);
	return;
}

void get_device_id(int * const id)
{
	*id = cudaGetMaxGflopsDeviceId();
	if(verbose) printf("INFO: device id = %d\n", *id);
	return;
}

void set_device_id(int const device)
{	
	cudaSetDevice(device);
	if(verbose) printf("INFO: set device id = %d\n", device);
	return;
}

void get_device_property(int const device)
{
	cudaDeviceProp properties;
	int numDevices;
	
	cutilSafeCall( cudaGetDeviceProperties(&properties, device) );
	cutilSafeCall( cudaGetDeviceCount(&numDevices) );
	
	printf("INFO: Device name:                %s\n", properties.name);
	printf("INFO: Total device:               %d\n", numDevices);
	printf("INFO: Max grid size:              %ld\n", properties.maxGridSize);

	if(properties.totalGlobalMem >= 1024 * 1024 * 1024) 
	printf("INFO: Total GPU Memory:           %.4f GB\n", properties.totalGlobalMem / (1024.f * 1024.f * 1024.f)); 
	else
	printf("INFO: Total GPU Memory:           %.4f MB\n", properties.totalGlobalMem / (1024.f * 1024.f));
	
	printf("INFO: Compute capability:         %d.%d\n", properties.major, properties.minor);
	printf("INFO: Number of multiprocessor:   %d\n\n", properties.multiProcessorCount);

	if(verbose) 
	printf("INFO: get device properties of Device %d\n", device);	
	
printf("device id                                              %d\n", device);

printf("  CUDA Capability Major revision number:               %d\n", properties.major);
printf("  CUDA Capability Minor revision number:               %d\n", properties.minor);
// note: carefully convert integer type to floating point type.
printf("  Total amount of global memory:                       %.0f G bytes\n", properties.totalGlobalMem/(1024.f*1024.f*1024.f));
printf("  Number of streaming multiprocessors:                 %d\n", properties.multiProcessorCount);

int coresPerMultiProcessor = 8;
if(properties.major == 3) {
coresPerMultiProcessor = 16;
}
else if(properties.major == 5) {
coresPerMultiProcessor = 32;
}
else if(properties.major == 6) {
coresPerMultiProcessor = 32;
}
// note: 
printf("  Number of cores:                                     %d\n", properties.multiProcessorCount * coresPerMultiProcessor);
printf("  Total amount of constant memory:                     %.0f K bytes\n", properties.totalConstMem / 1024.f);
printf("  Total amount of shared memory per block:             %.0f K bytes\n", properties.sharedMemPerBlock/1024.f);
printf("  Total number of registers available per block:       %d\n", properties.regsPerBlock);
printf("  Warp size:                                           %d\n", properties.warpSize);
printf("  Maximum number of threads per block:                 %d\n", properties.maxThreadsPerBlock);
printf("  Maximum sizes of each dimension of a block:          %d x %d x %d\n", 
       properties.maxThreadsDim[0],
       properties.maxThreadsDim[1],
       properties.maxThreadsDim[2]);
printf("  Maximum sizes of each dimension of a grid:           %d x %d x %d\n",
       properties.maxGridSize[0],
       properties.maxGridSize[1],
       properties.maxGridSize[2]);
printf("  Maximum memory pitch:                                %u bytes\n", properties.memPitch);
printf("  Texture alignment:                                   %u bytes\n", properties.textureAlignment);
printf("  Clock rate:                                          %.2f GHz\n", properties.clockRate*1e-6f);
printf("  Concurrent copy and execution:                       %s\n", properties.deviceOverlap ? "Yes":"No");
printf("  Run time limit on kernels:                           %s\n", properties.kernelExecTimeoutEnabled == 1 ? "yes" : "no");
printf("  Integrated:                                          %s\n", properties.integrated == 1 ? "yes" : "no");
printf("  Support host page-locked memory mapping:             %s\n", properties.canMapHostMemory == 1 ? "yes" : "no");
printf("  Compute mode:                                        %d\n\n", properties.computeMode);
	
	return;
}

int get_compute_major(int const device)
{
	cudaDeviceProp properties;
	cutilSafeCall( cudaGetDeviceProperties(&properties, device) );
	return properties.major;
}

int get_compute_minor(int const device)
{
	cudaDeviceProp properties;
	cutilSafeCall( cudaGetDeviceProperties(&properties, device) );
	return properties.minor;
}

int get_multiprocessor_count(int const device)
{
	cudaDeviceProp properties;
	cutilSafeCall( cudaGetDeviceProperties(&properties, device) );
	return properties.multiProcessorCount;
}

int get_device_gmemsize(int const device)
{
	cudaDeviceProp properties;
	cutilSafeCall( cudaGetDeviceProperties(&properties, device) );
	return (int)(properties.totalGlobalMem / (1024.f * 1024.f));
}

void device_memory_allocate(void **p, long int const ibytes)
{
	cutilSafeCall( cudaMalloc(p, ibytes) );
	if(verbose) printf("INFO: allocate device memory %lld bytes\n", ibytes);
}

void host_memory_allocate(void **p, long int const ibytes)
{
	cutilSafeCall( cudaMallocHost(p, ibytes) );
	if(verbose) printf("INFO: allocate host memory %lld bytes\n", ibytes);
}

void cuda_memset(void *p, long int const ibytes, int const content)
{
	if(content == 0) cutilSafeCall( cudaMemset(p, 0, ibytes) );
	else if(content == 1) cutilSafeCall( cudaMemset(p, 0xff, ibytes) );
	if(verbose) printf("INFO: cudamemset host memory %lld bytes with %x\n", ibytes, content == 0 ? 0 : 0xff);
}

void copy_to_device(void * out, void * in, long int const ibytes)
{
	cutilSafeCall( cudaMemcpy(out, in, ibytes, cudaMemcpyHostToDevice) );
	if(verbose) printf("INFO: copy host to device %lld bytes\n", ibytes);
}

void copy_to_host(void * out, void * in, long int const ibytes)
{
	cutilSafeCall( cudaMemcpy(out, in, ibytes, cudaMemcpyDeviceToHost) );
	if(verbose) printf("INFO: copy device to host %lld bytes\n", ibytes);
}

void copy_device_device(void * out, void * in, long int const ibytes)
{
	cutilSafeCall( cudaMemcpy(out, in, ibytes, cudaMemcpyDeviceToDevice) );
	if(verbose) printf("INFO: copy device to divece %lld bytes\n", ibytes);	
}

void async_copy_to_device(void * out, void * in, long int const ibytes, int streamId)
{
	cutilSafeCall( cudaMemcpyAsync(out, in, ibytes, cudaMemcpyHostToDevice, (cudaStream_t)streamId) );
	if(verbose) printf("INFO: async copy host to device %lld bytes for stream %d\n", ibytes, streamId);	
}

void async_copy_to_host(void * out, void * in, long int const ibytes, int streamId)
{
	cutilSafeCall( cudaMemcpyAsync(out, in, ibytes, cudaMemcpyDeviceToHost, (cudaStream_t)streamId) );
	if(verbose) printf("INFO: async copy devie to host %lld bytes for stream %d\n", ibytes, streamId);
}

void free_device_memory(void *p)
{
	if(p) cutilSafeCall( cudaFree(p) );
	if(verbose) printf("INFO: free device memory\n");
}

void free_host_memory(void *p)
{
	if(p) cutilSafeCall( cudaFreeHost(p) );
	if(verbose) printf("INFO: free host memory\n");
}

void stream_create(cudaStream_t * p)
{
	cutilSafeCall( cudaStreamCreate((cudaStream_t *)p) );
	if(verbose) printf("INFO: create stream: %d\n", (*p));
}

void stream_destroy(cudaStream_t p)
{
	cutilSafeCall( cudaStreamDestroy((cudaStream_t)p) );
	if(verbose) printf("INFO: destroy stream: %d\n", p);
}

void create_event(cudaEvent_t * ievent)
{
	cutilSafeCall( cudaEventCreate((cudaEvent_t *)ievent) );
	if(verbose) printf("INFO: create cuda event %d\n", (cudaEvent_t)(*ievent));
}

void destroy_event(cudaEvent_t const ievent)
{
	cutilSafeCall( cudaEventDestroy((cudaEvent_t)ievent) );
	if(verbose) printf("INFO: destroy cuda event %d\n", (cudaEvent_t)ievent);
}

void record_event(cudaEvent_t const ievent)
{
	cutilSafeCall( cudaEventRecord((cudaEvent_t)ievent, 0) );
	if(verbose) printf("INFO: record cuda event %d\n", ievent);
}

void sync_event(cudaEvent_t const ievent)
{
	cutilSafeCall( cudaEventSynchronize((cudaEvent_t)ievent) );
	if(verbose) printf("INFO: synchronize cuda event %d\n", ievent);
}

bool event_query(cudaEvent_t const ievent)
{
	return cudaEventQuery((cudaEvent_t) ievent) == cudaErrorNotReady;
}

void event_elapsed_time(float * gputime, cudaEvent_t const start, cudaEvent_t const stop)
{
	cutilSafeCall( cudaEventElapsedTime(gputime, (cudaEvent_t) start, (cudaEvent_t) stop) );
	if(verbose) printf("INFO: start event %d, stop event %d, call event elapsed time\n", start, stop);
}

// end of extern "C"
}
