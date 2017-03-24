#ifndef CUDA_COMMON_API_H
#define CUDA_COMMON_API_H

// sys
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// cuda related
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" 
{
// check error
void checkCudaMem(const char * msg, const char * file, const int line, int num_gpus);
void checkCudaError(const char * msg, const char * file, const int line);

// threads
void cuda_threads_sync(void);
void cuda_thread_exit(void);

// device properties
void get_device_count(int * const num);
void get_device_id(int * const id);
void set_device_id(int const device);
void get_device_property(int const device);
int get_compute_major(int const device);
int get_compute_minor(int const device);
int get_multiprocessor_count(int const device);
int get_device_gmemsize(int const device);

// memory
void device_memory_allocate(void **p, long int const ibytes);
void host_memory_allocate(void **p, long int const ibytes);
void cuda_memset(void *p, long int const ibytes, int const content);
void copy_to_device(void * out, void * in, long int const ibytes);
void copy_to_host(void * out, void * in, long int const ibytes);
void copy_device_device(void * out, void * in, long int const ibytes);
void async_copy_to_device(void * out, void * in, long int const ibytes, int streamId);
void async_copy_to_host(void * out, void * in, long int const ibyte, int streamId);
void free_device_memory(void *p);
void free_host_memory(void *p);

// stream
void stream_create(cudaStream_t * p);
void stream_destroy(cudaStream_t p);

// event
void create_event(cudaEvent_t * ievent);
void destroy_event(cudaEvent_t const ievent);
void record_event(cudaEvent_t const ievent);
void sync_event(cudaEvent_t const ievent);
bool event_query(cudaEvent_t const ievent);
void event_elapsed_time(float * gputime, cudaEvent_t const start, cudaEvent_t const stop);
} // end of extern "C"

#endif
