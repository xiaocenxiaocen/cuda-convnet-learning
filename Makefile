# Location of the CUDA Toolkit
# CUDA_PATH = /home/xiaocen/Software/cuda/cuda-8.0
CUDA_PATH = /home/zx/Software/cuda-7.5
CC = gcc -O3 -g -Wall -std=c99 -msse2 -msse3 -msse4.1 -msse4.2
CXX = g++ -O3 -g -Wall -std=c++0x -Wno-deprecated -msse2 -msse3 -msse4.1 -msse4.2

NVCC = nvcc -ccbin g++ -Xcompiler -fopenmp

#NVCC = nvcc -ccbin icc -Xcompiler -openmp

CUDA_INCLUDE = $(CUDA_PATH)/include
#CUDA_COMMON_INCLUDE = /home/xiaocen/Software/cuda/samples/NVIDIA_CUDA-8.0_Samples/common/inc
CUDA_COMMON_INCLUDE = $(CUDA_PATH)/samples/common/inc

OPENCV_PATH = /home/zx/Software/opencv
OPENCV_INCLUDE = $(OPENCV_PATH)/include

#LAPACKE_PATH = /home/xiaocen/Software/lapack/lapack_build
#LAPACKE_INCLUDE = $(LAPACKE_PATH)/include
MKL_PATH = /home/zx/intel/composer_xe_2013_sp1/mkl
MKL_INCLUDE = $(MKL_PATH)/include
MKL_FFTW_INCLUDE = $(MKL_PATH)/include/fftw

INCLUDES = -I$(MKL_INCLUDE) -I$(MKL_FFTW_INCLUDE) -I$(CUDA_COMMON_INCLUDE) -I$(CUDA_INCLUDE) -I$(OPENCV_INCLUDE)
#INCLUDES = -I$(MKL_INCLUDE) -I$(MKL_FFTW_INCLUDE) -I$(CUDA_COMMON_INCLUDE) -I$(CUDA_INCLUDE) -I$(OPENCV_INCLUDE) -I$(LAPACKE_INCLUDE)

#GENCODE_FLAGS = -m64 -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS = -m64 -gencode arch=compute_20,code=sm_20
CUDA_FLAGS = --ptxas-options=-v
CFLAGS = $(CUDA_FLAGS)

#CXXFLAGS = -fopenmp -D_OPENMP -D_SSE2 $(INCLUDES)
CXXFLAGS = -fopenmp -D_SSE2 $(INCLUDES)

#LIBRARIES = -L$(OPENCV_PATH)/lib -L$(CUDA_PATH)/lib64 -L$(LAPACKE_PATH)/lib
LIBRARIES = -L$(OPENCV_PATH)/lib -L$(CUDA_PATH)/lib64 -L$(MKL_PATH)/lib/intel64

LDFLAGS = -lcudart -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core  -lm -lpthread -lcudart -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab
#-llapacke -llapack -lcblas -lblas -lgfortran

all: target

#target: gpumatrix_test mnist_batch_loader
target: mnist_batch_loader

#gpumatrix_test: gpumatrix.o cuda_common.o
#	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
mnist_batch_loader: mnist.o cuda_common.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) 

mnist.o: mnist.cu gpumatrix.cuh

%.o : %.cu
	$(NVCC) $(INCLUDES) $(CFLAGS) -DTEST $(GENCODE_FLAGS) -o $@ -c $<

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $<

.c.o:
	$(CC) -c $(CXXFLAGS) $<


.PHONY: clean
clean:
	-rm -f *.o
	-rm -f gpumatrix_test
