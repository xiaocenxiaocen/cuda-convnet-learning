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
#include <vector>

#include <arpa/inet.h>

#include "gpumatrix.cuh"

using namespace std;
using namespace cv;

void batch_load_mnist_data(vector<Mat>& images, vector<uchar>& labels, const char * fimage, const char * flabel)
{
	FILE * imageHandle_t = fopen(fimage, "rb");
	if(imageHandle_t == NULL) {
		fprintf(stderr, "ERROR: cannot open MNIST train image file %s, in file <%s>, line %i\n", fimage, __FILE__, __LINE__);
		exit(0);
	}
	int numImages;
	int numRows;
	int numCols;
	fread(&numImages, sizeof(int), 1, imageHandle_t); // magic number
	fread(&numImages, sizeof(int), 1, imageHandle_t); // number of images
	fread(&numRows, sizeof(int), 1, imageHandle_t); // number of rows
	fread(&numCols, sizeof(int), 1, imageHandle_t); // number of columns
	numImages = ntohl(numImages);
	numRows = ntohl(numRows);
	numCols = ntohl(numCols);
fprintf(stdout, "INFO: numImages = %d, numRows = %d, numCols = %d\n", numImages, numRows, numCols);
	images.clear();
	/* NOTE: images = vector<Mat>(numImages, Mat(width, height, CV_8UC1));
	 * here cv::Mat in vector<Mat> images have the same data pointer, pay attention!
	 */	
	images = vector<Mat>(numImages, Mat());
	for(int image = 0; image < numImages; image++) {
		images[image] = Mat(numRows, numCols, CV_8UC1);
		uchar * dst_ptr = images[image].ptr<uchar>(0);
		fread(dst_ptr, sizeof(uchar), numRows * numCols, imageHandle_t);
	}
	fclose(imageHandle_t);
	FILE * labelHandle_t = fopen(flabel, "rb");
	if(labelHandle_t == NULL) {
		fprintf(stderr, "ERROR: cannot open MNIST train labels file %s, in file <%s>, line %i\n", flabel, __FILE__, __LINE__);
		exit(0);
	}
	int numItems;
	fread(&numItems, sizeof(int), 1, labelHandle_t); // magic number
	fread(&numItems, sizeof(int), 1, labelHandle_t); // number of images
	numItems = ntohl(numItems);
fprintf(stdout, "INFO: numItems = %d\n", numItems);
	assert(numItems == numImages);
	labels.clear();
	labels = vector<uchar>(numItems, 0x0);
	fread(&labels[0], sizeof(uchar), numItems, labelHandle_t);
	fclose(labelHandle_t);
}

void cvt_mnist_images_to_gpumatrix(GPUMatrix<float>& d_mat, const vector<Mat>& images)
{
	int numImages = images.size();	
	int numRows = images[0].cols;
	int numCols = images[0].rows;
	GPUMatrix<float> d_temp(numImages, numRows * numCols);
	float * h_mat;
	cutilSafeCall( cudaMallocHost((void**)&h_mat, sizeof(float) * numRows * numCols * numImages) );
	float * h_ptr = h_mat;
	int h_stride = numRows * numCols; 
	for(int image = 0; image < numImages; image++, h_ptr += h_stride) {
		float * ptr = h_ptr;
		MatConstIterator_<uchar> src_it = images[image].begin<uchar>(), src_end = images[image].end<uchar>();
		for(; src_it != src_end; src_it++, ptr++) {
			*ptr = static_cast<float>(*src_it);
		}
	}
	d_temp.copyFromHost(h_mat, numRows * numCols);

	d_mat = d_temp.transpose();

	cutilSafeCall( cudaFreeHost(h_mat) );
}	

void test_valid(const GPUMatrix<float>& d_mat)
{
	int rows = d_mat.numRows;
	int cols = d_mat.numCols;

	Mat h_images(rows, cols, CV_32FC1);
	d_mat.copyToHost(h_images.ptr<float>(0), cols);

	int imageSizeX, imageSizeY;
	imageSizeX = imageSizeY = static_cast<int>(sqrt(rows));
	vector<Mat> images(cols, Mat());
	for(int i = 0; i < cols; i++) {
		images[i] = Mat(imageSizeY, imageSizeX, CV_8UC1);
		for(int y = 0; y < imageSizeY; y++) {
			for(int x = 0; x < imageSizeX; x++) {
				images[i].at<uchar>(y, x) = saturate_cast<uchar>(h_images.at<float>(y * imageSizeX + x, i));
			}
		} 
	}
	imwrite("test.jpg", images[100]);
}

int main(int argc, char * argv[])
{
	vector<Mat> images;
	vector<uchar> labels;

	batch_load_mnist_data(images, labels, "train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	
	if(argc < 3) {
		fprintf(stdout, "Usage: imageindex outputfile\n");
		exit(0);
	}
	int idx = atoi(argv[1]);
fprintf(stdout, "INFO: image index = %d, label = %d\n", idx, labels[idx]);
	const char * outputimage = argv[2];
	imwrite(outputimage, images[idx]);

	GPUMatrix<float> d_mat;
	cvt_mnist_images_to_gpumatrix(d_mat, images);
	test_valid(d_mat);

	return 0;			
}
