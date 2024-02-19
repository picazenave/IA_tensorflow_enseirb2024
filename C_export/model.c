#include <math.h>
#include <string.h>
#include <stdio.h>
#include "./include/k2c_include.h"
#include "./include/k2c_tensor_include.h"

void model(k2c_tensor *input_1_input, k2c_tensor *dense_output, double *conv2d_output_array, double *conv2d_kernel_array, double *conv2d_bias_array, double *max_pooling2d_output_array, double *conv2d_1_output_array, double *conv2d_1_kernel_array, double *conv2d_1_bias_array, double *max_pooling2d_1_output_array, double *flatten_output_array, double *dense_kernel_array, double *dense_bias_array)
{

	size_t conv2d_stride[2] = {1, 1};
	size_t conv2d_dilation[2] = {1, 1};
	k2c_tensor conv2d_output = {conv2d_output_array, 3, 21632, {26, 26, 32, 1, 1}};
	k2c_tensor conv2d_kernel = {conv2d_kernel_array, 4, 288, {3, 3, 1, 32, 1}};
	k2c_tensor conv2d_bias = {conv2d_bias_array, 1, 32, {32, 1, 1, 1, 1}};

	size_t max_pooling2d_stride[2] = {2, 2};
	size_t max_pooling2d_pool_size[2] = {2, 2};
	k2c_tensor max_pooling2d_output = {max_pooling2d_output_array, 3, 5408, {13, 13, 32, 1, 1}};

	size_t conv2d_1_stride[2] = {1, 1};
	size_t conv2d_1_dilation[2] = {1, 1};
	k2c_tensor conv2d_1_output = {conv2d_1_output_array, 3, 7744, {11, 11, 64, 1, 1}};
	k2c_tensor conv2d_1_kernel = {conv2d_1_kernel_array, 4, 18432, {3, 3, 32, 64, 1}};
	k2c_tensor conv2d_1_bias = {conv2d_1_bias_array, 1, 64, {64, 1, 1, 1, 1}};

	size_t max_pooling2d_1_stride[2] = {2, 2};
	size_t max_pooling2d_1_pool_size[2] = {2, 2};
	k2c_tensor max_pooling2d_1_output = {max_pooling2d_1_output_array, 3, 1600, {5, 5, 64, 1, 1}};

	k2c_tensor flatten_output = {flatten_output_array, 1, 1600, {1600, 1, 1, 1, 1}};
	k2c_tensor dense_kernel = {dense_kernel_array, 2, 16000, {1600, 10, 1, 1, 1}};
	k2c_tensor dense_bias = {dense_bias_array, 1, 10, {10, 1, 1, 1, 1}};
	// double dense_fwork[17600] = {0};

	double *dense_fwork = (double *)malloc(17600 * sizeof(double));

	k2c_conv2d(&conv2d_output, input_1_input, &conv2d_kernel,
			   &conv2d_bias, conv2d_stride, conv2d_dilation, k2c_relu);

	k2c_maxpool2d(&max_pooling2d_output, &conv2d_output, max_pooling2d_pool_size,
				  max_pooling2d_stride);

	k2c_conv2d(&conv2d_1_output, &max_pooling2d_output, &conv2d_1_kernel,
			   &conv2d_1_bias, conv2d_1_stride, conv2d_1_dilation, k2c_relu);

	k2c_maxpool2d(&max_pooling2d_1_output, &conv2d_1_output, max_pooling2d_1_pool_size,
				  max_pooling2d_1_stride);

	k2c_flatten(&flatten_output, &max_pooling2d_1_output);

	k2c_dense(dense_output, &flatten_output, &dense_kernel,
			  &dense_bias, k2c_softmax, dense_fwork);
	free(dense_fwork);
}

void model_initialize(double **conv2d_output_array, double **conv2d_kernel_array, double **conv2d_bias_array, double **max_pooling2d_output_array, double **conv2d_1_output_array, double **conv2d_1_kernel_array, double **conv2d_1_bias_array, double **max_pooling2d_1_output_array, double **flatten_output_array, double **dense_kernel_array, double **dense_bias_array)
{
	*conv2d_output_array = (double*) malloc(21632 * sizeof(double));
	memset(*conv2d_output_array,0,21632);
    // *conv2d_output_array = k2c_read_array("model_v2conv2d_output_array.csv",21632);
    *conv2d_kernel_array = k2c_read_array("model_v2conv2d_kernel_array.csv",288);
    *conv2d_bias_array = k2c_read_array("model_v2conv2d_bias_array.csv",32);
	*max_pooling2d_output_array = (double*) malloc(5408 * sizeof(double));
	memset(*max_pooling2d_output_array,0,5408);
    // *max_pooling2d_output_array = k2c_read_array("model_v2max_pooling2d_output_array.csv",5408);
	*conv2d_1_output_array = (double*) malloc(7744 * sizeof(double));
	memset(*conv2d_1_output_array,0,7744);
    // *conv2d_1_output_array = k2c_read_array("model_v2conv2d_1_output_array.csv",7744);
    *conv2d_1_kernel_array = k2c_read_array("model_v2conv2d_1_kernel_array.csv",18432);
    *conv2d_1_bias_array = k2c_read_array("model_v2conv2d_1_bias_array.csv",64);
	*max_pooling2d_1_output_array = (double*) malloc(1600 * sizeof(double));
	memset(*max_pooling2d_1_output_array,0,1600);
    // *max_pooling2d_1_output_array = k2c_read_array("model_v2max_pooling2d_1_output_array.csv",1600);
	*flatten_output_array = (double*) malloc(1600 * sizeof(double));
	memset(*flatten_output_array,0,1600);
    // *flatten_output_array = k2c_read_array("model_v2flatten_output_array.csv",1600);
    *dense_kernel_array = k2c_read_array("model_v2dense_kernel_array.csv",16000);
    *dense_bias_array = k2c_read_array("model_v2dense_bias_array.csv",10);
}

void model_terminate(double *conv2d_output_array, double *conv2d_kernel_array, double *conv2d_bias_array, double *max_pooling2d_output_array, double *conv2d_1_output_array, double *conv2d_1_kernel_array, double *conv2d_1_bias_array, double *max_pooling2d_1_output_array, double *flatten_output_array, double *dense_kernel_array, double *dense_bias_array)
{

	free(conv2d_output_array);
	free(conv2d_kernel_array);
	free(conv2d_bias_array);
	free(max_pooling2d_output_array);
	free(conv2d_1_output_array);
	free(conv2d_1_kernel_array);
	free(conv2d_1_bias_array);
	free(max_pooling2d_1_output_array);
	free(flatten_output_array);
	free(dense_kernel_array);
	free(dense_bias_array);
}
