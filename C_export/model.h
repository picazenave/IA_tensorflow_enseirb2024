#pragma once 
#include "./include/k2c_tensor_include.h" 
void model(k2c_tensor* input_1_input, k2c_tensor* dense_output,double* conv2d_output_array,double* conv2d_kernel_array,double* conv2d_bias_array,double* max_pooling2d_output_array,double* conv2d_1_output_array,double* conv2d_1_kernel_array,double* conv2d_1_bias_array,double* max_pooling2d_1_output_array,double* flatten_output_array,double* dense_kernel_array,double* dense_bias_array); 
void model_initialize(double** conv2d_output_array 
,double** conv2d_kernel_array 
,double** conv2d_bias_array 
,double** max_pooling2d_output_array 
,double** conv2d_1_output_array 
,double** conv2d_1_kernel_array 
,double** conv2d_1_bias_array 
,double** max_pooling2d_1_output_array 
,double** flatten_output_array 
,double** dense_kernel_array 
,double** dense_bias_array 
); 
void model_terminate(double* conv2d_output_array,double* conv2d_kernel_array,double* conv2d_bias_array,double* max_pooling2d_output_array,double* conv2d_1_output_array,double* conv2d_1_kernel_array,double* conv2d_1_bias_array,double* max_pooling2d_1_output_array,double* flatten_output_array,double* dense_kernel_array,double* dense_bias_array); 
