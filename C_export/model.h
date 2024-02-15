#pragma once 
#include "./include/k2c_tensor_include.h" 
void model(k2c_tensor* input_1_input, k2c_tensor* dense_output,float* conv2d_output_array,float* conv2d_kernel_array,float* conv2d_bias_array,float* max_pooling2d_output_array,float* conv2d_1_output_array,float* conv2d_1_kernel_array,float* conv2d_1_bias_array,float* max_pooling2d_1_output_array,float* flatten_output_array,float* dense_kernel_array,float* dense_bias_array); 
void model_initialize(float** conv2d_output_array 
,float** conv2d_kernel_array 
,float** conv2d_bias_array 
,float** max_pooling2d_output_array 
,float** conv2d_1_output_array 
,float** conv2d_1_kernel_array 
,float** conv2d_1_bias_array 
,float** max_pooling2d_1_output_array 
,float** flatten_output_array 
,float** dense_kernel_array 
,float** dense_bias_array 
); 
void model_terminate(float* conv2d_output_array,float* conv2d_kernel_array,float* conv2d_bias_array,float* max_pooling2d_output_array,float* conv2d_1_output_array,float* conv2d_1_kernel_array,float* conv2d_1_bias_array,float* max_pooling2d_1_output_array,float* flatten_output_array,float* dense_kernel_array,float* dense_bias_array); 
