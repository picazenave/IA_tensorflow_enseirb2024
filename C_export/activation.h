#ifndef ACTIVATION_H
#define ACTIVATION_H

void relu(double **x, const int row, const int col, double alpha);
void convolution2D(double **input, double **output, int rows, int cols, double **kernel, int k_rows, int k_cols, double bias);
void maxPooling2D(double **input, int rows, int cols, int poolSize, double **output);
void flatten(double ***input, int input_depth, int input_rows, int input_cols, double *output);
void dense(double *input, int input_size, int *weights, int *biases, int output_size, double *output);
void softmax(const double *input, const int row, double *output);
#endif //ACTIVATION_H