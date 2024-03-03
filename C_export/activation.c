#include "activation.h"
#include <stdlib.h>
#include<math.h>
#include <stdio.h>

void relu(double **x, const int row, const int col, double alpha)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if (x[i][j] <= 0)
                x[i][j] = alpha * x[i][j];
        }
    }
}

void convolution2D(double **input, double **output, int rows, int cols, double **kernel, int k_rows, int k_cols, double bias)
{
    int i, j, m, n;
    int sum;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            sum = 0;

            for (m = 0; m < k_rows; m++)
            {
                for (n = 0; n < k_cols; n++)
                {
                    int ii = i - k_rows / 2 + m;
                    int jj = j - k_cols / 2 + n;

                    if (ii >= 0 && ii < rows && jj >= 0 && jj < cols)
                    {
                        sum += input[ii][jj] * kernel[m][n];
                    }
                }
            }
            output[i][j] = sum + bias;
        }
    }
}



void maxPooling2D(double **input, int rows, int cols, int poolSize, double **output)
{
    int i, j, m, n;
    int maxVal;

    // Nombre de lignes et de colonnes dans la sortie
    int outRows = (rows + poolSize - 1) / poolSize;
    int outCols = (cols + poolSize - 1) / poolSize;

    for (i = 0; i < rows; i += poolSize)
    {
        for (j = 0; j < cols; j += poolSize)
        {
            maxVal = input[i][j];

            // Déterminer les bornes de la fenêtre de pooling
            int endRow = i + poolSize;
            int endCol = j + poolSize;
            if (endRow > rows) endRow = rows;
            if (endCol > cols) endCol = cols;

            // Parcourir la fenêtre de pooling
            for (m = i; m < endRow; m++)
            {
                for (n = j; n < endCol; n++)
                {
                    if (input[m][n] > maxVal)
                    {
                        maxVal = input[m][n];
                    }
                }
            }

            output[i / poolSize][j / poolSize] = maxVal;
        }
    }
}


void flatten(double ***input, int input_depth, int input_rows, int input_cols, double *output)
{
    int i, j, k, l = 0;
    for (i = 0; i < input_depth; i++)
    {
        for (j = 0; j < input_rows; j++)
        {
            for (k = 0; k < input_cols; k++)
            {
                output[l++] = input[i][j][k];
            }
        }
    }
}


void dense(double *input, int input_size, int *weights, int *biases, int output_size, double *output)
{
    int i, j;

    for (i = 0; i < output_size; i++)
    {
        output[i] = biases[i];

        for (j = 0; j < input_size; j++)
        {
            output[i] += input[j] * weights[i * input_size + j];
        }
    }
}

void softmax(const double *input, const int row, double *output)
{
    double m = input[0];
    for (int i = 0; i < row; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < row; i++) {
        sum += exp(input[i] - m);
    }

    double offset = m + log(sum);
    for (int i = 0; i < row; i++) {
        output[i] = exp(input[i] - offset);
    }
}