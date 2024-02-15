#include <stdio.h>
#include <stdlib.h>

#include "Bmp2Matrix.h"
#include "k2c_tensor_include.h"
#include "model.h"

int main(int argc, char *argv[])
{
    BMP bitmap;
    FILE *pFichier = NULL;

    pFichier = fopen("0_1.bmp", "rb"); // Ouverture du fichier contenant l'image
    if (pFichier == NULL)
    {
        printf("%s\n", "0_1.bmp");
        printf("Erreur dans la lecture du fichier\n");
    }
    LireBitmap(pFichier, &bitmap);
    fclose(pFichier); // Fermeture du fichier contenant l'image

    ConvertRGB2Gray(&bitmap);
    printf("%d\n", bitmap.mPixelsGray[10][10]);

    int im_size = bitmap.infoHeader.hauteur * bitmap.infoHeader.largeur;
    printf("bitmap(l/h):%d/%d;im_size=%d\n", bitmap.infoHeader.largeur,bitmap.infoHeader.hauteur,im_size);
    float *bitmap_div = (float *)malloc(im_size * sizeof(float));
    printf("bitmap_div malloc OK\r\n");
    for (int i = 0; i < im_size; i++)
    {
        int j=(int)(i / bitmap.infoHeader.largeur);
        //printf("bitmap(i/j):%d/%d\n",i-j*bitmap.infoHeader.largeur,j);
        bitmap_div[i] = bitmap.mPixelsGray[i-j*bitmap.infoHeader.largeur][j] / 255.f;
    }
    printf("bitmap_div for OK\r\n");
    k2c_tensor img_tensor = {.array = bitmap_div,
                             .ndim = 2,
                             .numel = bitmap.infoHeader.hauteur * bitmap.infoHeader.largeur,
                             .shape = {bitmap.infoHeader.largeur, bitmap.infoHeader.hauteur, 1, 1, 1}};
    k2c_tensor output_tensor;
    printf("k2c_tensor OK\r\n");
    float *conv2d_output_array;
    float *conv2d_kernel_array;
    float *conv2d_bias_array;
    float *max_pooling2d_output_array;
    float *conv2d_1_output_array;
    float *conv2d_1_kernel_array;
    float *conv2d_1_bias_array;
    float *max_pooling2d_1_output_array;
    float *flatten_output_array;
    float *dense_kernel_array;
    float *dense_bias_array;
    model_initialize(&conv2d_output_array, &conv2d_kernel_array, &conv2d_bias_array, &max_pooling2d_output_array, &conv2d_1_output_array, &conv2d_1_kernel_array, &conv2d_1_bias_array, &max_pooling2d_1_output_array, &flatten_output_array, &dense_kernel_array, &dense_bias_array);
    printf("model init OK\r\n");
    model(&img_tensor, &output_tensor, conv2d_output_array, conv2d_kernel_array, conv2d_bias_array, max_pooling2d_output_array, conv2d_1_output_array, conv2d_1_kernel_array, conv2d_1_bias_array, max_pooling2d_1_output_array, flatten_output_array, dense_kernel_array, dense_bias_array);
    printf("model OK\r\n");
    DesallouerBMP(&bitmap);
    printf("DesallouerBMP OK\r\n");
    model_terminate(conv2d_output_array, conv2d_kernel_array, conv2d_bias_array, max_pooling2d_output_array, conv2d_1_output_array, conv2d_1_kernel_array, conv2d_1_bias_array, max_pooling2d_1_output_array, flatten_output_array, dense_kernel_array, dense_bias_array);
    printf("model_terminate OK\r\n");
    return 0;
}