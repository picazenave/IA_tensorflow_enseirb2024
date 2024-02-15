#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"
#define RESET "\x1B[0m"

#include "TraitementImage.h"
#include "k2c_tensor_include.h"
#include "model.h"
#include "Bmp2Matrix.h"

int main(int argc, char *argv[])
{
    // BMP bitmap;
    // FILE *pFichier = NULL;
    // char filename[] = "1_0.bmp";
    // pFichier = fopen(filename, "rb"); // Ouverture du fichier contenant l'image
    // if (pFichier == NULL)
    // {
    //     printf("%s\n", "0_1.bmp");
    //     printf("Erreur dans la lecture du fichier\n");
    // }
    // LireBitmap(pFichier, &bitmap);
    // fclose(pFichier); // Fermeture du fichier contenant l'image

    // ConvertRGB2Gray(&bitmap);
    // printf("%d\n", bitmap.mPixelsGray[10][10]);

    if (argc > 1)
    {
        if (strstr(argv[1], "convert"))
        {
            BMP bitmap_traitement;
            TraitementDonnees(&bitmap_traitement, "../../Database/Prof/bmpProcessedSeuil/", "../bmp_traitee/", 0, 9, 0, 20, (unsigned char)(255 / 2.f), 0, 1, 28, 28);
        }
        else
            printf("argc=%d; argv[0]=\"%s\"; argv[1]=\"%s\";", argc, argv[0], argv[1]);
    }

    // BMP bitmap;
    // FILE *pFichier = NULL;
    // char filename[] = "../bmp_traitee/0_0.bmp";
    // pFichier = fopen(filename, "rb"); // Ouverture du fichier contenant l'image
    // if (pFichier == NULL)
    // {
    //     printf("%s\n", filename);
    //     printf("Erreur dans la lecture du fichier\n");
    //     while (1)
    //         ;
    // }
    // LireBitmap(pFichier, &bitmap);
    // fclose(pFichier); // Fermeture du fichier contenant l'image

    FILE *pFichier;
    BMP bitmap;
    int chiffreDebut = 0, chiffreFin = 9;
    int quantiteDebut = 0, quantiteFin = 10;
    char nomFichierSource[200] = "../bmp_traitee/";
    char filename[200] = {0};
    char nom[200];
    for (int i = chiffreDebut; i <= chiffreFin; i++)
    {
        for (int j = quantiteDebut; j <= quantiteFin; j++)
        {
            strcpy(filename, "");
            strcpy(filename, nomFichierSource);
            strcpy(nom, "");
            sprintf(nom, "%d_%d.bmp", i, j);
            strcat(filename, nom);
            pFichier = fopen(filename, "rb"); // Ouverture du fichier contenant l'image
            if (pFichier == NULL)
            {
                printf("%s\n", filename);
                printf("Erreur dans la lecture du fichier\n");
            }
            LireBitmap(pFichier, &bitmap);
            fclose(pFichier); // Fermeture du fichier contenant l'image

            int im_size = bitmap.infoHeader.hauteur * bitmap.infoHeader.largeur;
            printf("bitmap(l/h):%d/%d;im_size=%d\n", bitmap.infoHeader.largeur, bitmap.infoHeader.hauteur, im_size);
            float *bitmap_div = (float *)malloc(im_size * sizeof(float));
            for (int i = 0; i < im_size; i++)
            {
                int j = (int)(i / bitmap.infoHeader.largeur);
                // printf("bitmap(i/j):%d/%d []=%f\n",i-j*bitmap.infoHeader.largeur,j,bitmap.mPixelsGray[i-j*bitmap.infoHeader.largeur][j] / 255.f);
                bitmap_div[i] = bitmap.mPixelsGray[i - j * bitmap.infoHeader.largeur][j] / 255.f;
            }
            k2c_tensor img_tensor = {.array = bitmap_div,
                                     .ndim = 2,
                                     .numel = bitmap.infoHeader.hauteur * bitmap.infoHeader.largeur,
                                     .shape = {bitmap.infoHeader.largeur, bitmap.infoHeader.hauteur, 1, 1, 1}};

            // k2c_tensor img_tensor = {.array = bitmap_div,
            //                          .ndim = 2,
            //                          .numel = 784,
            //                          .shape = {28,28, 1, 1, 1}};
            int output_size = 10;
            float *output_tensor_array = (float *)malloc(output_size * sizeof(float));
            k2c_tensor output_tensor = {output_tensor_array, 1, output_size, {output_size, 1, 1, 1, 1}};
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

            model(&img_tensor, &output_tensor, conv2d_output_array, conv2d_kernel_array, conv2d_bias_array, max_pooling2d_output_array, conv2d_1_output_array, conv2d_1_kernel_array, conv2d_1_bias_array, max_pooling2d_1_output_array, flatten_output_array, dense_kernel_array, dense_bias_array);

            printf("Prediction (%s)=", filename);
            for (int i = 0; i < output_size; i++)
            {
                printf("   %02d   ||", i);
            }
            printf("\r\n");
            printf("Results    (%s)=", filename);
            for (int i = 0; i < output_size; i++)
            {
                printf(" %.4f ||", output_tensor.array[i]);
            }
            printf("\r\n");
            printf("\r\n=====================\r\n");

            DesallouerBMP(&bitmap);
            model_terminate(conv2d_output_array, conv2d_kernel_array, conv2d_bias_array, max_pooling2d_output_array, conv2d_1_output_array, conv2d_1_kernel_array, conv2d_1_bias_array, max_pooling2d_1_output_array, flatten_output_array, dense_kernel_array, dense_bias_array);
            free(bitmap_div);
            free(output_tensor_array);
        }
    }
    return 0;
}