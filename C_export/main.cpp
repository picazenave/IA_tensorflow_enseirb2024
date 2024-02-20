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

#include "TraitementImage.hpp"
#include "Bmp2Matrix.hpp"
#include <fdeep/fdeep.hpp>

using namespace std;

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

    printf("aaaa\r\n");
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

#if 0
    int output_size = 10;
    double *output_tensor_array = (double *)malloc(output_size * sizeof(double));
    k2c_tensor output_tensor = {output_tensor_array, 1, output_size, {output_size, 1, 1, 1, 1}};
    double *conv2d_output_array;
    double *conv2d_kernel_array;
    double *conv2d_bias_array;
    double *max_pooling2d_output_array;
    double *conv2d_1_output_array;
    double *conv2d_1_kernel_array;
    double *conv2d_1_bias_array;
    double *max_pooling2d_1_output_array;
    double *flatten_output_array;
    double *dense_kernel_array;
    double *dense_bias_array;
    model_initialize(&conv2d_output_array, &conv2d_kernel_array, &conv2d_bias_array, &max_pooling2d_output_array, &conv2d_1_output_array, &conv2d_1_kernel_array, &conv2d_1_bias_array, &max_pooling2d_1_output_array, &flatten_output_array, &dense_kernel_array, &dense_bias_array);

    FILE *pFichier;
    BMP bitmap;
    int chiffreDebut = 0, chiffreFin = 9;
    int quantiteDebut = 0, quantiteFin = 10;
    char nomFichierSource[200] = "../bmp_traitee/";
    char filename[200] = {0};
    char nom[200];

    for (int chiffre = chiffreDebut; chiffre <= chiffreFin; chiffre++)
    {
        for (int quantite = quantiteDebut; quantite <= quantiteFin; quantite++)
        {
            strcpy(filename, "");
            strcpy(filename, nomFichierSource);
            strcpy(nom, "");
            sprintf(nom, "%d_%d.bmp", chiffre, quantite);
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
            double *bitmap_div = (double *)malloc(im_size * sizeof(double));
            int width, height;
            width = bitmap.infoHeader.largeur;
            height = bitmap.infoHeader.hauteur;
            for (int col = 0; col < width; col++)
            {
                for (int row = 0; row < height; row++)
                {
                    bitmap_div[width * row + col] = (double)((double)bitmap.mPixelsGray[row][col] / 255.f);
                }
            }
            // for (int i = 0; i < im_size; i++)
            // {
            //     int j = (int)(i / width);
            //     // printf("bitmap(i/j):%d/%d []=%f\n",i-j*bitmap.infoHeader.largeur,j,bitmap.mPixelsGray[i-j*bitmap.infoHeader.largeur][j] / 255.f);
            //     bitmap_div[i] = (double)((double)bitmap.mPixelsGray[j][i - j * height] / 255.f);
            // }
            k2c_tensor img_tensor = {.array = bitmap_div,
                                     .ndim = 2,
                                     .numel = bitmap.infoHeader.hauteur * bitmap.infoHeader.largeur,
                                     .shape = {bitmap.infoHeader.largeur, bitmap.infoHeader.hauteur, 1, 1, 1}};

            // k2c_tensor img_tensor = {.array = bitmap_div,
            //                          .ndim = 2,
            //                          .numel = 784,
            //                          .shape = {28,28, 1, 1, 1}};

            model(&img_tensor, &output_tensor, conv2d_output_array, conv2d_kernel_array, conv2d_bias_array, max_pooling2d_output_array, conv2d_1_output_array, conv2d_1_kernel_array, conv2d_1_bias_array, max_pooling2d_1_output_array, flatten_output_array, dense_kernel_array, dense_bias_array);

            printf("Prediction (%s)=", filename);
            int index_local_max = 0;
            double temp = 0;
            for (int i = 0; i < output_size; i++)
            {
                if (output_tensor.array[i] > temp)
                {
                    temp = output_tensor.array[i];
                    index_local_max = i;
                }
            }
            for (int i = 0; i < output_size; i++)
            {
                if (i == chiffre)
                    printf("   %02d   ||", i);
                else if (i == index_local_max)
                    printf(RED "   %02d   ||" RESET, i);
                else
                    printf(YEL "   %02d   ||" RESET, i);
            }
            printf("\r\n");
            printf("Results    (%s)=", filename);
            for (int i = 0; i < output_size; i++)
            {
                if (i == chiffre)
                    printf(" %.4f ||", output_tensor.array[i]);
                else if (i == index_local_max)
                    printf(RED " %.4f ||" RESET, output_tensor.array[i]);
                else
                    printf(YEL " %.4f ||" RESET, output_tensor.array[i]);
            }
            printf("\r\n=====================\r\n");

            DesallouerBMP(&bitmap);
            free(bitmap_div);
        }
    }

    free(output_tensor_array);
    model_terminate(conv2d_output_array, conv2d_kernel_array, conv2d_bias_array, max_pooling2d_output_array, conv2d_1_output_array, conv2d_1_kernel_array, conv2d_1_bias_array, max_pooling2d_1_output_array, flatten_output_array, dense_kernel_array, dense_bias_array);
#endif
    return 0;
}