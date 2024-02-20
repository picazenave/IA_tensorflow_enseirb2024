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
#include <fdeep/common.hpp>
#include <vector>
#include <Eigen/Core>
using namespace std;

int main(int argc, char *argv[])
{
    // BMP bitmap;
    // FILE *pFichier = NULL;
    // char filename[] = "../bmp_traitee/1_0.bmp";
    // pFichier = fopen(filename, "rb"); // Ouverture du fichier contenant l'image
    // if (pFichier == NULL)
    // {
    //     printf("%s\n", filename);
    //     printf("Erreur dans la lecture du fichier\n");
    // }
    // LireBitmap(pFichier, &bitmap);
    // fclose(pFichier); // Fermeture du fichier contenant l'image

    // ConvertRGB2Gray(&bitmap);
    // printf("%d\n", bitmap.mPixelsGray[10][10]);

    // printf("aaaa\r\n");
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

    // int im_size = bitmap.infoHeader.hauteur * bitmap.infoHeader.largeur;
    // printf("bitmap(l/h):%d/%d;im_size=%d\n", bitmap.infoHeader.largeur, bitmap.infoHeader.hauteur, im_size);
    // uint8_t *bitmap_div = (uint8_t *)malloc(im_size * sizeof(uint8_t));
    // int width, height;
    // width = bitmap.infoHeader.largeur;
    // height = bitmap.infoHeader.hauteur;
    // for (int col = 0; col < width; col++)
    // {
    //     for (int row = 0; row < height; row++)
    //     {
    //         bitmap_div[width * row + col] = bitmap.mPixelsGray[row][col];
    //     }
    // }

    // //  Use the correct scaling, i.e., low and high.
    // const auto input = fdeep::tensor_from_bytes(bitmap_div,
    //                                             static_cast<std::size_t>(height),
    //                                             static_cast<std::size_t>(width),
    //                                             static_cast<std::size_t>(1),
    //                                             0.0f, 1.0f);

    const auto model = fdeep::load_model("fdeep_model.json");
    // const auto result = model.predict({input});

    // float results_float[10];
    // printf("aaaa\r\n");
    // sscanf(fdeep::show_tensors(result).c_str(),"[[[[[[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]]]]]",&results_float[0],&results_float[1],&results_float[2],&results_float[3],&results_float[4],&results_float[5],&results_float[6],&results_float[7],&results_float[8],&results_float[9]);
    // float local_max;
    // int i_local_max;
    // for (int i = 0; i < 10; i++)
    // {
    //     printf("for:%d value=%f\r\n",i,results_float[i]);
    //     float a = results_float[i];
    //     if (a > local_max)
    //     {
    //         local_max = a;
    //         i_local_max = i;
    //     }
    // }
    // printf("max[%d]=%f\r\n", i_local_max, local_max);
    // std::cout << fdeep::show_tensors(result) << std::endl;

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

#if 1
    int output_size = 10;
    int error = 0;
    int loop_count = 0;
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
            loop_count++;
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

            ConvertRGB2Gray(&bitmap);
            int im_size = bitmap.infoHeader.hauteur * bitmap.infoHeader.largeur;
            // printf("bitmap(l/h):%d/%d;im_size=%d\n", bitmap.infoHeader.largeur, bitmap.infoHeader.hauteur, im_size);
            uint8_t *bitmap_div = (uint8_t *)malloc(im_size * sizeof(uint8_t));
            int width, height;
            width = bitmap.infoHeader.largeur;
            height = bitmap.infoHeader.hauteur;
            for (int col = 0; col < width; col++)
            {
                for (int row = 0; row < height; row++)
                {
                    bitmap_div[width * row + col] = bitmap.mPixelsGray[row][col];
                }
            }

            //  Use the correct scaling, i.e., low and high.
            const auto input = fdeep::tensor_from_bytes(bitmap_div,
                                                        static_cast<std::size_t>(height),
                                                        static_cast<std::size_t>(width),
                                                        static_cast<std::size_t>(1),
                                                        0.0f, 1.0f);
            /**
             * ======================================
             */
            const auto result = model.predict({input});
            float results_float[10];
            sscanf(fdeep::show_tensors(result).c_str(), "[[[[[[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]]]]]", &results_float[0], &results_float[1], &results_float[2], &results_float[3], &results_float[4], &results_float[5], &results_float[6], &results_float[7], &results_float[8], &results_float[9]);
            float local_max = 0;
            int i_local_max;
            for (int i = 0; i < 10; i++)
            {
                if (results_float[i] > local_max)
                {
                    local_max = results_float[i];
                    i_local_max = i;
                }
            }
            printf("max[%d]=%f\r\n", i_local_max, local_max);
            if (i_local_max != chiffre)
                error++;
            /**
             * ======================================
             */
            printf("Prediction (%s)=", filename);
            for (int i = 0; i < output_size; i++)
            {
                if (i == chiffre)
                    printf("   %02d   ||", i);
                else if (i == i_local_max)
                    printf(RED "   %02d   ||" RESET, i);
                else
                    printf(YEL "   %02d   ||" RESET, i);
            }
            printf("\r\n");
            printf("Results    (%s)=", filename);
            for (int i = 0; i < output_size; i++)
            {
                if (i == chiffre)
                    printf(" %.4f ||", results_float[i]);
                else if (i == i_local_max)
                    printf(RED " %.4f ||" RESET, results_float[i]);
                else
                    printf(YEL " %.4f ||" RESET, results_float[i]);
            }
            printf("\r\n=====================\r\n");

            DesallouerBMP(&bitmap);
            free(bitmap_div);
        }
    }
#endif
    printf("error count=%d/%d\r\n", error, loop_count);
    return 0;
}