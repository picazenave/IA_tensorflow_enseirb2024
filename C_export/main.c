#include <stdio.h>
#include <stdlib.h>
#include "Bmp2Matrix.h"
#include <string.h>
#include "activation.h"

#define ROWS 3 // Nombre de lignes de la matrice
#define COLS 3 // Nombre de colonnes de la matrice

void read_matrice(double **matrix, FILE *fp)
{
    char *line = NULL;
    size_t len = 0;
    int read;
    int row = 0;
    int col = 0;
    while ((read = getline(&line, &len, fp)) != -1)
    {
        matrix[row][col] = atof(line);
        if (col == 2 && row == 2)
            break;
        else if (col == 2 && row < 2)
        {
            col = 0;
            row++;
        }
        else
            col++;
    }
}

void read_weigth(double *matrix, FILE *fp)
{
    char *line = NULL;
    size_t len = 0;
    int read;
    int row = 0;
    printf("lg=%d file=%s\r\n", __LINE__, __FILE__);
    while ((read = getline(&line, &len, fp)) != -1)
    {
        printf("%d lg=%d file=%s\r\n",row, __LINE__, __FILE__);
        matrix[row] = atof(line);
        row++;
    }
    printf("lg=%d file=%s\r\n", __LINE__, __FILE__);
}

int main()
{
    FILE *fp;
    fp = fopen("../../results/weights_0.txt", "r");
    if (fp == NULL)
    {
        perror("Erreur lors de l'ouverture du fichier");
        return -1;
    }
    

    FILE *fp_2;
    fp_2 = fopen("../../results/weights_2.txt", "r");
    if (fp_2 == NULL)
    {
        perror("Erreur lors de l'ouverture du fichier");
        return -1;
    }
    

    FILE *bias;
    bias = fopen("../../results/bias_0.txt", "r");
    if (bias == NULL)
    {
        perror("Erreur lors de l'ouverture du fichier");
        return -1;
    }
    
    

    FILE *bias_2;
    bias_2 = fopen("../../results/bias_0.txt", "r");
    if (bias_2 == NULL)
    {
        perror("Erreur lors de l'ouverture du fichier");
        return -1;
    }

    FILE *pFichier;
    BMP bitmap;
    int chiffreDebut = 0, chiffreFin = 9;
    int quantiteDebut = 0, quantiteFin = 10;
    char nomFichierSource[200] = "../bmp_traitee/";
    char filename[200] = {0};
    char nom[200];

    // var lecture bias
    char *line = NULL;
    size_t len = 0;
    int read;
    int row = 0;
    double bias_read;

    // for (int chiffre = chiffreDebut; chiffre <= chiffreFin; chiffre++)
    //{
    // for (int quantite = quantiteDebut; quantite <= quantiteFin; quantite++)
    //{
    strcpy(filename, "");
    strcpy(filename, nomFichierSource);
    strcpy(nom, "");
    sprintf(nom, "%d_%d.bmp", 0, 1);
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
    // double **bitmap_div = (double **)malloc(im_size * sizeof(double)); // double **bitmap_div = (double **)malloc(sizeof(double[bitmap.infoHeader.hauteur][bitmap.infoHeader.largeur]));
    int width, height;
    width = bitmap.infoHeader.largeur;
    height = bitmap.infoHeader.hauteur;
    double **bitmap_div = (double **)malloc(width * sizeof(double *));
    for (int i = 0; i < height; i++)
    {
        bitmap_div[i] = (double *)malloc(height * sizeof(double));
    }

    for (int col = 0; col < width; col++)
    {
        for (int row = 0; row < height; row++)
        {
            bitmap_div[row][col] = (double)((double)bitmap.mPixelsGray[row][col] / 255.f);
        }
    }

    double **weight = (double **)malloc(ROWS * sizeof(double *));
    for (int i = 0; i < ROWS; i++)
    {
        weight[i] = (double *)malloc(COLS * sizeof(double));
    }

    double **output_layer2 = (double **)malloc(width * sizeof(double *));
    for (int i = 0; i < height; i++)
    {
        output_layer2[i] = (double *)malloc(COLS * sizeof(double));
    }

    int filters = 32;

    double ***filter_weights = (double ***)malloc(64 * sizeof(double **));
    for (int f = 0; f < 64; f++)
    {
        filter_weights[f] = (double **)malloc((width -2) * sizeof(double *));
        for (int i = 0; i < (width -2); i++)
        {
            filter_weights[f][i] = (double *)malloc((height -2)* sizeof(double));
        }
    }

    double ***output_max_poll = (double ***)malloc(64 * sizeof(double **));
    for (int f = 0; f < 64; f++)
    {
        output_max_poll[f] = (double **)malloc(((width -2)/2) * sizeof(double *));
        for (int i = 0; i < ((width -2)/2); i++)
        {
            output_max_poll[f][i] = (double *)malloc(((height -2)/2)* sizeof(double));
        }
    }
    // first convolution
    for (int i = 0; i < filters; i++)
    {
        read_matrice(weight, fp);
        read = getline(&line, &len, bias);
        bias_read = atof(line);
        
        convolution2D(bitmap_div, filter_weights[i], width-2, height-2, weight, ROWS, COLS, bias_read);
        relu(filter_weights[i], width-2, height-2, 0.01);
        maxPooling2D(filter_weights[i], width-2, height-2, 2, output_max_poll[i]);
    }



    double ***output_second_conv = (double ***)malloc(64 * sizeof(double **));
    for (int f = 0; f < 64; f++)
    {
        output_second_conv[f] = (double **)malloc(((width -2)/2) * sizeof(double *));
        for (int i = 0; i < ((width -2)/2); i++)
        {
            output_second_conv[f][i] = (double *)malloc(((height -2)/2) * sizeof(double));
        }
    }

    double ***output_max_poll_2 = (double ***)malloc(64 * sizeof(double **));
    for (int f = 0; f < 64; f++)
    {
        output_max_poll_2[f] = (double **)malloc((((width -2)/2)/2) * sizeof(double *));
        for (int i = 0; i < ((width -2)/2); i++)
        {
            output_max_poll_2[f][i] = (double *)malloc((((height -2)/2)/2)* sizeof(double));
        }
    }
    
    //second convolution
    for (int i = 0; i < 64; i++)
    {
        read_matrice(weight, fp_2);
        // printf("%d\n",((width -2)/2));
        read = getline(&line, &len, bias_2);
        bias_read = atof(line);
        convolution2D(output_max_poll[i], output_second_conv[i], ((width -2)/2), ((height -2)/2), weight, ROWS, COLS,bias_read);
        relu(output_second_conv[i], ((width -2)/2), ((height -2)/2), 0.01);
        maxPooling2D(output_second_conv[i], ((width -2)/2), ((height -2)/2), 2, output_max_poll_2[i]);
    }

    double *output_flaten = (double *)malloc(ROWS * sizeof(double ));
    
    flatten(output_second_conv, 64, width/4, height/4, output_flaten);


    double* weight_dense=(double *)malloc(16000 * sizeof(double));;
    printf("lg=%d file=%s\r\n", __LINE__, __FILE__);
    FILE *fp_weight;
    fp_weight = fopen("../../results/weights_5.txt", "r");
    printf("lg=%d file=%s\r\n", __LINE__, __FILE__);
    if (fp_weight == NULL)
    {
        perror("Erreur lors de l'ouverture du fichier");
        return -1;
    }
    read_weigth(weight_dense, fp_weight);

    // double bias_dense[10];
    // FILE *fp_bias;
    // fp_bias = fopen("../../results/weights_5.txt", "r");
    // if (fp_bias == NULL)
    // {
    //     perror("Erreur lors de l'ouverture du fichier");
    //     return -1;
    // }
    // read_weigth(bias_dense, fp_bias);

    // double output_dense[10];
    // double output_softmax[10];
    // dense(output_flaten, 64*(((width -2)/2)/2)*(((height -2)/2)), weight_dense, bias_dense,10,output_dense);
    // softmax(output_dense,10,output_softmax);
    // printf("lg=%d file=%s\r\n", __LINE__, __FILE__);
    // fclose(fp);


    DesallouerBMP(&bitmap);
    free(bitmap_div);
    free(weight);
    free(filter_weights);
    free(output_flaten);
    free(output_second_conv);

    free(output_layer2);
    free(output_max_poll);
    free(output_max_poll_2);

    free(weight_dense);

    fclose(fp);
    fclose(fp_2);
    fclose(bias);
    fclose(bias_2);

    return 0;
}
