#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define N_SAMPLES 60000
#define N_DIMS 784
#define N_CLASSES 10
#define N_TEST_SAMPLES 10000

void scale_data(double** data, char* dataset){
    int total_samples;
    if(strcmp(dataset, "train") == 0){
        total_samples = N_SAMPLES;
    }
    else if(strcmp(dataset, "test") == 0){
        total_samples = N_TEST_SAMPLES;
    }
    for(int i=0;i<total_samples;i++){
        for(int j=0;j<N_DIMS;j++){
            data[i][j] = (double)data[i][j]/(double)255.0;
        }
    }
}

void read_csv_file(double** data, double* y_temp, double** y, char* dataset){
    FILE *file;
    if(strcmp(dataset, "train") == 0){
        file = fopen(".../Dataset/mnist_train.csv", "r");
    }
    else if(strcmp(dataset, "test") == 0){
        file = fopen(".../Dataset/mnist_test.csv", "r");
    }
    if(file == NULL){
        printf("Error reading the file!");
        exit(1);
    }
    char buffer[3200];
    int i = 0;
    while(fgets(buffer, sizeof(buffer), file)){
        char* token = strtok(buffer, ",");
        int j = 0;
        while(token != NULL){
            if(j == 0){
                y_temp[i] = (double)atoi(token);
            }
            else{
                data[i][j-1] = (double)atoi(token);
            }
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }
    fclose(file);
    int total_samples;
    if(strcmp(dataset, "train") == 0){
        total_samples = N_SAMPLES;
    }
    else if(strcmp(dataset, "test") == 0){
        total_samples = N_TEST_SAMPLES;
    }
    for(int i=0;i<total_samples;i++){
        for(int j=0;j<N_CLASSES;j++){
            if((int)y_temp[i] == j){
                y[i][j] = 1.0;
            }
            else{
                y[i][j] = 0.0;
            }
        }
    }
}
