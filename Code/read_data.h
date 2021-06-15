#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define N_SAMPLES 60000
#define N_DIMS 784
#define N_CLASSES 10
#define N_TEST_SAMPLES 10000


// Function to read training and test data and store them appropriately
void read_csv_file(double** data, double* y_temp, double** y, char* dataset){
    FILE *file;
    if(strcmp(dataset, "train") == 0){
        file = fopen("mnist_train.csv", "r");
    }
    else if(strcmp(dataset, "test") == 0){
        file = fopen("mnist_test.csv", "r");
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

// Function to scale the dataset
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

// Function to normalize the dataset
void normalize_data(double** X_train, double** X_test){
    double* mean = malloc(N_DIMS*sizeof(double));
    double total = N_SAMPLES;
    for(int i=0;i<N_DIMS;i++){
        double sum = 0.0;
        for(int j=0;j<N_SAMPLES;j++){
            sum += X_train[j][i];
        }
        mean[i] = sum/total;
    }
    double* sd = malloc(N_DIMS*sizeof(double));
    for(int i=0;i<N_DIMS;i++){
        double sum = 0.0;
        for(int j=0;j<N_SAMPLES;j++){
            sum += pow(X_train[j][i] - mean[i], 2);
        }
        sd[i] = sqrt(sum/total);
    }
    for(int i=0;i<N_DIMS;i++){
        for(int j=0;j<N_SAMPLES;j++){
            if(sd[i]>0.0001){
                X_train[j][i] = (double)(X_train[j][i] - mean[i])/(double)sd[i];
            }
        }
        for(int j=0;j<N_TEST_SAMPLES;j++){
            if(sd[i]>0.0001){
                X_test[j][i] = (double)(X_test[j][i] - mean[i])/(double)sd[i];
            }
        }
    }
    free(sd);
    free(mean);
}
