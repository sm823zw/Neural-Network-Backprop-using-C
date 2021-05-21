#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "read_data.h"
#include "activation.h"

int seed;
double randn(){
  int a = 1103515245;
  int m = 2147483647;
  int c = 12345;
  seed = (a * seed + c) % m;
  double x = (double)seed/(double)m;
  return x;
}

// double randn(){
//     return (double)rand()/(double)RAND_MAX;
// }

struct NeuralNet{
    int n_layers;
    int* n_neurons_per_layer;
    double*** w;
    double** b;
    double*** delta_w;
    double** delta_b;
    double** theta;
    double** in;
    double** out;
    double* targets;
};

struct NeuralNet* newNet(int n_layers, int n_neurons_per_layer[]){
    struct NeuralNet* nn = malloc(sizeof(struct NeuralNet));
    nn->n_layers = n_layers;
    nn->n_neurons_per_layer = malloc(nn->n_layers * sizeof(int));
    for(int i=0;i<n_layers;i++){
        nn->n_neurons_per_layer[i] = n_neurons_per_layer[i];
    }
    nn->w = malloc((nn->n_layers-1)*sizeof(double**));
    nn->delta_w = malloc((nn->n_layers-1)*sizeof(double**));
    nn->b = malloc((nn->n_layers-1)*sizeof(double*));
    nn->delta_b = malloc((nn->n_layers-1)*sizeof(double*));
    for(int i=0;i<nn->n_layers-1;i++){
        nn->w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->delta_w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->delta_b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            nn->w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
            nn->delta_w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
        }
    }
    for(int k=0;k<nn->n_layers-1;k++){
        for(int i=1;i<nn->n_neurons_per_layer[k]+1;i++){
            nn->b[k][i] = 0.0;
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                nn->w[k][i][j] = randn();
            }
        }
    }
    nn->theta = malloc((nn->n_layers)*sizeof(double*));
    nn->in = malloc((nn->n_layers)*sizeof(double*));
    nn->out = malloc((nn->n_layers)*sizeof(double*));
    for(int i=0;i<nn->n_layers;i++){
        nn->in[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->out[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->theta[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
    }
    nn->targets = malloc((nn->n_neurons_per_layer[nn->n_layers-1]+1)*sizeof(double));
    return nn;
}

void shuffle(int* arr, size_t n){
    if(n > 1){
        for(size_t i=0;i<n-1;i++){
        size_t j = i+rand()/(RAND_MAX/(n-i)+1);
          int t = arr[j];
          arr[j] = arr[i];
          arr[i] = t;
        }
    }
}

void forward_propagation(struct NeuralNet* nn, char* activation_fun){
    for(int i=0;i<nn->n_layers;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            nn->in[i][j] = 0.0;
        }
    }
    for(int k=1;k<nn->n_layers;k++){
        for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
            nn->in[k][j] += 1.0 * nn->b[k-1][j];
        }
        for(int i=1;i<nn->n_neurons_per_layer[k-1]+1;i++){
            for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                nn->in[k][j] += nn->out[k-1][i] * nn->w[k-1][i][j];
            }
        }
        if(k == nn->n_layers-1){
            for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                nn->out[k][j] = sigmoid(nn->in[k][j]);
            }
        }
        else{
            for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                if(strcmp(activation_fun, "sigmoid") == 0){
                    nn->out[k][j] = sigmoid(nn->in[k][j]);
                }
                else if(strcmp(activation_fun, "tanh") == 0){
                    nn->out[k][j] = tanh(nn->in[k][j]);
                }
                else if(strcmp(activation_fun, "relu") == 0){
                    nn->out[k][j] = relu(nn->in[k][j]);
                }
                else{
                    nn->out[k][j] = sigmoid(nn->in[k][j]);
                }
            }
        }
    }
}

double mean_square_error_loss(struct NeuralNet* nn){
    double mse = 0.0;
    int last_layer = nn->n_layers-1;
    for(int i=1;i<nn->n_neurons_per_layer[last_layer]+1;i++){
        mse += (nn->out[last_layer][i] - nn->targets[i]) * (nn->out[last_layer][i] - nn->targets[i]);
	}
    mse *= 0.5;
    return mse;
}

void back_propagation(struct NeuralNet* nn, char* activation_fun, double learning_rate, double momentum){
    int last_layer = nn->n_layers-1;
    for(int i=1;i<nn->n_neurons_per_layer[last_layer]+1;i++){
        double grad;
        if(strcmp(activation_fun, "sigmoid") == 0){
            grad = sigmoid_d(nn->out[last_layer][i]);
        }
        else if(strcmp(activation_fun, "tanh") == 0){
            grad = sigmoid_d(nn->out[last_layer][i]);
        }
        else if(strcmp(activation_fun, "relu") == 0){
            grad = sigmoid_d(nn->out[last_layer][i]);
        }
        else{
            grad = sigmoid_d(nn->out[last_layer][i]);
        }
        nn->theta[last_layer][i] = grad * (nn->targets[i] - nn->out[last_layer][i]);
    }
    for(int k=nn->n_layers-2;k>0;k--){
        
        for(int i=1;i<nn->n_neurons_per_layer[k]+1;i++){
            double sum = 0.0;
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                sum += nn->b[k][j] * nn->theta[k+1][j];
            }
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                sum += nn->w[k][i][j] * nn->theta[k+1][j];
            }
            double grad;
            if(strcmp(activation_fun, "sigmoid") == 0){
                grad = sigmoid_d(nn->out[k][i]);
            }
            else if(strcmp(activation_fun, "tanh") == 0){
                grad = tanh_d(nn->out[k][i]);
            }
            else if(strcmp(activation_fun, "relu") == 0){
                grad = relu_d(nn->out[k][i]);
            }
            else{
                grad = sigmoid_d(nn->out[k][i]);
            }
            nn->theta[k][i] = grad * sum;
        }
    }
    for(int k=0;k<nn->n_layers-1;k++){
        for(int i=1;i<nn->n_neurons_per_layer[k]+1;i++){
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                nn->delta_w[k][i][j] = (learning_rate * nn->theta[k+1][j] * nn->out[k][i]) + (momentum * nn->delta_w[k][i][j]);
                nn->w[k][i][j] += nn->delta_w[k][i][j];
            }
        }
        for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
            nn->delta_b[k][j] = (learning_rate * nn->theta[k+1][j] * 1.0) + (momentum * nn->delta_b[k][j]);
            nn->b[k][j] += nn->delta_b[k][j];
        }
    }
}

double* model_train(struct NeuralNet* nn, double** X_train, double** y_train, double* y_train_temp, char* activation_fun, double learning_rate, double momentum, int num_samples_to_train){
    int arr[num_samples_to_train];
    for(int i=0;i<num_samples_to_train;i++){
        arr[i] = i;
    }
    int correct = 0;
    double error = 0.0;
    for(int i=0;i<num_samples_to_train;i++){
        shuffle(arr, num_samples_to_train);
        int idx = -1;
        double max_val = (double)INT_MIN;
        for(int j=1;j<nn->n_neurons_per_layer[0]+1;j++){
            nn->out[0][j] = X_train[arr[i]][j-1];
        }
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            nn->targets[j] = y_train[arr[i]][j-1];
        }
        forward_propagation(nn, activation_fun);
        back_propagation(nn, activation_fun, learning_rate, momentum);
        error += mean_square_error_loss(nn);
            
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            if(nn->out[nn->n_layers-1][j] > max_val){
                max_val =nn->out[nn->n_layers-1][j];
                idx = j-1;
            }
        }
        if(idx == (int)y_train_temp[arr[i]]){
            correct++;
        }
    }
    error /=(double)num_samples_to_train;
    double accuracy = (double)correct/(double)num_samples_to_train;
    static double metrics[2];
    metrics[0] = error;
    metrics[1] = accuracy;
    return metrics;
}

double* model_test(struct NeuralNet* nn, double** X_test, double** y_test, double* y_test_temp, char* activation_fun){
    double error = 0.0;
    int correct = 0;
    for(int i=0;i<N_TEST_SAMPLES;i++){
        int idx = -1;
        double max_val = (double)INT_MIN;
        for(int j=1;j<nn->n_neurons_per_layer[0]+1;j++){
            nn->out[0][j] = X_test[i][j-1];
        }
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            nn->targets[j] = y_test[i][j-1];
        }
        forward_propagation(nn, activation_fun);
        error += mean_square_error_loss(nn);
            
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            if(nn->out[nn->n_layers-1][j] > max_val){
                max_val =nn->out[nn->n_layers-1][j];
                idx = j-1;
            }
        }
        if(idx == (int)y_test_temp[i]){
            correct++;
        }
    }
    error /= (double)N_TEST_SAMPLES;
    double accuracy = (double)correct/(double)N_TEST_SAMPLES;
    static double metrics[2];
    metrics[0] = error;
    metrics[1] = accuracy;
    return metrics;
}

int main(){


    srand(time(NULL));
    seed = rand();

    int n_layers = 3;
    int n_neurons_per_layer[] = {784, 64, 10};
    struct NeuralNet* nn = newNet(n_layers, n_neurons_per_layer);

    double learning_rate = 1e-4;
    double momentum = 0.9;
    char* activation_fun = "relu";
    int num_samples_to_train = 10000;
    int epochs = 50;

    double** X_train = malloc(N_SAMPLES*sizeof(double*));
    for(int i=0;i<N_SAMPLES;i++){
        X_train[i] = malloc(N_DIMS*sizeof(double));
    }
    double** y_train = malloc(N_SAMPLES * sizeof(double*));
    for(int i=0;i<N_SAMPLES;i++){
        y_train[i] = malloc(N_CLASSES * sizeof(double));
    }
    double* y_train_temp = malloc(N_SAMPLES*sizeof(double));
    read_csv_file(X_train, y_train_temp, y_train, "train");
    scale_data(X_train, "train");

    double** X_test = malloc(N_TEST_SAMPLES*sizeof(double*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        X_test[i] = malloc(N_DIMS*sizeof(double));
    }
    double** y_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        y_test[i] = malloc(N_CLASSES * sizeof(double));
    }
    double* y_test_temp = malloc(N_TEST_SAMPLES*sizeof(double));
    read_csv_file(X_test, y_test_temp, y_test, "test");
    scale_data(X_test, "test");
    
    for(int itr=0;itr<epochs;itr++){
        double* train_metrics = model_train(nn, X_train, y_train, y_train_temp, activation_fun, learning_rate, momentum, num_samples_to_train);
        double train_error = train_metrics[0];
        double train_acc = train_metrics[1];
        double* test_metrics = model_test(nn, X_test, y_test, y_test_temp, activation_fun);
        double test_error = test_metrics[0];
        double test_acc = test_metrics[1];
        printf("Epoch: %d, ", itr+1);
        printf("Train Error: %lf, ", train_error);
        printf("Train Accuracy: %lf, ", train_acc);
        printf("Test Error: %lf, ", test_error);
        printf("Test Accuracy: %lf\n", test_acc);
    }

    return 0;
}
