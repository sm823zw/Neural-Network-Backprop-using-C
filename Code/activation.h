#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x){
    double ans = (double)1/(double)(1 + exp(-x));
    return ans;
}

double sigmoid_d(double x){
    return sigmoid(x)*(1-sigmoid(x));
}

double relu(double x){
    if(x < 0.0){
        return 0.0;
    }
    return x;
}

double relu_d(double x){
    if(x < 0.0){
        return 0.0;
    }
    return 1.0;
}

double tanh_d(double x){
    double res = 1.0 - tanh(x)*tanh(x);
    return res;
}
