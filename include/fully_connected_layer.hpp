#ifndef _FULLY_CONNECTED_LAYER_HPP_
#define _FULLY_CONNECTED_LAYER_HPP_

#include "error_util.hpp"
#include <cublas_v2.h>

class FullyConnectedLayer {
public:
    FullyConnectedLayer(int batch_size_, int in_size_, int out_size_);
    ~FullyConnectedLayer();
    void InitParameter();
    void EvalSet();
    float* forward(float *input, cublasHandle_t *handle);
    float* backward(float *top_diff, cublasHandle_t *handle);
    void UpdateWeights(float lr, cublasHandle_t *handle);

private:
    float *data_in;
    float *data_out;
    float *weight;
    float *bias;
    float *d_weight;
    float *d_bias;
    float *bottom_diff;
    // vector [1, 1, ... , 1] with batch_size on GPU
    float *onevec;

    int batch_size;
    int in_size;
    int out_size;
};

#endif //_FULLY_CONNECTED_LAYER_HPP_