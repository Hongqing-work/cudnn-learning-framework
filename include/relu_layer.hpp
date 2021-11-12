#ifndef _RELU_LAYER_HPP_
#define _RELU_LAYER_HPP_

#include "error_util.hpp"
#include <cudnn.h>

class ReluLayer {
public:
    ReluLayer(int batch_size_, int in_size_);
    ~ReluLayer();
    void InitParameter();
    void EvalSet();
    float* forward(float *input, cudnnHandle_t *handle);
    float* backward(float *top_diff, cudnnHandle_t *handle);

private:
    float *data_in;
    float *data_out;
    float *bottom_diff;

    int batch_size;
    int in_size;
    cudnnTensorDescriptor_t src_tensor_desc;
    cudnnTensorDescriptor_t dst_tensor_desc;
    cudnnActivationDescriptor_t activation_desc;

    void SetDescriptor();
};

#endif //_RELU_LAYER_HPP_