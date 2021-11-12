#ifndef _SOFTMAX_LOSS_LAYER_
#define _SOFTMAX_LOSS_LAYER_

#include "error_util.hpp"
#include <cudnn.h>

class SoftmaxLossLayer {
public:
    SoftmaxLossLayer(int batch_size_, int data_size_);
    ~SoftmaxLossLayer();
    void InitParameter();
    void EvalSet();
    float* forward(float *input, cudnnHandle_t *handle);
    float* backward(float *label);
    float getloss(float *label);

private:
    float *data_in;
    float *data_out;
    float *bottom_diff;
    int batch_size;
    int data_size;
    cudnnTensorDescriptor_t src_tensor_desc;
    cudnnTensorDescriptor_t dst_tensor_desc;
    cudnnActivationDescriptor_t activation_desc;

    void SetDescriptor();
};

#endif //_SOFTMAX_LOSS_LAYER_