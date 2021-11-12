#ifndef _POOLING_LAYER_HPP_
#define _POOLING_LAYER_HPP_

#include "error_util.hpp"
#include <cudnn.h>

class PoolingLayer {
public:
    PoolingLayer(int batch_size_, int in_channels_, int in_hight_, 
        int in_width_, int kernel_size_, int stride_, int padding_);
    ~PoolingLayer();
    void InitParameter();
    void EvalSet();
    float* forward(float *input, cudnnHandle_t *handle);
    float* backward(float *top_diff, cudnnHandle_t *handle);

private:
    float *data_in;
    float *data_out;
    float *bottom_diff;

    int batch_size;
    int in_channels;
    int in_height;
    int in_width;
    int kernel_size;
    int stride;
    int padding;
    int out_size; // for allocating data_out memory on device

    // cudnn tensor and layer descriptors
    cudnnTensorDescriptor_t src_tensor_desc;
    cudnnTensorDescriptor_t dst_tensor_desc;
    cudnnTensorDescriptor_t bias_tensor_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnPoolingDescriptor_t pooling_desc;

    void SetDescriptor();
};

#endif //_POOLING_LAYER_HPP_