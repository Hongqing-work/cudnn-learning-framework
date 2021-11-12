#ifndef _CONV_LAYER_HPP_
#define _CONV_LAYER_HPP_

#include "error_util.hpp"
#include <cudnn.h>
#include <cublas_v2.h>

class ConvLayer {
public:
    ConvLayer(int batch_size_, int in_channels_, int in_hight_, int in_width_, 
        int out_channels_, int kernel_size_, int stride_, int padding_);
    ~ConvLayer();
    void InitParameter(cudnnHandle_t *handle);
    void EvalSet();
    float* forward(float *input, cudnnHandle_t *handle);
    float* backward(float *top_diff, cudnnHandle_t *handle);
    void UpdateWeights(float lr, cublasHandle_t *handle);

private:
    float *data_in;
    float *data_out;
    float *weight;
    float *bias;
    float *d_weight;
    float *d_bias;
    float *bottom_diff;

    int batch_size;
    int in_channels;
    int in_height;
    int in_width;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int out_size; // for allocating data_out memory on device

    // cudnn tensor and layer descriptors
    cudnnTensorDescriptor_t src_tensor_desc;
    cudnnTensorDescriptor_t dst_tensor_desc;
    cudnnTensorDescriptor_t bias_tensor_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    // algorithms for forward and backward
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;

    void SetDescriptor();
    void SetAlgorithm(cudnnHandle_t *handle);
};

#endif //_CONV_LAYER_HPP_