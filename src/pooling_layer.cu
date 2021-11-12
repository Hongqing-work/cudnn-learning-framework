#include "pooling_layer.hpp"

PoolingLayer::PoolingLayer(int batch_size_, int in_channels_, int in_hight_, 
    int in_width_, int kernel_size_, int stride_, int padding_) {
    batch_size = batch_size_;
    in_channels = in_channels_;
    in_height = in_hight_;
    in_width = in_width_;
    kernel_size = kernel_size_;
    stride = stride_;
    padding = padding_;
}

PoolingLayer::~PoolingLayer() {
    // Destroy tensor descriptors
    checkCUDNN(cudnnDestroyTensorDescriptor(src_tensor_desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dst_tensor_desc));
    // Destroy pooling layer descriptor
    checkCUDNN(cudnnDestroyPoolingDescriptor(pooling_desc));
    // Free GPU Memory
    checkCudaErrors(cudaFree(bottom_diff));
    checkCudaErrors(cudaFree(data_out));
}

void PoolingLayer::SetDescriptor() {
    // Set input tensor descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(src_tensor_desc, CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_FLOAT, batch_size, in_channels, in_height, in_width));
    // Set pooling layer descriptor
    const int windowDimA[2] = {kernel_size, kernel_size};
    const int paddingA[2] = {padding, padding};
    const int strideA[2] = {stride, stride};
    checkCUDNN(cudnnSetPoolingNdDescriptor(pooling_desc, CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN, 2, windowDimA, paddingA, strideA));
    // Set output tensor descriptor
    // Use cudnnGetPoolingNdForwardOutputDim to avoid computing by myself
    int tensorOuputDimA[4];
    checkCUDNN(cudnnGetPoolingNdForwardOutputDim(pooling_desc, src_tensor_desc, 
        4, tensorOuputDimA));
    checkCUDNN(cudnnSetTensor4dDescriptor(dst_tensor_desc, CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_FLOAT, tensorOuputDimA[0], tensorOuputDimA[1], 
        tensorOuputDimA[2], tensorOuputDimA[3]));
    out_size = tensorOuputDimA[0] * tensorOuputDimA[1] * tensorOuputDimA[2] * 
        tensorOuputDimA[3];
}

void PoolingLayer::EvalSet() {
    batch_size = 1;
    SetDescriptor();
    checkCudaErrors(cudaFree(data_out));
    checkCudaErrors(cudaMalloc((void**)&data_out, sizeof(float) * out_size));
}

void PoolingLayer::InitParameter() {
    /**
    * Initialization for cudnn settings
    */
    // Create tensor descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&src_tensor_desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dst_tensor_desc));
    // Create pooling layer descriptor
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
    // Set descriptors
    SetDescriptor();

    /**
    * Initialization for data
    */
    int in_size = batch_size * in_channels * in_height * in_width;
    // Allocate memory for data_out and backward diff
    checkCudaErrors(cudaMalloc((void**)&bottom_diff, sizeof(float) * in_size));
    checkCudaErrors(cudaMalloc((void**)&data_out, sizeof(float) * out_size));
}

float* PoolingLayer::forward(float *input, cudnnHandle_t *handle) {
    data_in = input;
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnPoolingForward(*handle, pooling_desc, &alpha, src_tensor_desc, 
        data_in, &beta, dst_tensor_desc, data_out));
    return data_out;
}

float* PoolingLayer::backward(float *top_diff, cudnnHandle_t *handle) {
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnPoolingBackward(*handle, pooling_desc, &alpha, dst_tensor_desc, 
        data_out, dst_tensor_desc, top_diff, src_tensor_desc, data_in, &beta, 
        src_tensor_desc, bottom_diff));
    return bottom_diff;
}