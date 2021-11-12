#include "relu_layer.hpp"

ReluLayer::ReluLayer(int batch_size_, int in_size_) {
    batch_size = batch_size_;
    in_size = in_size_;
}

ReluLayer::~ReluLayer() {
    // Destroy tensor descriptors
    checkCUDNN(cudnnDestroyTensorDescriptor(src_tensor_desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dst_tensor_desc));
    // Destroy activation layer descriptor
    checkCUDNN(cudnnDestroyActivationDescriptor(activation_desc));
    // Free GPU Memory
    checkCudaErrors(cudaFree(bottom_diff));
    checkCudaErrors(cudaFree(data_out));
}

void ReluLayer::SetDescriptor() {
    // Set tensor descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(src_tensor_desc, CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_FLOAT, batch_size, in_size, 1, 1));
    checkCUDNN(cudnnSetTensor4dDescriptor(dst_tensor_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, batch_size, in_size, 1, 1));
    // Set activation layer descriptor
    checkCUDNN(cudnnSetActivationDescriptor(activation_desc, 
        CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
}

void ReluLayer::EvalSet() {
    batch_size = 1;
    SetDescriptor();
    checkCudaErrors(cudaFree(data_out));
    checkCudaErrors(cudaMalloc((void**)&data_out, sizeof(float) * in_size));
}

void ReluLayer::InitParameter() {
    /**
    * Initialization for cudnn settings
    */
    // Create tensor descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&src_tensor_desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dst_tensor_desc));
    // Create activation layer descriptor
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_desc));
    // Set descriptors
    SetDescriptor();
    
    /**
    * Initialization for data
    */
    // Allocate memory for data_out and backward diff
    int size = batch_size * in_size;
    checkCudaErrors(cudaMalloc((void**)&bottom_diff, sizeof(float) * size));
    checkCudaErrors(cudaMalloc((void**)&data_out, sizeof(float) * size));
}

float* ReluLayer::forward(float *input, cudnnHandle_t *handle) {
    data_in = input;
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationForward(*handle, activation_desc, &alpha, 
        src_tensor_desc, data_in, &beta, dst_tensor_desc, data_out));
    return data_out;
}

float* ReluLayer::backward(float *top_diff, cudnnHandle_t *handle) {
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationBackward(*handle, activation_desc, &alpha, 
        dst_tensor_desc, data_out, dst_tensor_desc, top_diff, src_tensor_desc, 
        data_in, &beta, src_tensor_desc, bottom_diff));
    return bottom_diff;
}