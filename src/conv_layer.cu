#include "conv_layer.hpp"
#include <cmath>
#include <random>
#include <ctime>

ConvLayer::ConvLayer(int batch_size_, int in_channels_, int in_hight_, 
    int in_width_, int out_channels_, int kernel_size_, int stride_, 
    int padding_) {
    batch_size = batch_size_;
    in_channels = in_channels_;
    in_height = in_hight_;
    in_width = in_width_;
    out_channels = out_channels_;
    kernel_size = kernel_size_;
    stride = stride_;
    padding = padding_;
}

ConvLayer::~ConvLayer() {
    // Destroy tensor descriptors
    checkCUDNN(cudnnDestroyTensorDescriptor(src_tensor_desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dst_tensor_desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(bias_tensor_desc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    // Destroy conv layer descriptors
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    // Free GPU Memory
    checkCudaErrors(cudaFree(weight));
    checkCudaErrors(cudaFree(bias));
    checkCudaErrors(cudaFree(d_weight));
    checkCudaErrors(cudaFree(d_bias));
    checkCudaErrors(cudaFree(bottom_diff));
    checkCudaErrors(cudaFree(data_out));
}

void ConvLayer::SetDescriptor() {
    // Set input, bias and filter descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(src_tensor_desc, CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_FLOAT, batch_size, in_channels, in_height, in_width));
    checkCUDNN(cudnnSetTensor4dDescriptor(bias_tensor_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, out_channels, 1, 1));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, out_channels, in_channels, kernel_size, kernel_size));
    // Set conv layer descriptor
    const int padA[2] = {padding, padding};
    const int filterStrideA[2] = {stride, stride};
    const int upscaleA[2] = {1, 1};
    checkCUDNN(cudnnSetConvolutionNdDescriptor(conv_desc, 2, padA, filterStrideA,
        upscaleA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    // Set output tensor descriptor
    // Use cudnnGetConvolutionNdForwardOutputDim to avoid computing by myself
    int tensorOuputDimA[4];
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(conv_desc, src_tensor_desc, 
        filter_desc, 4, tensorOuputDimA));
    checkCUDNN(cudnnSetTensor4dDescriptor(dst_tensor_desc, CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_FLOAT, tensorOuputDimA[0], tensorOuputDimA[1], 
        tensorOuputDimA[2], tensorOuputDimA[3]));
    out_size = tensorOuputDimA[0] * tensorOuputDimA[1] * tensorOuputDimA[2] * 
        tensorOuputDimA[3];
}

void ConvLayer::SetAlgorithm(cudnnHandle_t *handle) {
    // Set forward algorithm
    int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount = -1;
    cudnnConvolutionFwdAlgoPerf_t fwd_results[2 * requestedAlgoCount];
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(*handle, src_tensor_desc, 
        filter_desc, conv_desc, dst_tensor_desc, requestedAlgoCount, 
        &returnedAlgoCount, fwd_results));
    fwd_algo = fwd_results[0].algo;
    // Set backward algorithm
    // for filter
    requestedAlgoCount = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    returnedAlgoCount = -1;
    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_results[2 * requestedAlgoCount];
    checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(*handle, src_tensor_desc, 
        dst_tensor_desc, conv_desc, filter_desc, requestedAlgoCount, 
        &returnedAlgoCount, bwd_filter_results));
    bwd_filter_algo = bwd_filter_results[0].algo;
    // for data
    requestedAlgoCount = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    returnedAlgoCount = -1;
    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_results[2 * requestedAlgoCount];
    checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(*handle, filter_desc, 
        dst_tensor_desc, conv_desc, src_tensor_desc, requestedAlgoCount, 
        &returnedAlgoCount, bwd_data_results));
    bwd_data_algo = bwd_data_results[0].algo;
}

void ConvLayer::EvalSet() {
    batch_size = 1;
    SetDescriptor();
    checkCudaErrors(cudaFree(data_out));
    checkCudaErrors(cudaMalloc((void**)&data_out, sizeof(float) * out_size));
}

void ConvLayer::InitParameter(cudnnHandle_t *handle) {
    /**
    * Initialization for cudnn settings
    */
    // Create tensor descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&src_tensor_desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dst_tensor_desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&bias_tensor_desc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    // Create conv layer descriptors
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    // Set descriptors
    SetDescriptor();
    // Set algorithm
    SetAlgorithm(handle);
    /**
    * Initialization for data
    */
    int in_size = batch_size * in_channels * in_height * in_width;
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    // Random init weight and bias
    float *host_weight = (float *)malloc(sizeof(float) * weight_size);
    float *host_bias = (float *)malloc(sizeof(float) * out_channels);
    // Init host tensor
    std::default_random_engine e(time(0));
    float wconv = sqrt(3.0f / (kernel_size * kernel_size * in_channels));
    std::uniform_real_distribution<> dconv(-wconv, wconv);
    for (int i = 0; i < weight_size; i++) {
        host_weight[i] = dconv(e);
    }
    for (int i = 0; i < out_channels; i++) {
        host_bias[i] = dconv(e);
    }
    // Copy to device tensor
    checkCudaErrors(cudaMalloc((void**)&weight, sizeof(float) * weight_size));
    checkCudaErrors(cudaMemcpy(weight, host_weight, sizeof(float) * weight_size, 
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&bias, sizeof(float) * out_channels));
    checkCudaErrors(cudaMemcpy(bias, host_bias, sizeof(float) * out_channels, 
        cudaMemcpyHostToDevice));
    free(host_weight);
    free(host_bias);
    // Allocate memory for data_out and backward diff
    checkCudaErrors(cudaMalloc((void**)&d_weight, sizeof(float) * weight_size));
    checkCudaErrors(cudaMalloc((void**)&d_bias, sizeof(float) * out_channels));
    checkCudaErrors(cudaMalloc((void**)&bottom_diff, sizeof(float) * in_size));
    checkCudaErrors(cudaMalloc((void**)&data_out, sizeof(float) * out_size));
}

float* ConvLayer::forward(float *input, cudnnHandle_t *handle) {
    data_in = input;
    // allocate workspace
    size_t sizeInBytes=0;
    void* workSpace=NULL;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(*handle, src_tensor_desc, 
        filter_desc, conv_desc, dst_tensor_desc, fwd_algo, &sizeInBytes));
    if (sizeInBytes!=0) {
        checkCudaErrors(cudaMalloc(&workSpace, sizeInBytes));
    }
    // conv
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(*handle, &alpha, src_tensor_desc,
        data_in, filter_desc, weight, conv_desc, fwd_algo, workSpace, sizeInBytes, 
        &beta, dst_tensor_desc, data_out));
    // addBias
    checkCUDNN(cudnnAddTensor(*handle, &alpha, bias_tensor_desc, bias, &alpha,
        dst_tensor_desc, data_out));
    if (sizeInBytes!=0) {
        checkCudaErrors(cudaFree(workSpace));
    }
    return data_out;
}

float* ConvLayer::backward(float *top_diff, cudnnHandle_t *handle) {
    // allocate workspace for cudnnConvolutionBackward
    size_t sizeInBytes = 0;
    size_t sizeTempBytes = 0;
    void* workSpace=NULL;
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(*handle, 
        src_tensor_desc, dst_tensor_desc, conv_desc, filter_desc, 
        bwd_filter_algo, &sizeTempBytes));
    sizeInBytes = sizeTempBytes;
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(*handle, 
        filter_desc, dst_tensor_desc, conv_desc, src_tensor_desc, 
        bwd_data_algo, &sizeTempBytes));
    sizeInBytes = sizeInBytes > sizeTempBytes ? sizeInBytes : sizeTempBytes;
    if (sizeInBytes!=0) {
        checkCudaErrors(cudaMalloc(&workSpace, sizeInBytes));
    }

    float alpha = 1.0f, beta = 0.0f;
    // bias backward
    checkCUDNN(cudnnConvolutionBackwardBias(*handle, &alpha, dst_tensor_desc, 
        top_diff, &beta, bias_tensor_desc, d_bias));
    // weight backward
    checkCUDNN(cudnnConvolutionBackwardFilter(*handle, &alpha, src_tensor_desc, 
        data_in, dst_tensor_desc, top_diff, conv_desc, bwd_filter_algo, 
        workSpace, sizeInBytes, &beta, filter_desc, d_weight));
    // data backward
    checkCUDNN(cudnnConvolutionBackwardData(*handle, &alpha, filter_desc, weight, 
        dst_tensor_desc, top_diff, conv_desc, bwd_data_algo, workSpace, sizeInBytes,
        &beta, src_tensor_desc, bottom_diff));
    if (sizeInBytes!=0) {
        checkCudaErrors(cudaFree(workSpace));
    }
    data_in = nullptr;
    return bottom_diff;
}

void ConvLayer::UpdateWeights(float lr, cublasHandle_t *handle) {
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    float alpha = -lr;
    checkCublasErrors(cublasSaxpy(*handle, weight_size, &alpha, d_weight, 1, 
        weight, 1));
    checkCublasErrors(cublasSaxpy(*handle, out_channels, &alpha, d_bias, 1, 
        bias, 1));
}