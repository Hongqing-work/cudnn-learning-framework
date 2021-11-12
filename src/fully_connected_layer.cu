#include "fully_connected_layer.hpp"
#include <cmath>
#include <random>
#include <ctime>

const int block_size = 64;

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void FillOnes(float *vec, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    vec[idx] = 1.0f;
}

FullyConnectedLayer::FullyConnectedLayer(int batch_size_, int in_size_, 
    int out_size_) {
    batch_size = batch_size_;
    in_size = in_size_;
    out_size = out_size_;
}

FullyConnectedLayer::~FullyConnectedLayer() {
    // Free GPU Memory
    checkCudaErrors(cudaFree(weight));
    checkCudaErrors(cudaFree(bias));
    checkCudaErrors(cudaFree(d_weight));
    checkCudaErrors(cudaFree(d_bias));
    checkCudaErrors(cudaFree(bottom_diff));
    checkCudaErrors(cudaFree(data_out));
    checkCudaErrors(cudaFree(onevec));
}

void FullyConnectedLayer::InitParameter() {
    int weight_size = in_size * out_size;
    // Random init weight and bias
    float *host_weight = (float *)malloc(sizeof(float) * weight_size);
    float *host_bias = (float *)malloc(sizeof(float) * out_size);
    // Init host tensor
    std::default_random_engine e(time(0));
    float wfc = sqrt(3.0f / (weight_size));
    std::uniform_real_distribution<> dfc(-wfc, wfc);
    for (int i = 0; i < weight_size; i++) {
        host_weight[i] = dfc(e);
    }
    for (int i = 0; i < out_size; i++) {
        host_bias[i] = dfc(e);
    }
    // Copy to device tensor
    checkCudaErrors(cudaMalloc((void**)&weight, sizeof(float) * weight_size));
    checkCudaErrors(cudaMemcpy(weight, host_weight, sizeof(float) * weight_size, 
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&bias, sizeof(float) * out_size));
    checkCudaErrors(cudaMemcpy(bias, host_bias, sizeof(float) * out_size, 
        cudaMemcpyHostToDevice));
    free(host_weight);
    free(host_bias);
    // Allocate memory for data_out and backward diff
    checkCudaErrors(cudaMalloc((void**)&d_weight, sizeof(float) * weight_size));
    checkCudaErrors(cudaMalloc((void**)&d_bias, sizeof(float) * out_size));
    checkCudaErrors(cudaMalloc((void**)&bottom_diff, sizeof(float) * batch_size 
        * in_size));
    checkCudaErrors(cudaMalloc((void**)&data_out, sizeof(float) * batch_size * 
        out_size));
    checkCudaErrors(cudaMalloc((void**)&onevec, sizeof(float) * batch_size));
    // Create a [1,1,...,1] for bias add
    int grid_size = (batch_size + block_size - 1) / block_size;
    FillOnes<<<grid_size, block_size>>>(onevec, batch_size);
}

void FullyConnectedLayer::EvalSet() {
    batch_size = 1;
    checkCudaErrors(cudaFree(data_out));
    checkCudaErrors(cudaMalloc((void**)&data_out, sizeof(float) * out_size));
    checkCudaErrors(cudaFree(onevec));
    checkCudaErrors(cudaMalloc((void**)&onevec, sizeof(float)));
    // Create a [1,1,...,1] for bias add
    int grid_size = (batch_size + block_size - 1) / block_size;
    FillOnes<<<grid_size, block_size>>>(onevec, batch_size);
}

float* FullyConnectedLayer::forward(float *input, cublasHandle_t *handle) {
    data_in = input;
    // cublas is col major , so use C_T = B_T * A_T
    float alpha = 1.0f, beta = 0.0f;
    checkCublasErrors(cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, out_size, 
        batch_size, in_size, &alpha, weight, out_size, data_in, in_size, 
        &beta, data_out, out_size));
    // Add bias using onevec
    checkCublasErrors(cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, out_size, 
        batch_size, 1, &alpha, bias, out_size, onevec, 1, &alpha, data_out, 
        out_size));
    return data_out;
}

float* FullyConnectedLayer::backward(float *top_diff, cublasHandle_t *handle) {
    float alpha = 1.0f, beta = 0.0f;
    // bias backward, need to * onevec
    checkCublasErrors(cublasSgemv(*handle, CUBLAS_OP_N, out_size, batch_size, 
        &alpha, top_diff, out_size, onevec, 1, &beta, d_bias, 1));
    // weight backward
    checkCublasErrors(cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, out_size, 
        in_size, batch_size, &alpha, top_diff, out_size, data_in, in_size, 
        &beta, d_weight, out_size));
    // data backward
    checkCublasErrors(cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, in_size, 
        batch_size, out_size, &alpha, weight, out_size, top_diff, out_size, 
        &beta, bottom_diff, in_size));
    return bottom_diff;
}

void FullyConnectedLayer::UpdateWeights(float lr, cublasHandle_t *handle) {
    int weight_size = in_size * out_size;
    float alpha = -lr;
    checkCublasErrors(cublasSaxpy(*handle, weight_size, &alpha, d_weight, 1, 
        weight, 1));
    checkCublasErrors(cublasSaxpy(*handle, out_size, &alpha, d_bias, 1, bias, 1));
}