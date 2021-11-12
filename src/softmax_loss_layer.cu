#include "softmax_loss_layer.hpp"

const int block_size = 64;

/**
 * Computes the Softmax loss for a batch
 * Uses the softmax values obtained from forward propagation to compute.
 *
 * @param label The training batch label values.
 * @param num_labels The number of possible labels.
 * @param batch_size The size of the trained batch.
 * @param data_out The probability result.
 * @param partial_sum The partial loss array.
 */
__global__ void SoftmaxGetLoss(const float *label, int num_labels, 
    int batch_size, float *data_out, float *partial_sum) {
    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size)
        return;
    const int label_value = static_cast<int>(label[tid]);
    int cacheIndex = threadIdx.x;
    cache[cacheIndex] = -log(data_out[tid * num_labels+ label_value]);
    __syncthreads();

    // reduction
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0) {
        partial_sum[blockIdx.x] = cache[0];
    }
}

/**
 * Computes the backpropagation results of the Softmax loss for each result in a batch.
 * Uses the softmax values obtained from forward propagation to compute the difference.
 *
 * @param label The training batch label values.
 * @param num_labels The number of possible labels.
 * @param batch_size The size of the trained batch.
 * @param bottom_diff The resulting gradient.
 */
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, 
    int batch_size, float *bottom_diff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;
    const int label_value = static_cast<int>(label[idx]);

    // For each item in the batch, decrease the result of the label's value by 1
    bottom_diff[idx * num_labels + label_value] -= 1.0f;
}

SoftmaxLossLayer::SoftmaxLossLayer(int batch_size_, int data_size_) {
    batch_size = batch_size_;
    data_size = data_size_;
}

SoftmaxLossLayer::~SoftmaxLossLayer() {
    // Destroy tensor descriptors
    checkCUDNN(cudnnDestroyTensorDescriptor(src_tensor_desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dst_tensor_desc));
    // Free GPU Memory
    checkCudaErrors(cudaFree(bottom_diff));
    checkCudaErrors(cudaFree(data_out));
}

void SoftmaxLossLayer::SetDescriptor() {
    checkCUDNN(cudnnSetTensor4dDescriptor(src_tensor_desc, CUDNN_TENSOR_NCHW, 
        CUDNN_DATA_FLOAT, batch_size, data_size, 1, 1));
    checkCUDNN(cudnnSetTensor4dDescriptor(dst_tensor_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, batch_size, data_size, 1, 1));
}

void SoftmaxLossLayer::EvalSet() {
    batch_size = 1;
    SetDescriptor();
    checkCudaErrors(cudaFree(data_out));
    checkCudaErrors(cudaMalloc((void**)&data_out, sizeof(float) * data_size));
}

void SoftmaxLossLayer::InitParameter() {
    /**
    * Initialization for cudnn settings
    */
    // Create tensor descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&src_tensor_desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dst_tensor_desc));
    // Set tensor descriptor
    SetDescriptor();
    
    /**
    * Initialization for data
    */
    checkCudaErrors(cudaMalloc((void**)&bottom_diff, sizeof(float) * batch_size * 
        data_size));
    checkCudaErrors(cudaMalloc((void**)&data_out, sizeof(float) * batch_size * 
        data_size));
}

float* SoftmaxLossLayer::forward(float *input, cudnnHandle_t *handle) {
    data_in = input;
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnSoftmaxForward(*handle, CUDNN_SOFTMAX_ACCURATE, 
        CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, src_tensor_desc, data_in, &beta, 
        dst_tensor_desc, data_out));
    return data_out;
}

float SoftmaxLossLayer::getloss(float *label) {
    int grid_size = (batch_size + block_size - 1) / block_size;
    float *cross_entropy_partial_sum;
    checkCudaErrors(cudaMalloc((void**)&cross_entropy_partial_sum, 
        sizeof(float) * grid_size));
    int mem_size = sizeof(float) * grid_size;
    SoftmaxGetLoss<<<grid_size, block_size, mem_size>>>(label, data_size, 
        batch_size, data_out, cross_entropy_partial_sum);
    float *host_partial_sum = (float *)malloc(sizeof(float) * grid_size);
    checkCudaErrors(cudaMemcpy(host_partial_sum, cross_entropy_partial_sum, 
        sizeof(float) * grid_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(cross_entropy_partial_sum));
    float loss = 0.0f;
    for (int i = 0; i < grid_size; ++i) {
        loss+= host_partial_sum[i];
    }
    return loss/batch_size;
}

float* SoftmaxLossLayer::backward(float *label) {
    checkCudaErrors(cudaMemcpyAsync(bottom_diff, data_out, sizeof(float) * 
        batch_size * data_size, cudaMemcpyDeviceToDevice));
    int grid_size = (batch_size + block_size - 1) / block_size;
    SoftmaxLossBackprop<<<grid_size, block_size>>>(label, data_size, 
        batch_size, bottom_diff);
    return bottom_diff;
}