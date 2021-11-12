#include "utils.hpp"
#include "error_util.hpp"
#include "conv_layer.hpp"
#include "fully_connected_layer.hpp"
#include "pooling_layer.hpp"
#include "relu_layer.hpp"
#include "softmax_loss_layer.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

const int batch_size = 128;
const int max_iteration = 300;
const float learning_rate = 0.0005;
const float lr_gamma = 0.0001;
const float lr_power = 0.75;

// layer pointers
ConvLayer *conv1;
ReluLayer *relu1;
PoolingLayer *pool1;
ConvLayer *conv2;
ReluLayer *relu2;
PoolingLayer *pool2;
FullyConnectedLayer *fc1;
FullyConnectedLayer *fc2;
SoftmaxLossLayer *softmax;

void CreateNetwork(cudnnHandle_t *handle) {
    conv1 = new ConvLayer(batch_size, 1, 28, 28, 3, 3, 1, 1);
    relu1 = new ReluLayer(batch_size, 3 * 28 * 28);
    pool1 = new PoolingLayer(batch_size, 3, 28, 28, 2, 2, 0);
    conv2 = new ConvLayer(batch_size, 3, 14, 14, 6, 3, 1, 1);
    relu2 = new ReluLayer(batch_size, 6 * 14 * 14);
    pool2 = new PoolingLayer(batch_size, 6, 14, 14, 2, 2, 0);
    fc1 = new FullyConnectedLayer(batch_size, 6 * 7 * 7, 64);
    fc2 = new FullyConnectedLayer(batch_size, 64, 10);
    softmax = new SoftmaxLossLayer(batch_size, 10);
    // Initialization for layers
    conv1->InitParameter(handle);
    relu1->InitParameter();
    pool1->InitParameter();
    conv2->InitParameter(handle);
    relu2->InitParameter();
    pool2->InitParameter();
    fc1->InitParameter();
    fc2->InitParameter();
    softmax->InitParameter();
}

void Training(cudnnHandle_t *cudnnHandle, cublasHandle_t *cublasHandle, 
    float *train_data, float *train_label, int train_size) {
    // Create GPU data structures and prepare dataset in one batch
    float *device_data_ptr = nullptr, *device_label_ptr = nullptr;
    checkCudaErrors(cudaMalloc((void**)&device_data_ptr, sizeof(float) * 
        batch_size * 28 * 28));
    checkCudaErrors(cudaMalloc((void**)&device_label_ptr, sizeof(float) * 
        batch_size));
    printf("Training...\n");
    // Use SGD to train the network
    checkCudaErrors(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < max_iteration; ++iter) {
        printf("Iteration number : %d, ", iter);
        int imageid = iter % ((train_size  - batch_size + 1)/ batch_size);
        // Prepare current batch on device
        checkCudaErrors(cudaMemcpy(device_data_ptr, train_data + imageid * 
            batch_size * 28 * 28, sizeof(float) * batch_size * 28 * 28, 
            cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_label_ptr, train_label + imageid * 
            batch_size, sizeof(float) * batch_size, cudaMemcpyHostToDevice));
        // Forward propagation
        float *conv1_out = conv1->forward(device_data_ptr, cudnnHandle);
        float *relu1_out = relu1->forward(conv1_out, cudnnHandle);
        float *pool1_out = pool1->forward(relu1_out, cudnnHandle);
        float *conv2_out = conv2->forward(pool1_out, cudnnHandle);
        float *relu2_out = relu2->forward(conv2_out, cudnnHandle);
        float *pool2_out = pool2->forward(relu2_out, cudnnHandle);
        float *fc1_out = fc1->forward(pool2_out, cublasHandle);
        float *fc2_out = fc2->forward(fc1_out, cublasHandle);
        float *result = softmax->forward(fc2_out, cudnnHandle);
        printf("loss : %f \n", softmax->getloss(device_label_ptr));
        
        // Backward propagation
        float *softmax_back = softmax->backward(device_label_ptr);
        float *fc2_back = fc2->backward(softmax_back, cublasHandle);
        float *fc1_back = fc1->backward(fc2_back, cublasHandle);
        float *pool2_back = pool2->backward(fc1_back, cudnnHandle);
        float *relu2_back = relu2->backward(pool2_back, cudnnHandle);
        float *conv2_back = conv2->backward(relu2_back, cudnnHandle);
        float *pool1_back = pool1->backward(conv2_back, cudnnHandle);
        float *relu1_back = relu1->backward(pool1_back, cudnnHandle);
        conv1->backward(relu1_back, cudnnHandle);

        // Compute learning rate
        float lr = learning_rate * pow((1.0 + lr_gamma * iter), (lr_power));
        // Update weights
        conv1->UpdateWeights(lr, cublasHandle);
        conv2->UpdateWeights(lr, cublasHandle);
        fc1->UpdateWeights(lr, cublasHandle);
        fc2->UpdateWeights(lr, cublasHandle);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("Iteration time: %f ms\n", 
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() 
        / 1000.0f / max_iteration);
    checkCudaErrors(cudaFree(device_data_ptr));
    checkCudaErrors(cudaFree(device_label_ptr));
}

void EvalSet() {
    conv1->EvalSet();
    relu1->EvalSet();
    pool1->EvalSet();
    conv2->EvalSet();
    relu2->EvalSet();
    pool2->EvalSet();
    fc1->EvalSet();
    fc2->EvalSet();
    softmax->EvalSet();
}

void Testing(cudnnHandle_t *cudnnHandle, cublasHandle_t *cublasHandle, 
    float *test_data, float *test_label, int test_size) {
    // Create GPU data structures and prepare dataset in one batch
    float *device_data_ptr = nullptr;
    checkCudaErrors(cudaMalloc((void**)&device_data_ptr, sizeof(float) * 28 * 28));
    printf("Testing...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    int correct_count = 0;
    
    for (int test_id = 0; test_id < test_size; ++test_id) {
        // Prepare current batch on device
        checkCudaErrors(cudaMemcpy(device_data_ptr, test_data + test_id * 28 * 28, 
            sizeof(float) * 28 * 28, cudaMemcpyHostToDevice));
        // Forward propagations
        float *conv1_out = conv1->forward(device_data_ptr, cudnnHandle);
        float *relu1_out = relu1->forward(conv1_out, cudnnHandle);       
        float *pool1_out = pool1->forward(relu1_out, cudnnHandle);
        float *conv2_out = conv2->forward(pool1_out, cudnnHandle);
        float *relu2_out = relu2->forward(conv2_out, cudnnHandle);
        float *pool2_out = pool2->forward(relu2_out, cudnnHandle);
        float *fc1_out = fc1->forward(pool2_out, cublasHandle);
        float *fc2_out = fc2->forward(fc1_out, cublasHandle);
        float *result = softmax->forward(fc2_out, cudnnHandle);
        float *host_result = (float *)malloc(10 * sizeof(float));
        checkCudaErrors(cudaMemcpy(host_result, result, sizeof(float) * 10, 
            cudaMemcpyDeviceToHost));
        // Get the prediction
        int predict_num = 0;
        for (int i = 1; i < 10; ++i) {
            if (host_result[i] > host_result[predict_num]) {
                predict_num = i;
            }
        }
        // check the label
        float error = predict_num - test_label[test_id];
        if (error > -0.1 && error < 0.1) {
            correct_count++;
        }
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    printf("accuracy: %f\n", correct_count * 1.0 / test_size);
    checkCudaErrors(cudaFree(device_data_ptr));
}

int main() {
    /* 
    * Initialization of dataset
    */
    size_t width, height, channels = 1;
    printf("Reading input data\n");
    // Read dataset sizes
    // Mnist 28 * 28
    size_t train_size = ReadUByteDataset("train-images-idx3-ubyte", 
        "train-labels-idx1-ubyte", nullptr, nullptr, width, height);
    size_t test_size = ReadUByteDataset("t10k-images-idx3-ubyte", 
        "t10k-labels-idx1-ubyte", nullptr, nullptr, width, height);
    if (train_size == 0)
        return 1;
    // Allocate host memory for data_in
    std::vector<uint8_t> train_images(train_size * width * height * channels);
    std::vector<uint8_t> train_labels(train_size);
    std::vector<uint8_t> test_images(test_size * width * height * channels);
    std::vector<uint8_t> test_labels(test_size);
    // Read data from datasets
    if (ReadUByteDataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 
        &train_images[0], &train_labels[0], width, height) != train_size) {
        return 2;
    }
    if (ReadUByteDataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 
        &test_images[0], &test_labels[0], width, height) != test_size) {
        return 3;
    }
    printf("Preparing dataset\n");    
    // Normalize training set to be in [0,1]
    printf("Done. Training dataset size: %d, Test dataset size: %d\n", 
        (int)train_size, (int)test_size);
    printf("Batch size: %lld, iterations: %d\n", batch_size, max_iteration);

    // Create the network architecture
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
    checkCublasErrors(cublasCreate(&cublasHandle));
    checkCUDNN(cudnnCreate(&cudnnHandle));
    CreateNetwork(&cudnnHandle);
    
    // Training
    float *train_images_float = (float *)malloc(sizeof(float) * 
        train_images.size());
    float *train_labels_float = (float *)malloc(sizeof(float) * train_size);
    for (size_t i = 0; i < train_size * 28 * 28; ++i)
        train_images_float[i] = (float)train_images[i] / 255.0f;
    for (size_t i = 0; i < train_size; ++i)
        train_labels_float[i] = (float)train_labels[i];
    Training(&cudnnHandle, &cublasHandle, train_images_float, 
        train_labels_float, train_size);
    // Free training data
    free(train_images_float);
    free(train_labels_float);

    // Evaluation: batch_size = 1
    EvalSet();
    float *test_images_float = (float *)malloc(sizeof(float) * 
        test_images.size());
    float *test_labels_float = (float *)malloc(sizeof(float) * test_size);
    for (size_t i = 0; i < test_size * 28 * 28; ++i)
        test_images_float[i] = (float)test_images[i] / 255.0f;
    for (size_t i = 0; i < test_size; ++i)
        test_labels_float[i] = (float)test_labels[i];
    Testing(&cudnnHandle, &cublasHandle, test_images_float, test_labels_float, 
        test_size);
    // Free testing data
    free(test_images_float);
    free(test_labels_float);

    // Free layer structures
    free(conv1);
    free(relu1);
    free(pool1);
    free(conv2);
    free(relu2);
    free(pool2);
    free(fc1);
    free(fc2);
    free(softmax);
    checkCublasErrors(cublasDestroy(cublasHandle));
    checkCUDNN(cudnnDestroy(cudnnHandle));
    
    return 0;
}