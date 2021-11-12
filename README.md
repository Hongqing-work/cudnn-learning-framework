# cudnn-learning-framework
This is a tiny learning framework built by cudnn and cublas. (cuda-11.0, cudnn-8)

Now I have implemented 5 layers:  
conv_layer  
fully_connected_layer  
pooling_layer  
relu_layer  
softmax_loss_layer  

Therefore, you can already built a CNN by using these layers. There is also an example showing how to use this framework. You should download the Mnist dataset and put them in the top directory. Also, to build the sample program, don't forget to modify the cuda path and cudnn path in Makefile. Then you can make and run the sample program like this:

```bash
make
./Mnist
```
