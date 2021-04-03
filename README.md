# Lenet5-from-scratch

This is the implementation of Lenet-5 by Yann LeCun at AT&T Bell-Labs

Convolutional Neural Networks are a special kind of multi-layer neural networks. Like almost every other neural networks they are trained with a version of the back-propagation algorithm. Where they differ is in the architecture.  
Convolutional Neural Networks are designed to recognize visual patterns directly from pixel images with minimal preprocessing. 
They can recognize patterns with extreme variability (such as handwritten characters), and with robustness to distortions and simple geometric transformations.  

| Type | Accuracy |
| ------ | ------|
| Train  | 76%	| 
| Test | 65%   |

![alt text](lenet_arc.png)

The first layer is the input layer — this is generally not considered a layer of the network as nothing is learnt in this layer. The input layer is built to take in 32x32, and these are the dimensions of images that are passed into the next layer. Those who are familiar with the MNIST dataset will be aware that the MNIST dataset images have the dimensions 28x28. To get the MNIST images dimension to the meet the requirements of the input layer, the 28x28 images are padded.
The grayscale images used in the research paper had their pixel values normalized from 0 to 255, to values between -0.1 and 1.175. The reason for normalization is to ensure that the batch of images have a mean of 0 and a standard deviation of 1, the benefits of this is seen in the reduction in the amount of training time. In the image classification with LeNet-5 example below, we’ll be normalizing the pixel values of the images to take on values between 0 to 1.

The first layer is the input layer — this is generally not considered a layer of the network as nothing is learnt in this layer. The input layer is built to take in 32x32, and these are the dimensions of images that are passed into the next layer. Those who are familiar with the MNIST dataset will be aware that the MNIST dataset images have the dimensions 28x28. To get the MNIST images dimension to the meet the requirements of the input layer, the 28x28 images are padded.
The grayscale images used in the research paper had their pixel values normalized from 0 to 255, to values between -0.1 and 1.175. The reason for normalization is to ensure that the batch of images have a mean of 0 and a standard deviation of 1, the benefits of this is seen in the reduction in the amount of training time. In the image classification with LeNet-5 example below, we’ll be normalizing the pixel values of the images to take on values between 0 to 1.
The LeNet-5 architecture utilizes two significant types of layer construct: convolutional layers and subsampling layers.
Convolutional layers
Sub-sampling layers
Within the research paper and the image below, convolutional layers are identified with the ‘Cx’, and subsampling layers are identified with ‘Sx’, where ‘x’ is the sequential position of the layer within the architecture. ‘Fx’ is used to identify fully connected layers. This method of layer identification can be seen in the image above.
The official first layer convolutional layer C1 produces as output 6 feature maps, and has a kernel size of 5x5. The kernel/filter is the name given to the window that contains the weight values that are utilized during the convolution of the weight values with the input values. 5x5 is also indicative of the local receptive field size each unit or neuron within a convolutional layer. The dimensions of the six feature maps the first convolution layer produces are 28x28.
A subsampling layer ‘S2’ follows the ‘C1’ layer’. The ‘S2’ layer halves the dimension of the feature maps it receives from the previous layer; this is known commonly as downsampling.
The ‘S2’ layer also produces 6 feature maps, each one corresponding to the feature maps passed as input from the previous layer. This link contains more information on subsampling layers.

### Below is a table that summarises the key features of each layer:

![alt text](table.png)

### Dataset

We load the MNIST dataset using the Keras library. The Keras library has a suite of datasets readily available for use with easy accessibility.


### Procedure
Install the dependencies and devDependencies and start running knn_predict.py.

```sh
$ cd Lenet5-from-scratch
$ python lenet5.py
```

### Todos

 - Add RELU for better accuracy
 - Add optimizers with batch norm


### Development

Want to contribute? Great!
You can [contact](mailto:shubhpachchigar@gmail.com) me for any suggestion or feedback!


License
----

MIT