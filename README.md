# UltarSonic.

We have designed a `Convolutional Neural Network (CNN)` using `TensorFlow's Keras API` to address a specific classification problem; 
`Pneumonia & Lung Tumors Classification`.

The architecture of the model is carefully chosen to effectively learn hierarchical features from the input images and perform accurate classification tasks. The model consists of three pairs of `convolutional` and `max-pooling` layers, a `flatten` layer, and two `dense` layers, with each layer serving a specific purpose and having selected parameters to optimize performance.

The first `convolutional layer` has:
  * `32 filters`;
  * `3x3 kernel size`; 
  * `ReLU activation function`; 
  * and an `input shape of (224, 224, 3)` representing 224x224 pixel images with 3 color channels (RGB). 

We chose 32 filters as a starting point to balance computational efficiency and the ability to learn various low-level features. The `3x3 kernel` size is a common choice because it can effectively capture local patterns while reducing computational complexity compared to larger kernel sizes.

The first `max-pooling layer` has a 2x2 pool size, which reduces the spatial dimensions of the feature maps by half, making the model more computationally efficient and robust to small translations in the input images.

The second and third convolutional layers have 64 and 128 filters, respectively, both with 3x3 kernel sizes and ReLU activation functions. Increasing the number of filters in deeper layers allows the model to learn more complex and higher-level features from the data. The second and third max-pooling layers also have 2x2 pool sizes, further reducing the spatial dimensions and increasing the model's translation invariance.

After the third max-pooling layer, we use a flatten layer to convert the output feature maps into a one-dimensional tensor, which is necessary to connect the convolutional layers to the subsequent dense layers.

The first dense (fully connected) layer has 128 nodes and a ReLU activation function. We chose 128 nodes to strike a balance between the model's capacity to learn complex relationships in the data and its computational efficiency.

Finally, the output layer is a dense layer with 3 nodes and a softmax activation function, which provides a probability distribution over the target classes. The choice of 3 nodes corresponds to the number of classes in our specific classification problem, and the softmax activation ensures that the output probabilities sum to one.

In summary, the architecture and selected parameters of our CNN model are designed to effectively learn hierarchical features from the input images while maintaining computational efficiency and robustness to variations in the input data, ultimately leading to accurate classification performance.
