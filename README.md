# UltarSonic.tech

#### Trained with _`GPU P100`_

- _`import tensorflow as tf`_: Import `TensorFlow` library and use the alias `tf`. TensorFlow is an open-source library used for numerical computation and machine learning tasks, particularly deep learning.

+ `from tensorflow.keras.models import Sequential`: Import the `Sequential` class from the `tensorflow.keras.models` module. The Sequential class is used for creating a linear stack of layers in a deep learning model.

- `from tensorflow.keras.layers import Conv2D, LayerNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Activation`: Import various layer classes from the `tensorflow.keras.layers` module. These layers will be used to build the architecture of the deep learning model.

- `Conv2D`: Convolutional layer for 2D spatial convolution.
- `LayerNormalization`: Normalizes the activations of the previous layer.
- `MaxPooling2D`: Downsamples the input along its spatial dimensions.
- `Dropout`: Applies dropout to the input, a regularization technique to prevent overfitting.
- `GlobalAveragePooling2D`: Averages the spatial dimensions of a 3D tensor.
- `Dense`: Fully-connected layer for input-output transformation.
- `Activation`: Applies an activation function to the output.

- `from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau`: Import callback classes from the `keras.callbacks` module. Callbacks are functions that can be applied at certain stages of the training process.

- `EarlyStopping`: Stops training when a monitored quantity has stopped improving.
- `ModelCheckpoint`: Saves the model after every epoch.
- `ReduceLROnPlateau`: Reduces learning rate when a metric has stopped improving.

- `from tensorflow.keras.preprocessing.image import ImageDataGenerator`: Import the `ImageDataGenerator` class from the `tensorflow.keras.preprocessing.image` module. This class is used for real-time data augmentation during training.

- `from keras.regularizers import l2`: Import the L2 regularization function (`l2`) from the `keras.regularizers` module. L2 regularization is a technique used to penalize large weights in the model, preventing overfitting.

- `from keras.optimizers import Adam`: Import the `Adam` optimizer class from the `keras.optimizers` module. The Adam optimizer is an adaptive learning rate optimization algorithm used for training deep learning models.

- `import matplotlib.pyplot as plt`: Import the `pyplot` module from the `matplotlib` library and use the alias `plt`. Matplotlib is a plotting library used for creating static, animated, and interactive visualizations in Python.

- `import numpy`: Import the `NumPy` library. NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
