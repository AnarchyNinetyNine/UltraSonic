import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LayerNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import numpy as np
from sklearn.metrics import confusion_matrix

# Initialize global variables for each layer type
convolution_variable = numpy.array(0)
activation_variable = numpy.array(0, dtype = numpy.int8)
dropout_variable = numpy.array(0, dtype = numpy.int8)
dense_variable = numpy.array(0, dtype = numpy.int8)
max_pooling_variable = numpy.array(0, dtype = numpy.int8)

# Define the function to generate unique names for Convolutional layers
def Convolutional_Layer_Name_Generator():
    global convolution_variable                                       # Access the global variable 'convolution_variable'
    convolution_variable += 1                                         # Increment the variable by 1
    Layer_name = "Convolutional_Layer_" + str(convolution_variable)   # Create the layer name
    return Layer_name                                                 # Return the generated name

# Define the function to generate unique names for Activation layers
def Activation_Layer_Name_Generator():
    global activation_variable                                        # Access the global variable 'activation_variable'
    activation_variable += 1                                          # Increment the variable by 1
    Layer_name = "Activation_Layer_" + str(activation_variable)       # Create the layer name
    return Layer_name                                                 # Return the generated name

# Define the function to generate unique names for Dense layers
def Dense_Layer_Name_Generator():
    global dense_variable                                             # Access the global variable 'dense_variable'
    dense_variable += 1                                               # Increment the variable by 1
    Layer_name = "Dense_Layer_" + str(dense_variable)                 # Create the layer name
    return Layer_name                                                 # Return the generated name

# Define the function to generate unique names for Dropout layers
def Dropout_Layer_Name_Generator():
    global dropout_variable                                           # Access the global variable 'dropout_variable'
    dropout_variable += 1                                             # Increment the variable by 1
    Layer_name = "Dropout_Layer_" + str(dropout_variable)             # Create the layer name
    return Layer_name                                                 # Return the generated name

# Define the function to generate unique names for MaxPooling layers
def Max_Pooling_Layer_Name_Generator():
    global max_pooling_variable                                       # Access the global variable 'max_pooling_variable'
    max_pooling_variable += 1                                         # Increment the variable by 1
    Layer_name = "Max_Pooling_Layer_" + str(max_pooling_variable)     # Create the layer name
    return Layer_name                                                 # Return the generated name

# Set the path for the training dataset
train_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'

# Set the path for the testing dataset
test_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/test'

# Set the path for the validation dataset
valid_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/val'

# Set the batch size for loading images
BATCH_SIZE = 10

# Create an ImageDataGenerator instance for data augmentation with the following parameters:
# - preprocessing_function: Use the VGG16 model's preprocess_input function
# - rescale: Scale the pixel values by 1/255.
# - rotation_range: Rotate the images randomly within a range of 20 degrees
# - width_shift_range: Shift the images horizontally by 20% of the total width
# - height_shift_range: Shift the images vertically by 20% of the total height
# - horizontal_flip: Enable random horizontal flipping of the images
# - zoom_range: Apply random zooming within a range of 20%
train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
) 
# Load the images from the train_path directory with the following specifications:
# - target_size: Resize the images to 224 x 224 pixels
# - classes: Set the class labels as 'NORMAL' and 'PNEUMONIA'
# - batch_size: Use the defined BATCH_SIZE for loading images
# - class_mode: Set the class mode as 'binary' for binary classification
train_batches = train_batches.flow_from_directory(
    directory=train_path,
    target_size=(224, 224),
    classes=['NORMAL', 'PNEUMONIA'],
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle = True
)

# Create an ImageDataGenerator instance for the test dataset with the specified preprocessing function and rescaling factor
test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input # Use the VGG16 model's preprocessing function
).flow_from_directory(
    directory=test_path,                                              # Set the directory path for the test dataset
    target_size=(224, 224),                                           # Resize the input images to 224x224 pixels
    classes=['NORMAL', 'PNEUMONIA'],                                  # Define the two classes for classification: NORMAL and PNEUMONIA
    batch_size=BATCH_SIZE,                                            # Set the batch size for processing images
    class_mode='binary',                                              # Set the class mode for binary classification
    shuffle=False                                                     # Do not shuffle the images in the test dataset
)

valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(
    directory=valid_path,
    target_size=(224, 224),
    classes=['NORMAL', 'PNEUMONIA'],
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
# Define the function to create the Ultrasonic model
def Ultrasonic_Model_Wrapper_Function():
    
    UltraSonic_Model = Sequential()

    # Add a 2D convolutional layer with 8 filters, 3x3 kernel size, L2 regularization, and input shape of (224, 224, 3)
    UltraSonic_Model.add(Conv2D(8, (3, 3), kernel_regularizer=l2(0.001), input_shape=(224, 224, 3), name=Convolutional_Layer_Name_Generator()))
    # Add a ReLU activation layer
    UltraSonic_Model.add(Activation('relu', name=Activation_Layer_Name_Generator()))
    # Add a layer normalization layer, normalizing along axis 1
    UltraSonic_Model.add(LayerNormalization(axis=1))
    # Add a max pooling layer with pool size of 2x2
    UltraSonic_Model.add(MaxPooling2D(pool_size=(2, 2), name=Max_Pooling_Layer_Name_Generator()))

    # Add a 2D convolutional layer with 16 filters, 3x3 kernel size, and L2 regularization
    UltraSonic_Model.add(Conv2D(16, (3, 3), kernel_regularizer=l2(0.001), name=Convolutional_Layer_Name_Generator()))
    # Add a ReLU activation layer
    UltraSonic_Model.add(Activation('relu', name=Activation_Layer_Name_Generator()))
    # Add a layer normalization layer, normalizing along axis 1
    UltraSonic_Model.add(LayerNormalization(axis=1))
    # Add a max pooling layer with pool size of 2x2
    UltraSonic_Model.add(MaxPooling2D(pool_size=(2, 2), name=Max_Pooling_Layer_Name_Generator()))

    # Add a 2D convolutional layer with 32 filters, 3x3 kernel size, and L2 regularization
    UltraSonic_Model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(0.001), name=Convolutional_Layer_Name_Generator()))
    # Add a ReLU activation layer
    UltraSonic_Model.add(Activation('relu', name=Activation_Layer_Name_Generator()))
    # Add a layer normalization layer, normalizing along axis 1
    UltraSonic_Model.add(LayerNormalization(axis=1))
    # Add a max pooling layer with pool size of 2x2
    UltraSonic_Model.add(MaxPooling2D(pool_size=(2, 2), name=Max_Pooling_Layer_Name_Generator()))

    # Add a 2D convolutional layer with 64 filters, 3x3 kernel size, and L2 regularization
    UltraSonic_Model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(0.001), name=Convolutional_Layer_Name_Generator()))
    # Add a ReLU activation layer
    UltraSonic_Model.add(Activation('relu', name=Activation_Layer_Name_Generator()))
    # Add a layer normalization layer, normalizing along axis 1
    UltraSonic_Model.add(LayerNormalization(axis=1))
    # Add a max pooling layer with pool size of 2x2
    UltraSonic_Model.add(MaxPooling2D(pool_size=(2, 2), name=Max_Pooling_Layer_Name_Generator()))

    # Add a 2D convolutional layer with 128 filters, 3x3 kernel size, and L2 regularization
    UltraSonic_Model.add(Conv2D(128, (3, 3), kernel_regularizer=l2(0.001), name=Convolutional_Layer_Name_Generator()))
    # Add a ReLU activation layer
    UltraSonic_Model.add(Activation('relu', name=Activation_Layer_Name_Generator()))
    # Add a layer normalization layer, normalizing along axis 1
    UltraSonic_Model.add(LayerNormalization(axis=1))
    # Add a max pooling layer with pool size of 2x2
    UltraSonic_Model.add(MaxPooling2D(pool_size=(2, 2), name=Max_Pooling_Layer_Name_Generator()))

    # Add a dropout layer with dropout rate of 0.2
    UltraSonic_Model.add(Dropout(0.2, name=Dropout_Layer_Name_Generator()))
    # Add a global average pooling layer
    UltraSonic_Model.add(GlobalAveragePooling2D())

    # Add a dense layer with 128 units
    UltraSonic_Model.add(Dense(128, name=Dense_Layer_Name_Generator()))
    # Add a ReLU activation layer
    UltraSonic_Model.add(Activation('relu', name=Activation_Layer_Name_Generator()))
    # Add a layer normalization layer, normalizing along axis 1
    UltraSonic_Model.add(LayerNormalization(axis=1))
    # Add a dropout layer with dropout rate of 0.2
    UltraSonic_Model.add(Dropout(0.2, name=Dropout_Layer_Name_Generator()))

    # Add a dense layer with 64 units
    UltraSonic_Model.add(Dense(64, name=Dense_Layer_Name_Generator()))
    # Add a ReLU activation layer
    UltraSonic_Model.add(Activation('relu', name=Activation_Layer_Name_Generator()))
    # Add a layer normalization layer, normalizing along axis 1
    UltraSonic_Model.add(LayerNormalization(axis=1))
    # Add a dropout layer with dropout rate of 0.2
    UltraSonic_Model.add(Dropout(0.2, name=Dropout_Layer_Name_Generator()))

    # Add a dense layer with 1 unit
    UltraSonic_Model.add(Dense(1))
    # Add a Sigmoid activation layer
    UltraSonic_Model.add(Activation('sigmoid'))
    
    return UltraSonic_Model

# Create the model by calling the function
UltraSonic_Model = Ultrasonic_Model_Wrapper_Function()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model_checkpoint = ModelCheckpoint('Layer_Norm_6.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)

callbacks = [early_stopping, model_checkpoint, reduce_lr_on_plateau]

# Compile the model with optimizer, loss, and metrics
UltraSonic_Model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

UltraSonic_Model.summary()

history = UltraSonic_Model.fit( train_batches, validation_data=valid_batches, epochs=17, callbacks=callbacks )

test_loss, test_accuracy = UltraSonic_Model.evaluate(test_batches)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

predictions = UltraSonic_Model.predict(test_batches)
predicted_classes = np.round(predictions)
true_classes = test_batches.classes

confusion_mtx = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(confusion_mtx)

TP = confusion_mtx[0][0]
TN = confusion_mtx[1][1]
FP = confusion_mtx[0][1]
FN = confusion_mtx[1][0]

print("True Positive:", TP)
print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)

# Calculate the normalized confusion matrix
normalized_confusion_mtx = confusion_mtx / np.sum(confusion_mtx)

# Define the class names
class_names = ['NORMAL', 'PNEUMONIA']

# Create custom annotations for the heatmap by combining class names with the corresponding values and percentages
annotations = np.array([[f"{class_names[0]}\n{confusion_mtx[0, 0]} | {normalized_confusion_mtx[0, 0]*100:.2f}%",
                         f"{class_names[1]}\n{confusion_mtx[0, 1]} | {normalized_confusion_mtx[0, 1]*100:.2f}%"],
                        [f"{class_names[0]}\n{confusion_mtx[1, 0]} | {normalized_confusion_mtx[1, 0]*100:.2f}%",
                         f"{class_names[1]}\n{confusion_mtx[1, 1]} | {normalized_confusion_mtx[1, 1]*100:.2f}%"]])

# Plot the heatmap with custom annotations and class names on the x-axis and y-axis
fig, ax = plt.subplots()
sns.heatmap(confusion_mtx, annot=annotations, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

# Set the labels for the x-axis and y-axis
ax.set_xlabel('Predicted Classes')
ax.set_ylabel('True Classes')

# Calculate the number of samples in each class
num_samples_normal = np.sum(confusion_mtx[:, 0])
num_samples_pneumonia = np.sum(confusion_mtx[:, 1])

# Save the plot as a JPEG file
fig.savefig("confusion_matrix.jpg", dpi=300)

# Show the plot
plt.show()

UltraSonic_Model.save('UltraSonic')
