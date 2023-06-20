
# DataWhisper: Mastering DL Project Lifecycle

Welcome to the Deep Learning Project Starter Guide! This tutorial serves as a comprehensive resource for anyone looking to dive into the exciting world of deep learning. Whether you're a beginner or an experienced developer, this guide will take you through the process of building a deep learning project from start to finish.

## What you'll learn

In this tutorial, you will learn the essential steps involved in creating and deploying a deep learning model in a mobile app. We will cover the following topics:

1. Preparing the data: We'll explore various methods for data preprocessing to ensure a robust and reliable dataset for training.

2. Model creation: You'll discover how to design and build your CNN model.

3. Training the model: We'll delve into the process of training your deep learning model using TensorFlow.

4. Deployment in a mobile app: Once your model is trained, we'll guide you through the steps to integrate it into a mobile app using TensorFlow Lite. You'll understand how to make predictions on the go!

## Who should follow this tutorial

This tutorial is suitable for both beginners and intermediate developers who have a basic understanding of deep learning concepts and Python programming. Whether you're a data scientist, machine learning enthusiast, or mobile app developer, this guide will equip you with the necessary knowledge to kick-start your deep learning project.


## Need help or have questions?

If you encounter any issues, have questions, or need further clarification while following this tutorial, don't hesitate to create a GitHub issue in this repository. I'll be more than happy to assist you and provide the necessary guidance.

To create an issue, click on the "Issues" tab at the top of this repository's page and then click the "New issue" button. Please provide as much context and detail as possible about the problem you're facing or the question you have. This will help me understand your concern better and provide you with a prompt and accurate response.

Your feedback is valuable and can help improve this tutorial for other users as well. So, don't hesitate to reach out if you need any assistance. Let's learn and grow together!

## Let's get started!

To get started, make sure you have the required dependencies and libraries installed. The tutorial is divided into easy-to-follow sections, each covering a specific aspect of the deep learning project workflow. Feel free to jump to the sections that interest you the most or follow along from beginning to end.

Are you ready?

## Imports and loading the dataset

Let's start the necessary `import`s for our code. We will use the Fashion Mnist dataset in this tutorial.

```python
# Import the necessary libraries
from __future__ import print_function
import keras
from google.colab import drive
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
```

## Dataset Structure

In any deep learning project, understanding the data is crucial. Before diving into model creation and training, let's start by loading the data and gaining insights into its structure, variables, and overall characteristics.

```python
# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```
## Exploratory Data Analysis (EDA)
Now that the data is loaded, let's perform some exploratory data analysis to gain a better understanding of its characteristics. 
```python
print("Shape of the training data : ",x_train.shape)
print("Shape of the testing data : ",x_test.shape)
```
```output
Shape of the training data :  (60000, 28, 28)
Shape of the testing data :  (10000, 28, 28)
```
The Fashion MNIST dataset contains **70,000** grayscale images in 10 categories. The images show individual articles of clothing at low resolution **(28 by 28 pixels)**, as seen here:
![image](https://tensorflow.org/images/fashion-mnist-sprite.png)

**60,000** images are used to train the network and **10,000** images to evaluate how accurately the network learned to classify images.
```python
# Printing unique values in training data
unique_labels = np.unique(y_train, axis=0)
print("Unique labels in training data:", unique_labels)
```
```output
Unique labels in training data: [0 1 2 3 4 5 6 7 8 9]
```
The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
| Label  | RClass  |
| - |-|
| 0     | T-shirt/top|
| 1     | Trouser|
| 2     |Pullover|
| 3     |Dress|
| 4     |Coat|
| 5     |Sandal|
| 6     |Shirt|
| 7     |Sneaker |
| 8     |Bag|
| 9     | 	Ankle boot |

Since the class names are not included with the dataset, store them here to use later when plotting the images:

```python
# Numeric labels
numeric_labels = np.sort(np.unique(y_train, axis=0))
# String labels
string_labels = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
# Mapping numeric labels to string labels
numeric_to_string = dict(zip(numeric_labels, string_labels))
print("Numeric to String Label Mapping:")
print(numeric_to_string)
```
```output
Numeric to String Label Mapping:
{0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
```
## Preprocess the data

The data must be preprocessed before training the network.
We start by defining the number of classes in our dataset (which is 10 in this case) and the dimensions of the input images (28x28 pixels).
```python
num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28
```
This part is responsible for reshaping the input image data to match the expected format for the neural network model. The format depends on the backend being used (e.g., TensorFlow or Theano). In this snippet, we check the image data format using **K.image_data_format()** and apply the appropriate reshaping based on the result.

```python
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
```
The pixel values of the images in the data fall within the range of 0 to 255.
Scale these values to a range of 0 to 1 before feeding them to the CNN model.
```python
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
```
Convert the class labels (represented as integers) to a binary class matrix format, which is required for multi-class classification problems.
```python
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```
## Build the model
In this step, we define and build a convolutional neural network (CNN) model for image classification. The model architecture consists of multiple layers such as convolutional, pooling, dropout, and dense layers. The build_model function takes the number of classes, training and testing data as input and returns the training history and the built model.
```python
def build_model(num_classes,x_train,y_train,x_test,y_test):
  model = Sequential()
  model.add(BatchNormalization())
  model.add(Conv2D(64, kernel_size=(4, 4), padding='same',
                  activation='relu',
                  input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.1))
  model.add(Conv2D(64, kernel_size=(4, 4),
                  activation='relu',
                  input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=tf.optimizers.Adam(),
                metrics=['accuracy'])
  history = model.fit(x_train, y_train,epochs=80,batch_size=250,validation_data=(x_test, y_test))
  return history,model

history,model = build_model(num_classes,x_train,y_train,x_test,y_test)
```
```output
Epoch 80/80
240/240 [==============================] - 211s 878ms/step - loss: 0.0866 - accuracy: 0.9678 - val_loss: 0.2308 - val_accuracy: 0.9289
```
```python
# Function: is_drive_mounted
# Description: Checks if the Google Drive is mounted.
# Returns: True if the drive is mounted, False otherwise.
def is_drive_mounted():
    return os.path.isdir('/content/drive')
# Function: mount_drive
# Description: Mounts the Google Drive if it is not already mounted.
# Prints a message if the drive is already mounted.
def mount_drive():
  if not is_drive_mounted():
    drive.mount('/content/drive')
  else:
      print('Drive already mounted.')
# Function: move_to_drive
# Description: Moves a file to the specified folder in Google Drive.
# If the file already exists in the destination folder, prints a message.
# Parameters:
#   - filename: Name of the file to be moved.
#   - folder: Name of the destination folder.
# Returns: The path of the moved file in Google Drive.
def move_to_drive(filename, folder):
  mount_drive()
  # Define the file path and name in your Google Drive
  output_file_path = '/content/drive/MyDrive/'+folder
  if os.path.exists(output_file_path+filename):
    print("The file exists.")
  else:
    # Copy the downloaded file to your Google Drive
    !cp /content/'{filename}' '{output_file_path}'
  return output_file_path+filename
# Function: save_class_names
# Description: Saves a list of class names to a file.
# The file is created in the specified folder.
# Parameters:
#   - folder_path: Path of the destination folder.
#   - filename: Name of the output file.
# Returns: The path of the saved file in Google Drive and the list of class names.
def save_class_names(folder_path, filename):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  with open(filename, 'w') as f:
    for class_name in class_names:
        f.write(class_name + '\n')
  return move_to_drive(filename, folder_path),class_names
# Function: h52tflite
# Description: Converts a Keras model to TFLite format and saves the model.
# Parameters:
#   - model: Keras model to be converted.
#   - filename: Name of the output file.
# Returns: The path of the saved TFLite model in Google Drive and the TFLite model itself.
def h52tflite(model,filename):
  # Convert the model to TFLite format
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the TFLite model
  with open(filename, 'wb') as f:
      f.write(tflite_model)

  return move_to_drive(filename,'DLTutorial/model/'),tflite_model
# Function: plot_model_evaluation
# Description: Plots the training and validation loss and accuracy of a model.
# Parameters:
#   - model: Model to be evaluated.
#   - history: Training history of the model.
def plot_model_evaluation(model,history):
  plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

  ax1.plot(history.history['loss'], label = 'training loss')
  ax1.plot(history.history['accuracy'], label = 'training accuracy')
  ax1.legend()
  ax1.set_title('Training')

  ax2.plot(history.history['val_loss'], label = 'validation loss')
  ax2.plot(history.history['val_accuracy'], label = 'validation accuracy')
  ax2.legend()
  ax2.set_title('Validation')

  plt.show()
# Function: export_model
# Description: Saves a Keras model as an .h5 file, converts it to TFLite format, and saves the TFLite model.
# Parameters:
#   - model: Keras model to be exported.
#   - filename: Name of the output files.
# Returns: The paths of the saved model and TFLite model in Google Drive, and the TFLite model itself.
def export_model(model,filename):
  model.save(filename+'.h5')
  model_path = move_to_drive(filename+'.h5', 'DLTutorial/model/')
  tflite_model_path,tflite_model = h52tflite(model,filename+".tflite")
  return model_path,tflite_model_path,tflite_model
```
## Evaluate accuracy
To assess the performance of the trained model, we evaluate it on the test data. The evaluate method is used to calculate the test loss and accuracy. These metrics are then printed to the console.
```python
plot_model_evaluation(model,history)
```
![image](https://github.com/ABDELLAH-Hallou/From-Data-to-Deployment/blob/master/assets/evaluate.png)
```python
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```
```output
Test Loss: 0.2308
Test Accuracy: 0.9289
```
## Save and Export the Model
After training the model, we save it in the Hierarchical Data Format (HDF5) file format using the **save** method. The model is then exported to the Google Drive by calling the **move_to_drive** function. Additionally, the model is converted to the TensorFlow Lite format using the **h52tflite** function, and the resulting TFLite model is also saved in the Google Drive. The paths of the saved model and TFLite model are returned.
```python
model_path,tflite_model_path,tflite_model = export_model(model,"model")
```
## Make predictions
To visualize the model's predictions, we select a random set of test images. The model predicts the class labels for these images using the predict method. The predicted labels are then compared with the ground truth labels to display the images along with their corresponding predicted labels using **matplotlib**.
```python
# Predict class labels for test images
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Visualize a random selection of test images along with their predicted labels
num_rows = 5
num_cols = 5
num_images = num_rows * num_cols
random_indices = np.random.choice(len(x_test), size=num_images, replace=False)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
fig.suptitle("Fashion MNIST - Predictions", fontsize=16)
fig.tight_layout(pad=2.0)

for i, index in enumerate(random_indices):
    ax = axes[i // num_cols, i % num_cols]
    ax.imshow(x_test[index], cmap='gray')
    ax.set_title(f"Pred: {string_labels[predicted_labels[index]]}")
    ax.axis('off')

plt.show()
```
![image](https://github.com/ABDELLAH-Hallou/From-Data-to-Deployment/blob/master/assets/pred.png)

for more information about the model, check these ressources : 
1. https://www.tensorflow.org/tutorials/keras/classification
2. https://github.com/cmasch/zalando-fashion-mnist/tree/master

## Deployment
### Create a new flutter project
Before creating new flutter project, make sure that the Flutter SDK and other Flutter app development-related requirements are properly installed: https://docs.flutter.dev/get-started/install/windows

After the project has been set up, we will implement the UI to allow users to take pictures or upload images from the gallery and perform object recognition using a the exported TensorFlow Lite model.
First, we need to install these packages:
1. camera: **0.10.4**
2. image_picker:
3. tflite: **^1.1.2** 
to do so copy the following code snippet and paste it into the **pubspec.yaml** file of the project:
```yaml
dependencies:
  camera: 0.10.4
  image_picker:
  tflite: ^1.1.2
```
Import the necessary packages in the **main.dart** file of the project
```dart
import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
```
#### Configuring the Camera
To enable camera functionality, we'll utilize the **camera** package. First, import the necessary packages and instantiate the camera controller. Use the **availableCameras()** function to get a list of available cameras. In this tutorial, we'll use the first camera in the list.
```dart
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.first;
  runApp(MyApp(firstCamera));
}
```
#### Creating the Camera Screen
Create a new StatefulWidget called **CameraScreen** that will handle the camera preview and image capture functionality. In the **initState()** method, initialize the camera controller and set the resolution preset. Additionally, implement the **_takePicture()** method, which captures an image using the camera controller.
```dart
// main.dart
class CameraScreen extends StatefulWidget {
  final CameraDescription camera;

  const CameraScreen(this.camera);

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture = Future.value();

  @override
  void initState() {
    super.initState();
    _initCameraController();
  }

  Future<void> _initCameraController() async {
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
    );

    _initializeControllerFuture = _controller.initialize();

    setState(() {});
  }
    Future<void> _takePicture() async {
    try {
      await _initializeControllerFuture;
      final image = await _controller.takePicture();
      await _processImage(image.path);
    } catch (e) {
      print('Error capturing image: $e');
    }
  }
  // ...
}

```
#### Integrating Image Upload
To allow users to upload images from the gallery, import the &**image_picker** package. Implement the **_pickImage()** method, which utilizes the **ImagePicker** class to select an image from the gallery. Once an image is selected, it can be processed using the **_processImage()** method.
```dart
// main.dart
class _CameraScreenState extends State<CameraScreen> {
  // ...

  Future<void> _pickImage() async {
    try {
      final imagePicker = ImagePicker();
      final pickedImage = await imagePicker.getImage(source: ImageSource.gallery);

      if (pickedImage != null) {
        await _processImage(pickedImage.path);
      }
    } catch (e) {
      print('Error picking image: $e');
    }
  }

  // ...
}
```
#### Object Recognition with TensorFlow Lite
To perform object recognition, we'll use the TensorFlow Lite framework. Begin by importing the **tflite** package. In the **_initTensorFlow()** method, load the TensorFlow Lite model and labels from the assets. You can specify the model and label file paths and adjust settings like the number of threads and GPU delegate usage.
```dart
// main.dart
class _CameraScreenState extends State<CameraScreen> {
  // ...

  @override
  void initState() {
    super.initState();
    _initCameraController();
    _initTensorFlow();
  }

  Future<void> _initTensorFlow() async {
    String? res = await Tflite.loadModel(
      model: "assets/model/model.tflite",
      labels: "assets/model/labels.txt",
      numThreads: 1,
      isAsset: true,
      useGpuDelegate: false,
    );
  }

  // ...
}

```
#### Running the Model on Images
Implement the **_objectRecognition()** method, which takes an image file path as input and runs the TensorFlow Lite model on the image. The method returns the label of the recognized object.
```dart
// main.dart
class _CameraScreenState extends State<CameraScreen> {
  // ...
  Future<String> _objectRecognition(String filepath) async {
    var recognitions = await Tflite.runModelOnImage(
      path: filepath,
      numResults: 10,
      threshold: 0.1,
      asynch: true,
    );

    return recognitions![0]["label"].toString();
  }

  // ...
}
```
#### Displaying Results in a Dialog
When an image is processed, display the result in a dialog box using the **showDialog()** method. Customize the dialog to show the recognized object label and provide an option to cancel.
```dart
// main.dart
class _CameraScreenState extends State<CameraScreen> {
  // ...
Future<void> _processImage(String imagePath) async {
    final String label = await _objectRecongnition(imagePath);
    showDialog(
      context: context,
      barrierDismissible: false, dialog
      builder: (BuildContext context) {
        return Dialog(
          child: Container(
            padding: EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(label),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.of(context).pop(); // Close the dialog
                  },
                  child: Text('Cancel'),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
  // ...
}

```
#### Building the User Interface
```dart
void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final cameras = await availableCameras();
  final firstCamera = cameras.first;


  runApp(MyApp(firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp(this.camera);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fashion Mnist',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: CameraScreen(camera),
    );
  }
}

class CameraScreen extends StatefulWidget {
  CameraScreen(CameraDescription camera);

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture = Future.value();

  @override
  void initState() {
    super.initState();
    _initCameraController();
    _initTensorFlow();
  }
  Future<void> _initCameraController() async {
    final cameras = await availableCameras();
    final firstCamera = cameras.first;
    _controller = CameraController(
      firstCamera,
      ResolutionPreset.medium,
    );
    _initializeControllerFuture = _controller.initialize();
    setState(() {});
  }
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _takePicture() async {
    try {
      await _initializeControllerFuture;
      final image = await _controller.takePicture();
      await _processImage(image.path);
    } catch (e) {
      print('Error capturing image: $e');
    }
  }

  Future<void> _pickImage() async {
    try {
      final imagePicker = ImagePicker();
      final pickedImage = await imagePicker.getImage(
          source: ImageSource.gallery);
      if (pickedImage != null) {
        await _processImage(pickedImage.path);
      }
    }catch(e){
      print('Error picking image: $e');
    }
  }

  Future<void> _processImage(String imagePath) async {
    final String label = await _objectRecongnition(imagePath);
    showDialog(
      context: context,
      barrierDismissible: false, dialog
      builder: (BuildContext context) {
        return Dialog(
          child: Container(
            padding: EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(label),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.of(context).pop(); // Close the dialog
                  },
                  child: Text('Cancel'),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Fashion Mnist'),
      ),
      body: Column(
        children: <Widget>[
          Expanded(
            child: Container(
              child: FutureBuilder<void>(
                future: _initializeControllerFuture,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.done) {
                    return CameraPreview(_controller);
                  } else {
                    return Center(child: CircularProgressIndicator());
                  }
                },
              )
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton.icon(
                onPressed: _takePicture,
                icon: Icon(Icons.camera_alt),
                label: Text('Take Picture'),
              ),
              ElevatedButton.icon(
                onPressed: _pickImage,
                icon: Icon(Icons.photo_library),
                label: Text('Upload from Gallery'),
              ),
            ],
          ),
        ],
      ),
    );
  }
  Future<void> _initTensorFlow() async{
    String? res = await Tflite.loadModel(
        model: "assets/model/model.tflite",
        labels: "assets/model/labels.txt",
        numThreads: 1,
        isAsset: true,
        useGpuDelegate: false
    );
  }
  Future<String> _objectRecongnition(String filepath) async{
    var recognitions = await Tflite.runModelOnImage(
        path: filepath,
        numResults: 10,
        threshold: 0.1,
        asynch: true
    );
    return recognitions![0]["label"].toString();
  }
}
```
![image](https://github.com/ABDELLAH-Hallou/From-Data-to-Deployment/blob/master/assets/camera.png)
![image](https://github.com/ABDELLAH-Hallou/From-Data-to-Deployment/blob/master/assets/result.png)
