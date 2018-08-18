# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_architecture.png "Model Visualization"
[image2]: ./examples/center_2016_12_01_13_44_36_155.jpg "Center Camera"
[image3]: ./examples/left_2016_12_01_13_44_36_155.jpg "Left Camera"
[image4]: ./examples/right_2016_12_01_13_44_36_155.jpg "Right Camera"
[image5]: ./examples/right_2016_12_01_13_44_36_155.jpg "Sharp Curve"
[image6]: ./examples/center_2018_07_02_09_24_35_895_flipped.jpg "Sharp Curve Flipped"
[image7]: ./examples/center_2018_07_02_09_24_35_895_gray.jpg "Grayscaled Image"
[image8]: ./examples/center_2018_07_02_09_24_35_895_cropped.jpg "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model3.h5 containing a trained convolution neural network 
* README.md (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with LeNet architecture, with batch-normalization and dropout (model.py lines 133-159) 

The model includes RELU layers to introduce nonlinearity, and the data is grayscaled and normalized 
in the model using a Keras lambda layer (model.py lines 134, 135).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 153, 157). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 56-57, 164). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 159).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use architectures that are simple and have worked well in past.
Therefore, continuing from Traffic Sign classifier, I used LeNet architecture. 

In order to gauge how well the model was working, I split my image and steering angle data into a training, validation and test set.
Even though the model seemed to do well on all 3 sets, vehicle fell of track around curves.
```sh
python model.py model1.h5 data --all_cameras --nb_epoch 50
```

The model had a tendency to drive straight and steering was delayed to a point from where recovery was hard.
Therefore, I retrained a second model, on top of first model - this time ignoring all images with zero steer (model.py lines 69-72, 189)
and side cameras (model.py line 188, set to false)
```sh
python model.py model2.h5 data --ignore_no_steer --nb_epoch 10 --weights_path model1.h5
```

This considerably improved performance (car was able to drive longer), but vehicle swayed even on straight road and sometimes went off track on sharp curves.
Therefore, I gathered some more training data manually and retrained a third model on top of second.
The third model included all center images.
```sh
python model.py model3.h5 data --nb_epoch 20 --weights_path model2.h5
```

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 133-159) consisted of a convolution neural network with the following layers and layer sizes:

| Layer (type)                     | Output Shape          | Param #     | Connected to               |      
|----------------------------------|-----------------------|-------------|----------------------------|
| lambda_1 (Lambda)                | (None, 160, 320, 1)   | 0           | lambda_input_1[0][0]       |      
|                                  |                       |             |                            |
| lambda_2 (Lambda)                | (None, 160, 320, 1)   | 0           | lambda_1[0][0]             |      
|                                  |                       |             |                            |
| cropping2d_1 (Cropping2D)        | (None, 60, 320, 1)    | 0           | lambda_2[0][0]             |      
|                                  |                       |             |                            |
| convolution2d_1 (Convolution2D)  | (None, 56, 316, 6)    | 156         | cropping2d_1[0][0]         |      
|                                  |                       |             |                            |
| batchnormalization_1 (BatchNorma | (None, 56, 316, 6)    | 24          | convolution2d_1[0][0]      |      
|                                  |                       |             |                            |
| activation_1 (Activation)        | (None, 56, 316, 6)    | 0           | batchnormalization_1[0][0] |      
|                                  |                       |             |                            |
| maxpooling2d_1 (MaxPooling2D)    | (None, 28, 158, 6)    | 0           | activation_1[0][0]         |      
|                                  |                       |             |                            |
| convolution2d_2 (Convolution2D)  | (None, 24, 154, 10)   | 1510        | maxpooling2d_1[0][0]       |      
|                                  |                       |             |                            |
| batchnormalization_2 (BatchNorma | (None, 24, 154, 10)   | 40          | convolution2d_2[0][0]      |      
|                                  |                       |             |                            |
| activation_2 (Activation)        | (None, 24, 154, 10)   | 0           | batchnormalization_2[0][0] |      
|                                  |                       |             |                            |
| maxpooling2d_2 (MaxPooling2D)    | (None, 12, 77, 10)    | 0           | activation_2[0][0]         |      
|                                  |                       |             |                            |
| convolution2d_3 (Convolution2D)  | (None, 8, 73, 16)     | 4016        | maxpooling2d_2[0][0]       |      
|                                  |                       |             |                            |
| batchnormalization_3 (BatchNorma | (None, 8, 73, 16)     | 64          | convolution2d_3[0][0]      |      
|                                  |                       |             |                            |
| activation_3 (Activation)        | (None, 8, 73, 16)     | 0           | batchnormalization_3[0][0] |      
|                                  |                       |             |                            |
| maxpooling2d_3 (MaxPooling2D)    | (None, 4, 36, 16)     | 0           | activation_3[0][0]         |      
|                                  |                       |             |                            |
| flatten_1 (Flatten)              | (None, 2304)          | 0           | maxpooling2d_3[0][0]       |      
|                                  |                       |             |                            |
| dense_1 (Dense)                  | (None, 100)           | 230500      | flatten_1[0][0]            |      
|                                  |                       |             |                            |
| batchnormalization_4 (BatchNorma | (None, 100)           | 400         | dense_1[0][0]              |      
|                                  |                       |             |                            |
| activation_4 (Activation)        | (None, 100)           | 0           | batchnormalization_4[0][0] |      
|                                  |                       |             |                            |
| dropout_1 (Dropout)              | (None, 100)           | 0           | activation_4[0][0]         |     
|                                  |                       |             |                            |
| dense_2 (Dense)                  | (None, 84)            | 8484        | dropout_1[0][0]            |      
|                                  |                       |             |                            |
| batchnormalization_5 (BatchNorma | (None, 84)            | 336         | dense_2[0][0]              |     
|                                  |                       |             |                            |
| activation_5 (Activation)        | (None, 84)            | 0           | batchnormalization_5[0][0] |     
|                                  |                       |             |                            |
| dropout_2 (Dropout)              | (None, 84)            | 0           | activation_5[0][0]         |     
|                                  |                       |             |                            |
| dense_3 (Dense)                  | (None, 1)             | 85          | dropout_2[0][0]            |


| Parameters                |
|---------------------------|
| Total params: 245,615     |
| Trainable params: 245,183 |
| Non trainable params: 432 |


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


![alt text][image1]


#### 3. Creation of the Training Set & Training Process

For model1 (initial model), I used the data-set provided in assignment resources.
This comprises of images from center camera as well as left and right side-cameras.

The center camera images provided training data for center lane driving. Below is an image from center camera:


![alt text][image2]


For vehicle to recover from left and right sides of the road back to center, images from side-cameras were useful.
Below are images from left and right camera respectively. As can be seen, these are good proxy for recovery.


![alt text][image3]


![alt text][image4]


I gathered some data manually, by driving the car for few laps. This allowed getting more data around sharp curves.
Below is an example image:


![alt text][image5]


To augment the data sat, I also flipped images randomly (model.py lines 101, 107-109). 
For example, here is an original image, and corresponding flipped one:


![alt text][image5]


![alt text][image6]


After the collection process, I had 27242 number of data points. 
The pre-processing steps are included as part of model pipeline. 
These include converting the image to gray-scale and normalizing the image to center values.
Below are examples of gray-scaled and image cropped image.


![alt text][image7]


![alt text][image8]


Finally, I cropped the image to keep view of road immediately ahead of the car (top-cropping) and remove car panel (bottom-cropping).
Later is required so that images from side cameras are treated exactly identical as those from center camera.
Below id a sample cropped image.


The data was split in 80-20% into train and validation set. 

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting.
The batch size was set to standard 32 images. 
I did not use early stopping for first model and let the model train for long enough till the values stabilize.
For models 2 and 3, early stopping was used. This enables keeping effect of targeted fine-tuning in control.  
I used an adam optimizer so that manually training the learning rate was not necessary.
