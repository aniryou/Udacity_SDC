# Self Driving Car

I've enrolled for Udacity's course on [Self Driving Car](https://in.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) starting April 2018.
This repository has my submissions to SDC assignments and useful links.

## Term1

### Finding Lanes

This is first assignment for term 1 and requied detecting lanes on road using OpenCV library.
My model worked well under good lighting conditions, but fails on some parts of the optional challenge video.
Take a look at the [model](https://github.com/aniryou/Udacity_SDC/blob/master/CarND-LaneLines-P1/P1.ipynb) and [writeup](https://github.com/aniryou/Udacity_SDC/blob/master/CarND-LaneLines-P1/writeup_lane_finding.md).

### Traffic Sign Classification

This is second assignment for term 1 and required building a classifier for 
[German Traffic Sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
It also covers deep-learning topics and tensorflow library in depth.
Take a look at [LeNet Lab](https://github.com/udacity/CarND-TensorFlow-Lab) and [Tensorflow Lab](https://github.com/udacity/CarND-TensorFlow-Lab).

Using standard LeNet model with [Batch Normalization](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization) and 
[Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) I was able to get above 93% validation accuracy.
Here are links to the [model](https://github.com/aniryou/Udacity_SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb) 
and [writeup](https://github.com/aniryou/Udacity_SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup_traffic_sign_classifier.md).


### Behavior Cloning

This is third assignment for term 1. Here we create an end to end deep learning model to drive on a game simulator.
I reused the LeNet model from Traffic Sign Classifier, but changed loss function to MSE of actual vs predicted steering angles.
Take a look at the [Writeup](https://github.com/aniryou/Udacity_SDC/tree/master/CarND-Behavioral-Cloning-P3/README.md) 


### Advanced Lane Finding

In this fourth assignment for term1, we redo the lane detection from first assignment. This time we detected lanes with varying curvatures,
lighting and markings. We learn about camera calibration, perspective transformations and color channels. Using a combination
of all of these techniques allows us to detect lanes with surprisingly better accuracy compared to assignment 1.
Take a look at the [Writeup](https://github.com/aniryou/Udacity_SDC/blob/master/CarND-Advanced-Lane-Lines/README.md) 