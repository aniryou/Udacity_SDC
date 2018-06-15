# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/visualization_distr.png "Composition"
[image3]: ./examples/visualization_norm.png "Visualization Normalized"
[image4]: ./examples/original.png "Original"
[image5]: ./examples/transformed.png "Transformed"
[image6]: ./examples/visualization_distr_aug.png "Dataset composition after augmentation"
[image7]: ./examples/1.jpg "Traffic Sign 1"
[image8]: ./examples/2.jpg "Traffic Sign 2"
[image9]: ./examples/3.jpg "Traffic Sign 3"
[image10]: ./examples/4.jpg "Traffic Sign 4"
[image11]: ./examples/5.jpg "Traffic Sign 5"
[image12]: ./examples/6.jpg "Traffic Sign 6"
[image13]: ./examples/1_pred.png "Traffic Sign 1 Prediction"
[image14]: ./examples/2_pred.png "Traffic Sign 2 Prediction"
[image15]: ./examples/3_pred.png "Traffic Sign 3 Prediction"
[image16]: ./examples/4_pred.png "Traffic Sign 4 Prediction"
[image17]: ./examples/5_pred.png "Traffic Sign 5 Prediction"
[image18]: ./examples/6_pred.png "Traffic Sign 6 Prediction"
[image19]: ./examples/feature_map_input.png "Feature Map Input"
[image20]: ./examples/feature_map_conv1.png "Feature Map Convolution Layer 1"
[image21]: ./examples/feature_map_conv2.png "Feature Map Convolution Layer 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `32 x 32 x 3`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It captures a random image for each of the `43` classes in the dataset.

![Dataset Visualization][image1]

Below is Composition of training data in terms of number of images for each of the `43` classes.

![Dataset Composition][image2]

Clearly the dataset is imbalanced and this might pose a problem for some of the classes. As we will see later, data augmentation helps here.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique.

Looking at the visualization in first image above, some of the images are very poorly lit.
We can improve the luminosity using [CLAHE](http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist).

![Dataset Visualization Normalized][image3]

Further, as mentioned earlier, some of the classes have very few samples. Therefore, I used data augmentation.
The augmented images are randomized [affine transformations](http://www.scipy-lectures.org/packages/scikit-image/auto_examples/plot_features.html) (translation, rotation, scaling, shear) of randomly sampled images for a class.
I used below limits for sampling parameters of transformation, from uniform distribution.

 - Translation `[-2, 2]` pixels
 - Rotation `[-15, 15)` degrees
 - Scaling `[0.9, 1.1)`
 - Shear `[-0.2, 0.2)`

Below is an example of original to transformed image with `translation = (1, -1)`, `scaling = 1.1`, `rotation = -3`, `shear = 0.2`

![Original][image4] ![Transformed][image5]

The difference between the original data set and the augmented data set is the following, contrast this with the training set distribution earlier - all classes are sufficiently represented.

![Dataset composition after augmentation][image6]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:----------------------|:----------------------------------------------|
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Batch Normalization   | default momentum of 0.999 	                |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| Batch Normalization   | default momentum of 0.999 	                |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| Flatten		        | output size 400								|
| Fully connected		| output size 120        						|
| Batch Normalization   | default momentum of 0.999 	                |
| RELU					|												|
| Dropout				| keep probability 50%       					|
| Fully connected		| output size 84        						|
| Batch Normalization   | default momentum of 0.999 	                |
| RELU					|												|
| Dropout				| keep probability 50%       					|
| Fully connected		| output size 10        						|
| Softmax				|             									|
 
This is standard LeNet model, to which I added Batch Normalization and Dropout layers.

#### 3. Describe how you trained your model.

To train the model, I used AdamOptimizer. This combines Gradient Descent with momemtum and RMSProp and dampens oscillations - allowing us to use a high learning rate.
I used a learning rate of 0.01 which seems to work well, but can be further fine-tuned.
I used a batch size of 128, which is standard - increasing the batch size might be useful with Batch Normalization - but it comes at computational cost. It can also be further fine-tuned.
I used 50 epochs. In some of the courses I have attended, early stopping is discouraged as it violates orthogonality principle for hyper-parameter tuning. After 50 epochs, the validation accuracy seems to settle down around 94%, which is above the requirement of 93%.
The validation set seems easier for the model, as its accuracy is often higher than training accuracy. This could be accidental, or due to the fact that we used augmentation by random affine transformations in the training set - which makes it slightly hard.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 96.88
* validation set accuracy of 94.65
* test set accuracy of 92.40

I chose LeNet architecture due to ease of training on my laptop.

The original architecture achieved ~90% training accuracy, which improved significantly after introducing batch-normalization.
I tried few values of momentum for batch normalization, which didn't have a clear effect.
I also played with batch size, and a bigger batch size seemed to work better - perhaps due to better estimates for batch normalization.
However, eventually, I settled for standard batch-size and use more epochs instead.
Increasing the number of epochs and decreasing the learning rate have a positive influence - but it takes longer.
Therefore, I kept these to standard values manageable on my laptop.

I first assessed that the model is able to completely crack the training dataset, with 100% accuracy.
At this point, we don't care ourselves with overfitting but focus on model complexity.
Once it's established that we're able to achieve reasonably good accuracy on the training data, we need not look for a more complex architecture and instead can focus on reducing over-fitting.
For overfitting, dropout works extremely well. I also added a minor L2 regularization term to the loss, but didn't fine-tune it much since overfitting wasn't a problem.

After evaluating the model accuracy on validation and iterating on the model, I evaluated the accuracy on test set - which turned out to be above 90%.
Since the model hasn't seen the test set - we can be confident of generalizability.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report

Here are five German traffic signs that I found on the web:

![Traffic Sign 1][image7] ![Traffic Sign 2][image8] ![Traffic Sign 3][image9] ![Traffic Sign 4][image10]
![Traffic Sign 5][image11] ![Traffic Sign 6][image12]

One might expect that the third image is difficult to classify due to sign orientation, however model seems to work perfectly for all six.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set

Here are the results of the prediction:

| Image			                    |     Prediction	        					|
|:---------------------------------:|:---------------------------------------------:|
| Turn left ahead      	            | Turn left ahead   							|
| Yield					            | Yield											|
| Stop					            | Stop											|
| Right-of-way at next intersection | Right-of-way at next intersection             |
| Road work                         | Road work										|
| Speed limit (30km/h)              | Speed limit (30km/h)   						|


The model was able to correctly guess 6 of the ^ traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction

Below are the predictions alongside the softmax probabilities for each. First image has an observable non-zero probability for 'Ahead only' class, which isn't very surprising due to some overlap of the signs.

![Traffic Sign 1 Prediction][image13]

![Traffic Sign 2 Predictio][image14]

![Traffic Sign 3 Prediction][image15]

![Traffic Sign 4 Prediction][image16]

![Traffic Sign 5 Prediction][image17]

![Traffic Sign 6 Prediction][image18]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is a visualization of the featuremap for the two convolution layers for first image.

![Input Image][image19]

The features 1, 4, 5 in layer 1 behave like filters whereas 2, 3, and 6 behave as edge detectors.

![Convolution Layer 1][image20]

The features in layer 2 are difficult to interpret with a single image - we can look at same for a collection of images to understand activations better.

![Convolution Layer 2][image21]
