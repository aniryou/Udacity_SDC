## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/vehicle.png
[image2]: ./examples/non-vehicle.png
[image3]: ./examples/hog_features.png
[image4]: ./examples/sliding_window.png
[image5]: ./examples/test_image.png
[image6]: ./examples/bboxes_and_heat.png
[image7]: ./examples/labels_map.png
[image8]: ./examples/output_bboxes.png
[image9]: ./examples/vehicle_bbox2.png
[image10]: ./examples/test1.jpg
[image11]: ./examples/heatmap.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.
You can submit your writeup as markdown or pdf.

[Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for 
this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cells `3` to `14` of the IPython notebook `Vehicle_Detection_Model.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each 
of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, 
and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel for what 
the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, 
`pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I used default parameter values for hog transform, except for `block_norm=L2-Hys`. 
This was decided after some literature review on hog transforms (esp. [this lecture](https://www.youtube.com/watch?v=7S5qXET179I)).
The value seemed to work reasonably well, but there is scope of fine-tuning.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Given high-dimensionality of the data, I used SVM. I tried `linear` and `rbf` kernels with grid-search over hyper-parameters.
I used the method suggested in ['A Practical Guide to Support Vector Classification'](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) 
from one of the creators of scikit-learn SVM.
The code for classifier training can be found in `Vehicle_Detection_Model.ipynb` from cells `16` to `22`.
The final accuracy of model on hold-out test set was `99.26%`, which seems satisfactory (given the data-set is balanced).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for sliding window search can be found on `Vehicle_Detection-Prediction.ipynb` cell `4` `predict_heatmap()` function.
We restrict vehicle search window in horizontal slice of the image between pixels `400` to `680` (this will of-course not work while driving on slope).
I use window sizes from `64` to `280` pixels (i.e. filling the horizontal full-slice), in increments of `16` pixels.
I keep an overlap of `2/3` between sliding windows. This allows for robust detection and filtering of outliers.

![alt text][image4]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below are the steps in prediction pipeline:

1. Create a heat-map array with length and width same as the image, initialized to all zeros

2. Iterate over `window_size` between `64` to `280` pixels

3. Take a horizontal slice of the image of size `window_size`, ignoring region above `x=400` and below `x=680` 

    a) In order to improve performance, compute `hog` feature in one go over entire horizontal slice of size `window_size`. Note that we keep a separate copy of image in `YCrCb` format and use only the `Y` channel for `hog` computation. 
    
    b) compute the rescaling factor required to convert the image to `64x64` size, this will be used to compute number of windows among other uses
    
    c) Iterate over all windows and perform below computations
    
        - Extract `hog`, `spatial` and `histogram` features specific to the window and concatenate to create feature vector
        
        - Use the classifier to predict if vehicle is found in the window. If so, increment pixels pertaining to the window by `1`
        
4. Return heat-map and a normalized image (for display-purpose)

Below images capture the original image, sliding-windows and predicted heat-map image
![alt_text][image5]

![alt text][image4]

![alt_text][image6]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_with_cars.mp4)

The video uses 'Advanced Lane Finding' exercise to render end-to-end lane and vehicle detection.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.
I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of 
`scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

1. Original image

![alt text][image10]

2. Heat-map image

![alt text][image11]

3. Separating bounding boxes with `scipy.ndimage.measurements.label()`

![alt text][image7]

4. Image with bounding boxes

![alt text][image9]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.
Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, 
where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

1. Detecting vehicles coming from opposite side, at lane crossing, vehicles other than cars etc. This is easy to fix with augmenting the dataset (e.g. using [Udacity vehicle-detection datasets](https://github.com/udacity/self-driving-car/tree/master/vehicle-detection/)).

2. Making detection more robust: the hand-tuning of parameters leaves lot of scope for improvement. Given the variability of light, seasons, traffic etc the manual approach quickly breaks down. An end-to-end deep learning approach should work better. This will further benefit from architectures specific to object detection (such as detect-net from nvidia).
