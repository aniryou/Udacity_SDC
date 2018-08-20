## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/road_transformed.png "Road Transformed"
[image3]: ./examples/image_thresholds.png "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/polynomial_fit_window.png "Fit Visual"
[image6]: ./examples/test6.jpg "Test image"
[image7]: ./examples/detected_lanes_persp.png "Output"
[image8]: ./examples/output.png "Output"
[video1]: ./project_video_with_lanes.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in cells 2 to 5 of the IPython notebook located in "Advanced Lane Finding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `lst_objpoints` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I 
successfully detect all chessboard corners in a test image.  
`lst_imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `lst_objpoints` and `lst_imgpoints` to compute the camera calibration and distortion coefficients 
using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` 
function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cells 7 through 18 in "Advanced Lane Finding.ipynb").  
Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()` and `unwarp()`, which appears in cells 19 through 21 in "Advanced Lane Finding.ipynb".  
The `warp()` function takes as inputs an image (`img`).  I chose the hardcoded the source and destination points in the following manner:

```python
src = np.array([(100,720), (560,450), (720,450), (1180,720)], np.float32)
dst = np.array([(80,720), (80,0), (1200,0), (1200,720)], np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 100, 720      | 80, 720       | 
| 560, 450      | 80, 0         |
| 720, 450      | 1200, 0       |
| 1180, 720     | 1200, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` 
points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I implemented a class `Line` (cell 22 of "Advanced Lane Finding.ipynb") to encapsulate a lane.

I used histogram analysis of the bottom half of warped image to locate the base of left and right lanes.
Once the base is found, we can incrementally move upwards to detect lane pixels within a fixed window size - this allows us to filter
any outlier pixels, which might be passed from threshold filtering stage of the pipeline.
This is implemented in function `_find_lane_pixels` of `Line` class elaborated in image below:

![alt text][image5]

To avoid having to recompute the lane pixels each time, which is both time consuming and error-prone, we use previously detected lane line if available.
This is implemented in function `find_lane_pixels()`. 
Once I have the lane pixels, I fitted a second order polynomial to detect the lane lines.
This is captured in cell 21 and called from functions `fit()`, which take in a half-sliced vertical image.
We smooth the coefficient computed by fitting polynomial with previous coefficient, to make it smooth.
Finally, we predict the lane line in `pred()` function, which again takes in same half-slices vertical image as fit.

The complete process is captured in images below:

![alt text][image6]

![alt text][image7]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cell 40 in my code in "Advanced Lane Finding.ipynb". 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 26 of "Advanced Lane Finding.ipynb".  
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_with_lanes.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Few of the major challenges I faced during implementation of this project as mentioned below:

- Find a systematic way of identifing thresholds for gradients and color channels, and combining them. This is currently very brittle.
- Balancing the perspective view to discard outliers and accomodate changes in vehicle alignment from center
- Adapting coefficients: currently the polynomial coefficients for lane lines are computed freshly each time or re-used (or combination of the two). This fails for instance for the challenge video where several aspects of the image suddenly change.

A more systematic method might involve drawing the lanes manually for images (which makes it easy to detect) and doing a grid search over combination of parameters.

