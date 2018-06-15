# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./writeup_images/whiteCarLaneSwitch.jpg "Original"
[image2]: ./writeup_images/whiteCarLaneSwitch_gray.jpg "Grayscale"
[image3]: ./writeup_images/whiteCarLaneSwitch_blur.jpg "Blurred"
[image5]: ./writeup_images/whiteCarLaneSwitch_canny.jpg "Canny"
[image6]: ./writeup_images/whiteCarLaneSwitch_mask.jpg "Mask"
[image7]: ./writeup_images/whiteCarLaneSwitch_lines.jpg "Lines"
[image8]: ./writeup_images/whiteCarLaneSwitch_lanes.jpg "Lanes"
[image9]: ./writeup_images/whiteCarLaneSwitch_cap.jpg "Capped"

---

### Reflection

### 1. Pipeline Description

The pipeline consisted of 7 steps.

1. Converted image to grayscale: this is required for subsequent edge-detection algorithms to work

![Grayscale][image2]

2. Applied gaussian blur to smooth the image and get rid of noise, after some experimentation I found kernel size of 5 works best

![Blurred][image3]

3. Applied canny edge detection algorithm (as mentioned in the lecture I kept low-to-high ratio of 1:2)

![Canny][image5]

4. Created a quadilateral mask for lane markings, since position of camera is fixed (slightly to left for countries with right-side driving)

![Mask][image6]

5. Extracted lines using probabilistic hough transform, with parameters fine-tuned by trial and error

![Lines][image7]

6. If any lines were detected, splitted the lines into candidates for left and right lane by applying filter on slope

7. Combined the lines from previous step by average the slope and intercept (weighted by line segment length) and extrapolate to extreme ends on both sides. I apply a minimum length filter on lines to filter outliers, but if the filter seems too restrictive (i.e. no lines exceeds the minimum length), then I just take the longest line

![Extrapolation][image8]

8. Incorporate prior information if available, by taking linear combination of current and prior lane. Since I initialize the left and right lane to prior values, if a particular lane isn't detected the overall effect is falling back to prior value for that lane. This is evident in challenge video, where for a short segment, the left lane isn't detected and we carry over last detected lane.

9. Applied mask to keep only segments within region of interest

![Capped][image9]


### 2. Potential Shortcomings


Below are some immediate shortcomings of the pipeline

1. Doesn't work for sharp-turns

2. Might not work for merging or splitting lanes (e.g. in case of toll)

3. Sometimes edge of the line steps outside the lane


### 3. Improvements

Below are some possible improvements

1. Instead of extrapolating to line with single slope, allow smaller line segments with gradually changing slope

2. Normalize the image to even out effects of varying light / color and saturation

3. Work in HSV colorspace to apply range filters