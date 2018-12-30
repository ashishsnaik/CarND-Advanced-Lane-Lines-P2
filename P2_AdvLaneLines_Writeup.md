## **Advanced Lane Finding Project**
---
### **The Goal of this Project**

In this project, my goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car.


**Project Assumptions**

* The front facing camera is fixed at the center of the vehicle.
* The vehicle is driven on a road with a fairly level surface.

**The steps of this project are the following:**

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/chessboard_corners/calibration2_corners.jpg "Corners Image"
[image1]: ./camera_cal/calibration1.jpg "Original Distorted Image"
[image2]: ./output_images/calibration1_undist.jpg "Undistorted Image"
[image3]: ./test_images/test1.jpg "Test Image"
[image4]: ./output_images/test1_undist.jpg "Undistorted Test Image 1"
[image5]: ./output_images/test1_GSR_binary.jpg "GSR Thresh Binary"
[image6]: ./output_images/test1_HSV_binary.jpg "HSV Thresh Binary"
[image7]: ./output_images/test1_LAB_binary.jpg "LAB Thresh Binary"
[image8]: ./output_images/test2_undist.jpg "Undistorted Test Image 2"
[image9]: ./output_images/test2_LAB_binary.jpg "LAB Thresh Binary"
[image10]: ./output_images/straight_lines1_src_poly.jpg "Source Poly"
[image11]: ./output_images/straight_lines1_dst_lines.jpg "Destination lines"
[image12]: ./output_images/binary_polynomial_img.png "Binary Polynomial Image"
[image13]: ./output_images/straight_lines1_lanefill.jpg "Straight Line Image 1 Lanefill"

[video1]: ./output_images/project_video_lanefill.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/1966/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it! This writeup includes statements and supporting figures / images that explain how each rubric item was addressed, and specifically where in my code each step was handled.

***Note:*** *The **advanced_lane_finding.ipynb** shows execution of the project code and is referred to in all relevant sections below.*

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Source files: `camera_calibration.py` and `advanced_lane_finding.ipynb`

The class `ChessboardCameraCalibrator` handles the camera calibration based on chessboard images, with 9x6 pattern being the default one, and also performs the undistortion of an image based on the calibration. The two operations are defined by the `calibrate_camera()` and `undistort_image()` class member functions. The source file also contains a non-member `test_camera_calibration_and_undistortion()` test function, which can be used to test the calibration and undistortion functionalities.

For camera calibration, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here, I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection done using `cv2.findChessboardCorners()`.  I then use the output `object_points` and `image_points` to compute the camera matrix and the distortion coefficients using the `cv2.calibrateCamera()` function.

For undistorting an image, I apply to the image the camera matrix and distortion coefficients above, using `cv2.undistort()`.

The `camera_cal` folder contains the provided 20 calibrations images. The calibration process is able to find corners for 17 of the 20 images and has a re-projection error of 1.186897 (RMS). 

The calibration process was unable to find corners for 3 calibration images, namely, calibration1.jpg, calibration4.jpg and calibration5.jpg. In `advanced_lane_finding.ipynb` section *'Camera Calibration'*, I verify the calibration on these "unused" calibration images, a sample result of which is shown below. Also shown is a sample result of chessboard corners found in on of the images. Chessboard corners images are saved under the `output_images/chessboard_corners/` folder.

**Chessboard Corners Detection Result (Sample)**

![][image0]

**Distortion Correction Result (Sample: calibration1.jpg)**

Original Distorted Image   |Undistorted Image
:-------------------------:|:-------------------------:
![][image1]                |![][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Source file(s): `camera_calibration.py` and `advanced_lane_finding.ipynb`

The implementation for applying distortion correction to test images is in the non-member test function `test_camera_calibration_and_undistortion()` in `camera_calibration.py`. The test function applies the distortion correction to the test images present in the `test_images/` folder and stores the result in the `output_images/` folder with "\_undist" postfix, for eg., distortion correction result of `test_images/test1.jpg` is stored as `output_images/test1_undist.jpg`.

In the `advanced_lane_finding.ipynb`, under section *'Lane Detection Pipeling (single images)'*' sub-section *'Apply undistortion to test images and display results'*, I use the above test function and display the results. 

Below is a sample result of the distortation correction applied to the test1.jpg test image.

**Distortion Correction Result (Sample)**

Original Distorted Image   |Undistorted Image
:-------------------------:|:-------------------------:
![][image3]                |![][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Source file(s): `utils.py` and `advanced_lane_finding.ipynb`

To generate the thresholded binary image, I tried a few approaches, 3 of which are listed below and their implementation is in `utils.py`:
* Thresholding using a combination of Gradient (Sobel-X on grayscale), S channel (HLS color space), and R channel (RGB color space) implemented as `get_thresholded_binary()` function.
* HSV color space thresholding using ranges for white and yellow, implemented as the `get_HSV_threshold_binary()` function.
* LAB color space with thresholding for L and B channels, implemented as the `get_LAB_threshold_binary()` function.

In `advanced_lane_finding.ipynb`, section *'Lane Detection Pipeling (single images)'* sub-section *'Generate thresholded binary images'* shows the usage and results of binary thresholding. The thresholded binary images of all test images are saved in the `output_images/` folder, with postfix "\_GSR_binary", "\_HSV_binary", and "\_LAB_binary" respectively.

Below is a comparative visual of the above 3 techniques I experimented with.

Test Image 1 (Undistorted) |GSR Thresh Binary |HSV Thresh Binary |LAB Thresh Binary
:-------------------------:|:----------------:|:----------------:|:----------------:
![][image4]                |![][image5]       |![][image6]       |![][image7]

A closer look at the binary images of the above 3 methods revealed that, overall, although HSV thresholding looks good, LAB thresholded binary looked better and cleaner. Thresholding with a combination of Gradient, S channel, and R channel didn't seem to be at par with the other two, as far as identifying the lane was concerned. So, I used the the LAB binary thresholding for the project.

Below is a sample output of this step using the LAB binary thresholding.

Test Image 2 (Undistorted) |LAB Thresh Binary
:-------------------------:|:-------------------------:
![][image8]                |![][image9]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Source file(s): `perspective_transform.py` and `advanced_lane_finding.ipynb`

The perspective transform is handled by the `PerspectiveTransform` Class in `perspective_transform.py`. A non-member test function `visually_verify_perspective_transform()` is also present in the `py` file. Section `Lane Detection Pipeling (single images)` sub-section `Perspective Transform Verification` in `advanced_lane_finding.ipynb` uses this test function.

The code for my perspective transform, `PerspectiveTransform` Class, includes a member function called `transform_image()` and `get_birds_eye_view_image()`.  The `transform_image()` function takes as inputs an `image`, and a boolean parameter `unwarp`, which specifies whether to Warp the image or Un-Warp it. The `get_birds_eye_view_image()`, which also takes in an image as input, uses the `transform_image()` funtion to Warp the image and also scales a binary image from 0 or 1 to 0 or 255, respectively. This function also takes in a boolean `display_images` parameter, which gives an option to display the images with source and destination points, for debugging purposes.

*The perspective transform functionality assumes that the front-facing camera is fixed at the center of the vehicle and that the road is fairly level.*

I choose to hardcode the source (`self.src`) and destination (`self.dst`) points as below, based on the `straight_lines1.jpg` test image.

```python
self.src = np.float32([[210, 720],
					   [596, 450],
					   [684, 450],
					   [1104, 720]])
self.dst = np.float32([[320, 720],
					   [320, 0],
					   [960, 0],
					   [960, 720]])
```
This initialization is done in the constructor `__init__()`, though, the constructor also gives an option to specify other source and destination points during object creation. The constructor also initializes the map-matrix `self.M` and inverse map-matrix `self.M_inv` for the transformations, based on the source and destination points.

I verified that my perspective transform was working as expected by drawing the `self.src` and `self.dst` points onto the `straight_lines1.jpg` test image and its warped counterpart to verify that the lines appear parallel in the warped image. The result is saved in the `output_images/` folder as `straight_lines1_src_poly.jpg` and `straight_lines1_dst_lines.jpg`, and also shown below. 

In `advanced_lane_finding.ipynb`, section *'Lane Detection Pipeling (single images)'* sub-section *'Perspective Transform Verification'* shows this test code execution. 

Undistorted image with source points drawn|Warped image with dest. points drawn 
:-------------------------:|:-------------------------:
![][image10]                |![][image11]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Source file(s): `lane_line.py`, `lane_detector.py` and `advanced_lane_finding.ipynb`

My code for lane detection works on the bird's-eye (perspective transformed / warped) binary view of the road image, and is distributed across two source files: 

1. `lane_lines.py` - implements a `Line` Class that represents a lane line and performs functionalities such as fitting a polynomial to the line - *member functions: `fit_poly()` and `determine_best_fit()`* and finding the line's radius of curvature - *member function: calculate_radius_of_curvature()*. 
2. `lane_detector.py` - implements a `LaneDetector` Class to detect the lane. It contains objects of the `Line` Class to represent the left and right lane lines seperately, and implements member functions to perform operations such as processing an input image - *member function: `process_image()`*, detect the lane - *member function: `detect_lane()`*, find the lane line pixels in an image - *member functions: `find_lane_pixels()`, `search_lane_pixels()`, and  `search_lane_pixels_around_poly()`*. In addition having member variables to include the left and right lane `Line` objects, the class also contains objects to represent the `ChessboardCameraCalibrator` and `PerspectiveTransform`.


To identify the lane pixels, I use histogram peaks in the lower half of the thresholded binary image to identify the center x-position of the bases of left and right lane lines. Using the base as a starting point, I use a sliding window, with 200-pixels width (center x-position +/- 100 pixels margin), placed around the centers and traverse up the image to search for the lane lines. The member function `search_lane_pixels()` uses this technique. Once the lane lines are found, I fit polynomials to the set of left and right lane pixels. For the next frame, instead of searching the lane pixels from scratch, I search for lane pixels in the margin around the previously fitted polynomial. This mechanism is implemented by the member function `search_lane_pixels_around_poly()`.

For video processing, I also use polynomial smoothing which takes into account the difference between the polynomial coefficients of the current fit and the previous best fit and smoothens the current fit, if the difference is above a certain threshold. This is implemented by the `Line` Class member-function `determine_best_fit()`.

Section `Lane Detection Pipeling (single images)` sub-section `Lane-line Pixels with Polynomial Fit` in `advanced_lane_finding.ipynb` shows the usage of this functionality.

Below is the output of the lane pixels found for *test2.jpg* test image and my 2nd order polynomial fits for the left and right lane lines. The output is saved as `binary_polynomial_img.png` in `output_images/` folder.

![alt text][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Source files: `lane_line.py` and `lane_detector.py`

Once I fit the lane line polynomial, I also calculate the lane-line's radius of the curvature in meters, using the `Line` Class member function `calculate_radius_of_curvature()`. The function uses the member variables `xm_per_pix` and `ym_per_pix` (meters per pixel in the x and y directions, respectively), to map the pixel distance to meters. By default, the code assumes that the vehicle steering has a real-time response and the radius of curvature is calculated at the current position of the vehicle, i.e. the base of the image, so as to steer the vehicle correctly. Though, the distance from the vehicle at which the radius of curvature is calculated can be changed using the `mdist_from_pov` (distance from the point-of-view in meters) input parameter to the function.

I calculate the radius of curvature using the below formula:
```python
((1 + (2 * A * y_eval * my + B) ** 2 ) ** (3/2) ) / np.abs(2 * A)
```
*where,*

*A and B* = 2nd and 1st order polynomial coefficients

*y_eval* = pixel distance at which to calculate the radius

*my* = meters per pixel in the y-direction

The above describes calculating the radius of curvature for a single lane line. The radius of curvature of the lane itself is calculated by the `detect_lane()` member function of `LaneDetector` Class in `lane_detector.py` and is done by taking the average of the left and right lane's radius of curvature. 

In my code I also calculate the lane center-line polynomial, as the average of the left and right lane line polynomial fits, to mark it as a reference in the lane-fill. One other way of calculating the lane radius of curvature would be to calculate it for the lane center polynomial. I have also kept that commented code in the `detect_lane()` member function.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Source file: `advanced_lane_finding.ipynb`

I implemented this step in `advanced_lane_finding.ipynb` Section `Lane Detection Pipeling (single images)` sub-section `Lane Detection on Test Images`.

*I have also plotted the lane-center line, as reference, on the test images and videos.*

Here is an example of my result on a test image:

![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Source file: `advanced_lane_finding.ipynb`

The code for processing the project test video can be found under Section `Lane Detection Pipeling on Project Test Video` in `advanced_lane_finding.ipynb`. The output video is saved as `project_video_lanefill.mp4` in the `output_images/` folder.

Here's a [link to my video result](./output_images/project_video_lanefill.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For camera calibration using the provided 20 calibration images, the calibration process wasn't able to identify corners for 3 images and finished with a re-projection error of 1.186897 (RMS). Adding more calibration images would be helpful to better the calibration accuracy.


I spent a large amount of my experimentation time on generating a good thresholded binary image, in which the lane lines would be clearly visible and with the least amount of noise from other portions of the image. As I mentioned in the threshold binary section above, transforming the image to the CIELAB color space and thresholding it, gave me the best results (so far), especially in identifying the lane with a combinition of yellow and white lane lines. Though, the current implementation is not robust enough and would fail to work for challenging road, lighting, and/or whether conditions such as dense shadows, darkness, reflections, poor or missing patches of road markings, text or other markings on the road (speed limits, carpool signs) , rainy, snowy, or dusty conditions etc. The implementation certainly has room for improvements, which can be achieved by experiment with more color spaces, thresholds, and combinations thereof. 

As currently implemented, the polynomial smoothening logic would be sensitive to the combinition of the vehicle speed and the stability with which the vehicle is steered, and these parameters could be included in implementing a better polynomial smoothing algorithm. Also, adding an upper bound to the difference between the current and previous best fit, to discard possible erroneous fit coefficients, could also be considered.  

The overall lane detection algorithm would fail if one of the lane lines isn't in the camera's field of view as might be the case with winding and wavy roads. This could be taken care by considering whether the left and right lane-lines are detected, independently, and implementing logic which would consider the absense of one of those lane lines. 
