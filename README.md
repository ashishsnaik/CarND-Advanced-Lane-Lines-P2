## Advanced Lane Finding for Self-driving Cars
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image_sample]: ./output_images/straight_lines1_lanefill.jpg "Straight Line Image 1 Lanefill"


In this project, my goal is to write a software pipeline to identify and highlight the lane boundaries in a video taken from a front-facing camera fixed to the center of a vehicle. Please see below for the link to a detailed writeup of the project.

[Link to Project Write-Up](./P2_AdvLaneLines_Writeup.md)
---

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing the pipeline on single frames. The pipeline is finally tested on this [project_video](./project_video.mp4) to generate this [project_video_with_lanefill](./output_images/project_video_lanefill.mp4).

Sample Visual
---

![alt text][image_sample]