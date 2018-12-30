# Includes
import os
import numpy as np
import cv2
import glob
import math
import time
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


# Default HSV Binary Thresholds
_HSV_YW_THRESHOLDS = [np.array([15, 127, 127], dtype=np.uint8),  # yellow_dark
                      np.array([25, 255, 255], dtype=np.uint8),  # yellow_light
                      np.array([0, 0, 200], dtype=np.uint8),     # white_dark
                      np.array([255, 30, 255], dtype=np.uint8)]  # white_light


# read cv2 image as RGB image instead of BGR
def cv2_imread_rgb(fname):
    return cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)


# write cv2 image as RGB image instead of BGR
def cv2_imwrite_rgb(fname, img):
    cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# plot images
def plot_images(images, titles=None, cols=3, fontsize=12):

    n_imgs = len(images)

    if images is None or n_imgs < 1:
        print("No images to display.")
        return

    img_h, img_w = images[0].shape[:2]
    rows = math.ceil(n_imgs / cols)
    width = 21  # 15
    row_height = math.ceil((width/cols)*(img_h/img_w))  # they are 1280*720

    plt.figure(1, figsize=(width, row_height * rows))

    for i, image in enumerate(images):
        if len(image.shape) > 2:
            cmap = None
        else:
            cmap = 'gray'
        title = ""
        if titles is not None and i < len(titles):
            title = titles[i]
        plt.subplot(rows, cols, i+1)
        plt.title(title, fontsize=fontsize)
        plt.imshow(image, cmap=cmap)

    plt.tight_layout()
    plt.show()


# function to create image frames from a video
def create_frame_images(video_fname, tsec_start, tsec_end, output_folder="more_test_images/"):
    # name format for image frames we save
    frame_names = "frame_s" + str(int(tsec_start)) + "_e" + str(int(tsec_end)) + "_%04d.jpg"
    # get the video clip and save it's frames
    clip1 = VideoFileClip(video_fname).subclip(tsec_start, tsec_end)
    image_names = clip1.write_images_sequence(os.path.join(output_folder,frame_names))

    # return a list of imafe frame names saved
    return image_names


def get_binary_polynomial_image(binary_warped, binary_pixels, left_fitx, right_fitx, ploty):

    leftx, lefty, rightx, righty = binary_pixels
    binary_polynomial_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Visualization
    # Colors in the left and right lane regions
    binary_polynomial_img[lefty, leftx] = [255, 0, 0]
    binary_polynomial_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    plt.text(400, 500, r'$f_{left}(y) = A_{left}y^2 + B_{left}y + C_{left}$', color='y', fontsize=8)
    plt.text(450, 100, r'$f_{right}(y) = A_{right}y^2 + B_{right}y + C_{right}$', color='y', fontsize=8)
    plt.arrow(420, 520, -60, 60, linewidth=0.5, head_width=10, head_length=10, fc='y', ec='y')
    plt.arrow(880, 120, 60, 60, linewidth=0.5, head_width=10, head_length=10, fc='y', ec='y')

    return binary_polynomial_img


def get_HSV_threshold_binary(img, hsv_thresholds=_HSV_YW_THRESHOLDS):
    """
    Thresholds the image to binary based on HSV color space.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if hsv_thresholds is not None and len(hsv_thresholds) == 4:
        yellow_dark = hsv_thresholds[0]
        yellow_light = hsv_thresholds[1]
        white_dark = hsv_thresholds[2]
        white_light = hsv_thresholds[3]
    else:
        yellow_dark = np.array([15, 127, 127], dtype=np.uint8)
        yellow_light = np.array([25, 255, 255], dtype=np.uint8)
        white_dark = np.array([0, 0, 200], dtype=np.uint8)
        white_light = np.array([255, 30, 255], dtype=np.uint8)

    yellow_range = cv2.inRange(img, yellow_dark, yellow_light)
    white_range = cv2.inRange(img, white_dark, white_light)

    yellows_or_whites = yellow_range | white_range
    img = cv2.bitwise_and(img, img, mask=yellows_or_whites)

    return np.uint8(np.sum(img, axis=2, keepdims=False) > 0)


def get_thresholded_binary(img, s_thresh=(200, 225), sx_thresh=(20, 100), r_thresh=(210, 255)):

    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold R channel
    R = img[:, :, 2]
    r_binary = np.zeros_like(R)
    r_binary[(R > r_thresh[0]) & (R <= r_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack((r_binary, sxbinary, s_binary)) * 255

    thresh_binary = np.zeros_like(s_binary)
    thresh_binary[(s_binary == 1) | (sxbinary == 1) | (r_binary == 1)] = 1

    return color_binary, thresh_binary


#
# NEW THRESHOLD BINARY CODE
#
def get_image_channels(xyz_img):
    # ONLY A CONVENIENCE FUNCTION, NO ERROR CHECKING HERE.
    return xyz_img[:,:,0], xyz_img[:,:,1], xyz_img[:,:,2]


def get_color_space_image_and_channels(rgb_image, output_color_space=None):
    if len(rgb_image.shape) is not 3:
        print("ERROR: INPUT IMAGE MUST HAVE 3 (R, G, B) CHANNELS.")
        return

    if output_color_space is None:
        output_color_space = "RGB"

    if output_color_space is "HLS":
        xyz_image = cv2.cvtColor(rgb_image, code=cv2.COLOR_RGB2HLS)
    elif output_color_space is "HSV":
        xyz_image = cv2.cvtColor(rgb_image, code=cv2.COLOR_RGB2HSV)
    elif output_color_space is "LAB":
        xyz_image = cv2.cvtColor(rgb_image, code=cv2.COLOR_RGB2Lab)
    else:
        xyz_image = rgb_image

    x, y, z = get_image_channels(xyz_image)

    return xyz_image, x, y, z


def get_LAB_L_threshold_binary(rgb_image, yellow_min=175, thresh=(190, 225)):
    lab, ch_l, ch_a, ch_b = get_color_space_image_and_channels(rgb_image, "LAB")

    #     # use the b-channel, if the yellow color exists, normalize the channel
    #     if np.max(ch_l) > yellow_min:
    ch_l = np.uint8(ch_l * (255 / np.max(ch_l)))

    # apply threshold, exclusive lower - inclusive upper
    binary = np.zeros_like(ch_l)
    binary[((ch_l > thresh[0]) & (ch_l <= thresh[1]))] = 1

    return binary, ch_l


def get_LAB_A_threshold_binary(rgb_image, yellow_min=175, thresh=(190, 225)):
    lab, ch_l, ch_a, ch_b = get_color_space_image_and_channels(rgb_image, "LAB")

    #     # use the b-channel, if the yellow color exists, normalize the channel
    #     if np.max(ch_a) > yellow_min:
    ch_a = np.uint8(ch_a * (255 / np.max(ch_a)))

    # apply threshold, exclusive lower - inclusive upper
    binary = np.zeros_like(ch_a)
    binary[((ch_a > thresh[0]) & (ch_a <= thresh[1]))] = 1

    return binary, ch_a


def get_LAB_B_threshold_binary(rgb_image, yellow_min=175, thresh=(190, 225)):
    lab, ch_l, ch_a, ch_b = get_color_space_image_and_channels(rgb_image, "LAB")

    # use the b-channel, if the yellow color exists, normalize the channel
    if np.max(ch_b) > yellow_min:
        ch_b = np.uint8(ch_b * (255 / np.max(ch_b)))

    # apply threshold, exclusive lower - inclusive upper
    binary = np.zeros_like(ch_b)
    binary[((ch_b > thresh[0]) & (ch_b <= thresh[1]))] = 1

    return binary, ch_b


def get_HLS_L_threshold_binary(rgb_image, thresh=(220, 225)):
    hls, ch_h, ch_l, ch_s = get_color_space_image_and_channels(rgb_image, "HLS")
    ch_l = np.uint8(ch_l * (255 / np.max(ch_l)))  # normalize channel

    # apply threshold, exclusive lower - inclusive upper
    binary = np.zeros_like(ch_l)
    binary[((ch_l > thresh[0]) & (ch_l <= thresh[1]))] = 1

    return binary, ch_l


def get_HLS_S_threshold_binary(rgb_image, thresh=(220, 225)):
    hls, ch_h, ch_l, ch_s = get_color_space_image_and_channels(rgb_image, "HLS")
    ch_s = np.uint8(ch_s * (255 / np.max(ch_s)))  # normalize channel

    # apply threshold, exclusive lower - inclusive upper
    binary = np.zeros_like(ch_s)
    binary[((ch_s > thresh[0]) & (ch_s <= thresh[1]))] = 1

    return binary, ch_s


def get_LAB_threshold_binary(rgb_image, l_thresh=(210, 255), b_thresh=(190, 255)):
    lab_l_binary, lab_ch_l = get_LAB_L_threshold_binary(rgb_image, thresh=l_thresh)
    lab_b_binary, lab_ch_b = get_LAB_B_threshold_binary(rgb_image, thresh=b_thresh)

    # Combine Lab L and B channel thresholds
    combined_LB_binary = np.zeros_like(lab_l_binary)
    combined_LB_binary[(lab_l_binary == 1) | (lab_b_binary == 1)] = 1

    return lab_ch_l, lab_ch_b, lab_l_binary, lab_b_binary, combined_LB_binary


#
# Radius of curvature calculations
#
def get_radius_of_curvature_in_meters(fit, n_H, mx=3.7/666, my=27.432/720):
    # NOTE: This function is based on the below suggestion in the project description.
    # An insightful student has suggested an alternative approach which may scale more efficiently.
    # That is, once the parabola coefficients are obtained, in pixels, convert them into meters.
    # For example, if the parabola is x= a*(y**2) +b*y+c; and mx and my are the scale for the x and y axis,
    # respectively (in meters/pixel); then the scaled parabola is x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c

    # scale the fit from pixels to meters
    a, b, _ = fit
    A = (mx / my ** 2) * a
    B = (mx / my) * b

    # return the radius of curvature for the scaled Y-coordinate
    Y = (n_H - 1) * my
    return ((1 + (2 * A * Y + B) ** 2) ** (3 / 2)) / np.abs(2 * A)


#
# Data strings to superimpose on video
#
def get_data_strings(roc, vehicle_offset_meters):
    if roc is not None:
        roc_text = "Radius of Curvature = " + "{:04.2f}".format(roc) + "(m)"
    else:
        roc_text = "Radius of Curvature: Unidentified"

    if vehicle_offset_meters is not None:
        side = ""
        offset = abs(vehicle_offset_meters)
        if vehicle_offset_meters > 0:
            side = "right"
        elif vehicle_offset_meters < 0:
            side = "left"
        vehicle_pos_text = "Vehicle is " + "{:04.3f}".format(offset) + "m " + side + " of center"
    else:
        vehicle_pos_text = "Vehicle Position: Unidentified"

    return roc_text, vehicle_pos_text

