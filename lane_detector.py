from utils import *
from camera_calibration import *
from perspective_transform import *
from lane_line import *

# suppress scientific notation
np.set_printoptions(suppress=True)


class LaneDetector:
    def __init__(self, nwindows=9, margin=100, minpix=50, enable_smoothing=True, nsmooth=5,
                 diff_thresholds=[0.0002, 0.02, 85.0], xm_per_pix=3.7/666, ym_per_pix=27.432/720):

        # camera calibrator
        self.camera_calibrator = ChessboardCameraCalibrator()
        # perspective transform
        self.perspective_transform = PerspectiveTransform()

        # enable smoothing
        self.enable_smoothing = enable_smoothing

        # lane width in meters as per US regulations is minimum 12 feet or 3.7 meters,
        # which in the perspective transformed image is about 666 pixels (right_x_intercept - left_x_intercept).
        self.xm_per_pix = xm_per_pix
        # white dashed line is 10 feet (3.048 meters) and 720 px height of
        # perspective transformed image is about 9 times i.e. 27.432 meters
        self.ym_per_pix = ym_per_pix

        # left and right lane lines
        self.left_lane_line = Line(enable_smoothing=self.enable_smoothing, nsmooth=nsmooth,
                                   diff_thresholds=diff_thresholds, xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix)
        self.right_lane_line = Line(enable_smoothing=self.enable_smoothing, nsmooth=nsmooth,
                                    diff_thresholds=diff_thresholds, xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix)

        # vehicle offset in meters from the center of lane
        self.vehicle_offset_meters = None
        # radius of curvature
        self.radius_of_curvature = None

        ## HYPERPARAMETERS
        # number of sliding windows to traverse up the
        # image height for searching lane pixels
        self.nwindows=nwindows
        # width of each window +/- margin, and width of the
        # margin around the previous polynomial to search
        self.margin=margin
        # minimum number of pixels found to recenter window
        self.minpix = minpix

    def reset(self):
        # reset appropriate member variables
        self.vehicle_offset_meters = None
        self.radius_of_curvature = None
        # reset the left and right lane lines
        self.left_lane_line.reset()
        self.right_lane_line.reset()

    # finds lane pixels in the warped binary image
    def search_lane_pixels(self, binary_warped):

        # take histogram of the bottom half of the image where the lane lines are mostly straight.
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        # find the peak of the left and right halves of the histogram (essentially, also for the img base).
        # this will be the starting point for the left and right lines.
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # set window height based on image height and number of windows
        window_height = np.int(binary_warped.shape[0] // self.nwindows)
        # identify x and y for all non-zero positions in the image
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])  # rows are y axis
        nonzero_x = np.array(nonzero[1])  # columns are x axis

        # current position to be updated later for each position in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # empty lists to record left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # step through the windows
        for window in range(self.nwindows):

            # identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            # find the four below boundaries of the window
            win_xleft_low = leftx_current - self.margin  # Update this
            win_xleft_high = leftx_current + self.margin  # 0  # Update this
            win_xright_low = rightx_current - self.margin  # Update this
            win_xright_high = rightx_current + self.margin  # Update this

            # identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high)
                              & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high)
                               & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

            # append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # if we found > minpix pixels, recenter next window (`right` or `leftx_current`)
            # on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # avoids an error if the above is not implemented fully
            print("ERROR: Value Error!")
            return False

        # extract left and right line pixel positions and set the allx and ally for left and right lane lines
        self.left_lane_line.set_all_xy(nonzero_x[left_lane_inds], nonzero_y[left_lane_inds])
        self.right_lane_line.set_all_xy(nonzero_x[right_lane_inds], nonzero_y[right_lane_inds])

        return True

    def search_lane_pixels_around_poly(self, binary_warped):

        # get the previous polyfit coefficients
        left_fit = self.left_lane_line.best_fit
        right_fit = self.right_lane_line.best_fit

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # set the area of search based on activated x-values within the +/- margin
        # of our previously fitted polynomial function.
        left_lane_inds = ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y +
                                       left_fit[2] - self.margin)) & (nonzero_x < (left_fit[0] * (nonzero_y ** 2) +
                                                                             left_fit[1] * nonzero_y + left_fit[
                                                                                 2] + self.margin)))
        right_lane_inds = ((nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y +
                                        right_fit[2] - self.margin)) & (nonzero_x < (right_fit[0] * (nonzero_y ** 2) +
                                                                               right_fit[1] * nonzero_y + right_fit[
                                                                                   2] + self.margin)))

        # extract left and right line pixel positions and set the allx and ally for left and right lane lines
        self.left_lane_line.set_all_xy(nonzero_x[left_lane_inds], nonzero_y[left_lane_inds])
        self.right_lane_line.set_all_xy(nonzero_x[right_lane_inds], nonzero_y[right_lane_inds])

        return True

    def find_lane_pixels(self, binary_warped):

        # if we already have a fitted polyline search around it,
        # else use sliding window to find the lane pixels in the image
        if self.left_lane_line.is_detected() and self.right_lane_line.is_detected():
            retval = self.search_lane_pixels_around_poly(binary_warped)
        else:
            # search lane pixels from scratch
            retval = self.search_lane_pixels(binary_warped)

        return retval

    def fit_polynomial(self, binary_warped):
        self.left_lane_line.fit_poly(binary_warped)
        self.right_lane_line.fit_poly(binary_warped)

    def detect_lane(self, binary_warped):


        # lane fill to return
        lane_fill = np.ascontiguousarray(np.zeros_like(
            np.dstack((binary_warped, binary_warped, binary_warped))), dtype=np.uint8)

        retval = self.find_lane_pixels(binary_warped)

        # if lane pixels are found mark the lane, else return
        # the original binary image converted to 3-channel
        if retval is True:

            # fit the left and right lane polynomials
            self.fit_polynomial(binary_warped)

            left_fit, left_fitx = self.left_lane_line.get_best_fit_data()
            right_fit, right_fitx = self.right_lane_line.get_best_fit_data()

            if left_fit is not None and left_fitx is not None and right_fit is not None and right_fitx is not None:
                n_H, n_W = binary_warped.shape

                # x and y values for plotting
                ploty = np.linspace(0, n_H - 1, n_H)
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
                pts = np.hstack((pts_left, pts_right))

                # get the lane center poly fit, we will use the lane center to plot the center line on the image and
                # the lane center x-intercept to calculate the vehicle offset from center; the same can be
                # calculated using the x-intercepts of the left and right lane lines.
                center_fit = np.mean([left_fit, right_fit], axis=0)
                center_fitx = center_fit[0] * ploty ** 2 + center_fit[1] * ploty + center_fit[2]
                pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])

                # vehicle offset from lane center
                lane_center = center_fitx[len(center_fitx)-1]
                # assuming that the camera is fixed at the vehicle center
                vehicle_center = n_W//2
                self.vehicle_offset_meters = (vehicle_center-lane_center) * self.xm_per_pix

                # # calculate the radius of curvature based on the lane center poly-fit, using
                # # get_radius_of_curvature_in_meters in the utils.py file.
                # # the radius of curvature is calculated at the current position (bottom of the image), so, assuming
                # # that vehicle steering is real-time, the vehicle would be steered correctly at the current position.
                # self.radius_of_curvature = get_radius_of_curvature_in_meters(center_fit, n_H,
                #                                                              self.xm_per_pix, self.ym_per_pix)

                # calculate the radius of curvature as the average of left and right lane curvatures
                self.radius_of_curvature = (self.left_lane_line.get_radius_of_curvature() +
                                            self.right_lane_line.get_radius_of_curvature()) / 2

                lane_fill_color = [0, 255, 0]
                cv2.fillPoly(lane_fill, np.array([pts], dtype=np.int32), lane_fill_color)
                # cv2.polylines(lane_fill, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=15)
                # cv2.polylines(lane_fill, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=15)
                cv2.polylines(lane_fill, np.int32([pts_center]), isClosed=False, color=(200, 0, 200), thickness=2)
            else:
                if self.left_lane_line.best_fit is None:
                    print("left best_fit is None!")
                if self.right_lane_line.best_fit is None:
                    print("right best_fit is None!")

        return lane_fill

    def process_image(self, img, l_thresh=(210, 255), b_thresh=(190, 255), display_binary_images=False):

        # TODO: INCLUDE SANITY CHECK FOR INPUT PARAMETER

        # undistort the image
        undistorted_img = self.camera_calibrator.undistort_image(img)

        # get the thresholded binary image
        # color_binary, thresh_binary = get_thresholded_binary(undistorted_img)
        # thresh_binary = get_HSV_threshold_binary(undistorted_img)

        # LAB Binary Thresholding
        _, _, _, _, thresh_binary = get_LAB_threshold_binary(undistorted_img, l_thresh=l_thresh, b_thresh=b_thresh)

        # # perspective transform
        # M, M_inv, binary_warped = get_birds_eye_view_image(thresh_binary, display_images=display_binary_images)
        # # detect lanes and fill color
        # lane_fill = self.detect_lane(binary_warped)
        # # Unwarp the lane fill image
        # lane_fill_binary_unwraped = cv2.warpPerspective(lane_fill, M_inv,
        #                                                 (lane_fill.shape[1], lane_fill.shape[0]),
        #                                                 flags=cv2.INTER_LINEAR)

        # perspective transform
        binary_warped = self.perspective_transform.get_birds_eye_view_image(thresh_binary,
                                                                            display_images=display_binary_images)
        # detect lanes and fill color
        lane_fill = self.detect_lane(binary_warped)

        # Unwarp the lane fill image
        lane_fill_binary_unwraped = self.perspective_transform.transform_image(lane_fill, unwarp=True)

        # just to keep the lane marking clean, clip the top-most horizontal fill line as it
        # may not cover the complete lane width after the inverse perspective transform
        nonzeros = np.nonzero(np.sum(lane_fill_binary_unwraped[:, :, 1], axis=1))[0]  # sum the green channel rows
        if len(nonzeros) > 0:
            clip_idx = nonzeros[0]
            lane_fill_binary_unwraped[clip_idx:clip_idx + 1, :, 1] = 0  # mask out the topmost lane-marking row
        else:
            print("WARNING: no lane fill in unwarped binary image.")

        # Draw the roc and offset data text on the image
        roc_text, vehicle_pos_text = get_data_strings(self.radius_of_curvature, self.vehicle_offset_meters)
        cv2.putText(lane_fill_binary_unwraped, roc_text, (30, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(lane_fill_binary_unwraped, vehicle_pos_text, (30, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # overlay on the undistorted image
        lane_fill_img = cv2.addWeighted(undistorted_img, 1., lane_fill_binary_unwraped, 0.3, 0.)

        return lane_fill_img