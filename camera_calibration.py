from utils import *


# Class to calibrate a camera based on chessboard images
class ChessboardCameraCalibrator:

    def __init__(self, pattern_size=(9, 6), init_calibration=True,
                 input_images_path="camera_cal/", output_images_path="output_images/"):

        # folder with images for camera calibration
        self.input_images_path = input_images_path
        # folder for output images
        self.output_images_path = output_images_path
        # folder to output images with chessboard corners drawn
        self.corners_images_output_path = os.path.join(output_images_path, "chessboard_corners/")
        # chessboard pattern size of the input chessboard images
        self.pattern_size = pattern_size
        # arrays to store object points and image points from all the images.
        self.object_points = []  # 3d points in real world space
        self.image_points = []  # 2d points in image plane.
        # camera matrix
        self.camera_matrix = None
        # distortion coefficients
        self.distortion_coefs = None
        # saves the state, whether the camera is calibrated
        self.is_calibrated = False
        # array to store calibration image names for which no corners were found
        # (use for troubleshooting)
        self.no_corners_found = []

        # create the output folders if they don't exist
        if not os.path.exists(self.output_images_path):
            os.mkdir(self.output_images_path)
        if not os.path.exists(self.corners_images_output_path):
            os.mkdir(self.corners_images_output_path)

        if init_calibration is True:
            # calibrate the camera
            self.calibrate_camera()

    def calibrate_camera(self, recalibrate=False, save_corner_images=False):
        """
        Calibrates the camera.
        Args:
            (boolean) recalibrate: re-calibrate the camera?
            (boolean) save_corner_images: save images with corners drawn on them?
        Returns:
            True if successful, False otherwise.
        """
        if self.is_calibrated is False or recalibrate is True:

            self.is_calibrated = False
            # get the list of calibration images
            images = glob.glob(os.path.join(self.input_images_path, 'calibration*.jpg'))

            # ensure we got the images list
            if len(images) == 0:
                print("ERROR: Could not find calibration images.")
                self.is_calibrated = False
                return False

            #
            # perform calibration
            #
            pattern_size = self.pattern_size
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            # object points will be same for all calibration images we read in
            objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

            for fname in images:
                # read the image
                img = cv2.imread(fname)
                # convert image to gray scale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

                # if corners found, save object points and image points
                if ret is True:
                    self.object_points.append(objp)
                    self.image_points.append(corners)

                    # if requested, draw and save the images with corners
                    if save_corner_images is True:
                        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                        name, ext = os.path.splitext(os.path.basename(fname))
                        cv2.imwrite(os.path.join(self.corners_images_output_path, name + '_corners' + ext), img)
                else:
                    self.no_corners_found.append(fname)

            # sanity check for troubleshooting
            if len(self.no_corners_found) > 0:
                print("WARNING: Unable to find Corners for %s images - %s" %
                      (str(len(self.no_corners_found)), " - ".join(self.no_corners_found)))
            else:
                print("Found Corners for %s images." % str(len(self.image_points)))

            # check whether we have found corners for enough images to perform good calibration
            # (ideally, we should have images points for atleast 20 images)
            if len(self.image_points) == 0:
                print("ERROR: Could not find image points to continue with calibration.")
                self.is_calibrated = False
                return False

            # calibrate the camera
            img = cv2.imread(images[0])
            img_size = (img.shape[1], img.shape[0])

            # calibrate the camera given object points and image points
            retval, self.camera_matrix, self.distortion_coefs, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, img_size, None, None)

            self.is_calibrated = True
            print("Camera calibration successful with reprojection error %f (RMS)" % retval)

        else:
            print("Camera already calibrated. To recalibrate, set the 'recalibrate' parameter to True")

        return True

    def undistort_image(self, image):
        """
        Undistorts an image.
        Args:
            (cv2.Mat or ndarray) image: image to un-distort
        Returns:
            Undistorted image if successful, original image otherwise.
        """
        if not self.is_calibrated:
            self.calibrate_camera()

        # undistort the image if calibration is done
        if self.is_calibrated:
            return cv2.undistort(image, self.camera_matrix, self.distortion_coefs,
                                 None, self.camera_matrix)

        print("ERROR: Could not undistorting image as camera calibration failed. Returning original image.")

        return image


# test for camera calibration
def test_camera_calibration_and_undistortion(test_images_path="test_images/",
                                             output_images_path="output_images/"):

    print("Testing camera calibration and undistortion...")

    camera_calibrator = ChessboardCameraCalibrator(init_calibration=False)
    ret_val = camera_calibrator.calibrate_camera(save_corner_images=True)

    if ret_val is True:
        # get the list of test images
        test_images = glob.glob(os.path.join(test_images_path, '*.jpg'))

        # undistort the images and save
        for fname in test_images:
            image = cv2.imread(fname)
            dst = camera_calibrator.undistort_image(image)

            if output_images_path is not None:
                # create the output folder if it doesn't exist
                if not os.path.exists(output_images_path):
                    os.mkdir(output_images_path)

                name, ext = os.path.splitext(os.path.basename(fname))
                cv2.imwrite(os.path.join(output_images_path, name + "_undist" + ext), dst)

        print("Camera calibration and undistortion test DONE. Undistorted images saved to '%s' folder."
              % output_images_path)
    else:
        print("ERROR: Camera calibration failed.")

    return ret_val