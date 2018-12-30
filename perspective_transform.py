from utils import *


# Class to perform perspective transform
class PerspectiveTransform:

    def __init__(self, source_points=None, destination_points=None):

        if source_points is None or destination_points is None:
            self.src = np.float32([[210, 720],
                                   [596, 450],
                                   [684, 450],
                                   [1100, 720]])
            self.dst = np.float32([[320, 720],
                                   [320, 0],
                                   [960, 0],
                                   [960, 720]])
        else:
            self.src = source_points
            self.dst = destination_points

        # get the perspective transform map matrices
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

        print("PerspectiveTransform Initialized")

    def get_src_dst_points(self):
        """
        Returns the source and destination points used for perspective transform.
        """
        return self.src, self.dst

    def get_map_matrices(self):
        """
        Returns the M and M-Inverse perspective transform map matrices.
        """
        return self.M, self.M_inv

    def transform_image(self, image, unwarp=False):
        """
        Returns the birds-eye (perspective transformed) view of the input image.
        Args:
            (cv2.Mat or ndarray) image: image to transform, either RGB image (H,W,Ch)
                                        or single-channel binary image.
            (boolean) unwarp: whether to warp (False) or unwarp (True) the image.
        Returns:
            Either the Warped (birds-eye) view or Un-warped view of the input image.
        """
        map_matrix = self.M_inv if unwarp is True else self.M

        # return the warped or unwarped image
        return cv2.warpPerspective(image, map_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def get_birds_eye_view_image(self, image, display_images=False):
        """
        Returns the birds-eye (perspective transformed) view of the input image.
        Args:
            (cv2.Mat or ndarray) image: image to transform, either RGB image (H,W,Ch)
                                        or single-channel binary image.
            (boolean) display_images: display image transformations?
        Returns:
            Birds-eye (Warped) view image
        """

        # check whether input is a single-channel binary image
        is_binary = True if len(image.shape) == 2 else False

        # get the warped image
        warped_image = self.transform_image(image)

        if is_binary is True:
            warped_image *= 255

        if display_images is True:
            if is_binary is True:
                image_src_poly = np.dstack([image, image, image]) * 255
            else:
                image_src_poly = np.copy(image)

            # source polygon
            cv2.polylines(image_src_poly, [self.src.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

            # convert to 3-channel binary
            if is_binary is True:
                warped_image_dst_lines = np.dstack([warped_image, warped_image, warped_image])
            else:
                warped_image_dst_lines = np.copy(warped_image)

            # destination left line
            cv2.line(warped_image_dst_lines,
                     (self.dst[0][0], self.dst[0][1]), (self.dst[1][0], self.dst[1][1]),
                     (255, 0, 0), 5)
            # destination right line
            cv2.line(warped_image_dst_lines,
                     (self.dst[2][0], self.dst[2][1]), (self.dst[3][0], self.dst[3][1]),
                     (255, 0, 0), 5)

            plot_images([image, image_src_poly, warped_image, warped_image_dst_lines],
                        titles=["Image", "Image - Source Points",
                                "Warped Image", "Warped Image - Dest. Points"],
                        cols=2, fontsize=16)

        return warped_image


# Test function to visually test perspective transform
def visually_verify_perspective_transform(image, src=None, dst=None, display_images=False):

    perspective_transform = PerspectiveTransform(source_points=src, destination_points=dst)
    src, dst = perspective_transform.get_src_dst_points()

    # check whether input is a single-channel binary image
    is_binary = True if len(image.shape) == 2 else False

    # get the warped image
    warped_image = perspective_transform.transform_image(image)

    if is_binary is True:
        warped_image *= 255

    if is_binary is True:
        image_src_poly = np.dstack([image, image, image]) * 255
    else:
        image_src_poly = np.copy(image)

    # source polygon
    cv2.polylines(image_src_poly, [src.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    # convert to 3-channel binary
    if is_binary is True:
        warped_image_dst_lines = np.dstack([warped_image, warped_image, warped_image])
    else:
        warped_image_dst_lines = np.copy(warped_image)

    # destination left line
    cv2.line(warped_image_dst_lines,
             (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]),
             (255, 0, 0), 5)
    # destination right line
    cv2.line(warped_image_dst_lines,
             (dst[2][0], dst[2][1]), (dst[3][0], dst[3][1]),
             (255, 0, 0), 5)

    if display_images is True:
        plot_images([image, image_src_poly, warped_image, warped_image_dst_lines],
                    titles=["Image", "Image - Source Points",
                            "Warped Image", "Warped Image - Dest. Points"],
                    cols=2, fontsize=16)

    return image_src_poly, warped_image_dst_lines
