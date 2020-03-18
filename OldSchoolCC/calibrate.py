import cv2
import numpy as np
from tqdm import tqdm


def calibrate_camera(gray_imgs, pattern_size, win_size=(10, 10), zero_zone=(-1, -1), criteria=None):
    """
    Will do a normal camera calibration. This will be done by finding the chessboards in the provided grayscale
    images. ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(gray_imgs, pattern_size, win_size=(10,
    10), zero_zone=(-1, -1), criteria=None)

    Arguments:
    :param gray_imgs: Array there contains the images with chessboards. The images has to be in grayscale colorspace
    :param pattern_size: Number of inner corners per a chessboard row and column
    :param win_size: Half of the side length of the search window
    :param zero_zone:Half of the size of the dead region in the middle of the search zone over which the summation
    in the formula below is not done. It is used sometimes to avoid possible singularities of
    the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size.
    :param criteria: Criteria for termination of the iterative process of corner refinement. That is,
    the process of corner position refinement stops either after criteria.maxCount iterations or when
    the corner position moves by less than criteria.epsilon on some iteration.
    If None the criteria will be set to (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    Returns:
    :return ret: The RMS re-projection error in pixels
    :return camera_matrix: Output 3x3 floating-point camera matrix
    [[fx, 0,  cx],
     [0,  fy, cy],
     [0,  0,  1]]
    :return dist_coeffs: Output vector of distortion coefficients
    (8 coefficients, 6 zeros = 15 values (k1, k2, p1, p2, k3, k4, k5, k6, 0, 0, 0, 0, 0, 0))
    :return rvecs: Rotation vectors estimated for each pattern view
    :return tvecs: Translation vectors estimated for each pattern view
    """
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Object points for a chessboard
    objp = np.zeros((1, np.product(pattern_size[:2]), 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].transpose().reshape(-1, 2)

    # Arrays to hold the object points and coner point for all the chessboards
    obj_points = []
    img_points = []

    for i, gray_img in enumerate(tqdm(gray_imgs)):
        # Find roff coners in the images
        pattern_was_found, corners = cv2.findChessboardCorners(gray_img, pattern_size)

        # If there was a chessboard in the image
        if pattern_was_found:
            # Add object points for the chessboard
            obj_points.append(objp)

            # Find better sub pix position for the coners in the roff coners neighbourhood
            new_better_corners = cv2.cornerSubPix(gray_img, corners, win_size, zero_zone, criteria)

            # Add the better coners
            img_points.append(new_better_corners)

    # Do the camera calibrtions from the object points and coners found in the imagese
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,
                                                                        (gray_imgs[0].shape[1], gray_imgs[0].shape[0]),
                                                                        cameraMatrix=None, distCoeffs=None,
                                                                        flags=cv2.CALIB_RATIONAL_MODEL)
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs
