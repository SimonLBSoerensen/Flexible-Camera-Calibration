import cv2
import numpy as np
from tqdm import tqdm
from collections import Iterable


def calibrate_camera(gray_imgs, pattern_size, win_size=(10, 10), zero_zone=(-1, -1), criteria=None,
                     calibrate_camera_flags=(cv2.CALIB_RATIONAL_MODEL +
                                             cv2.CALIB_THIN_PRISM_MODEL +
                                             cv2.CALIB_TILTED_MODEL), verbose=0):
    """
    Will do a normal camera calibration. This will be done by finding the chessboards in the provided grayscale
    images.

    Arguments:
        gray_imgs (iterable): Array there contains the images with chessboards. The images has to be in grayscale colorspace and in a "int" datatype
        pattern_size (tuple): Number of inner corners per a chessboard row and column
        win_size (tuple): Half of the side length of the search window
        zero_zone (tuple): Half of the size of the dead region in the middle of the search zone over which the summation
            in the formula below is not done. It is used sometimes to avoid possible singularities of
            the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size.
        criteria (tuple): Criteria for termination of the iterative process of corner refinement. That is,
            the process of corner position refinement stops either after criteria.maxCount iterations or when
            the corner position moves by less than criteria.epsilon on some iteration.
            If None the criteria will be set to (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        calibrate_camera_flags (int): The flags used in cv2.calibrateCameraExtended. Default flags CALIB_RATIONAL_MODEL, CALIB_THIN_PRISM_MODEL and CALIB_TILTED_MODEL. Flages tehre can be used:
            CALIB_USE_INTRINSIC_GUESS cameraMatrix contains valid initial values of fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image center ( imageSize is used), and focal distances are computed in a least-squares fashion. Note, that if intrinsic parameters are known, there is no need to use this function just to estimate extrinsic parameters. Use solvePnP instead.
            CALIB_FIX_PRINCIPAL_POINT The principal point is not changed during the global optimization. It stays at the center or at a different location specified when CALIB_USE_INTRINSIC_GUESS is set too.
            CALIB_FIX_ASPECT_RATIO The functions considers only fy as a free parameter. The ratio fx/fy stays the same as in the input cameraMatrix . When CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are ignored, only their ratio is computed and used further.
            CALIB_ZERO_TANGENT_DIST Tangential distortion coefficients (p1,p2) are set to zeros and stay zero.
            CALIB_FIX_K1,...,CALIB_FIX_K6 The corresponding radial distortion coefficient is not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
            CALIB_RATIONAL_MODEL Coefficients k4, k5, and k6 are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the rational model and return 8 coefficients. If the flag is not set, the function computes and returns only 5 distortion coefficients.
            CALIB_THIN_PRISM_MODEL Coefficients s1, s2, s3 and s4 are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the thin prism model and return 12 coefficients. If the flag is not set, the function computes and returns only 5 distortion coefficients.
            CALIB_FIX_S1_S2_S3_S4 The thin prism distortion coefficients are not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
            CALIB_TILTED_MODEL Coefficients tauX and tauY are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the tilted sensor model and return 14 coefficients. If the flag is not set, the function computes and returns only 5 distortion coefficients.
            CALIB_FIX_TAUX_TAUY The coefficients of the tilted sensor model are not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
        verbose (int): 0, or 1. Verbosity mode.
                        0 = silent, 1 = progress bar and print out
    Returns:
        retval (float): The RMS re-projection error in pixels
        cameraMatrix (ndarray): Output 3x3 floating-point camera matrix
            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1]]
        distCoeffs (ndarray): Output vector of distortion coefficients of 4, 5, 8, 12 or 14 elements
            (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
        rvecs (ndarray): Rotation vectors estimated for each pattern view
        tvecs (ndarray): Translation vectors estimated for each pattern view
        stdDeviationsIntrinsics (): Standard deviations estimated for intrinsic parameters
        stdDeviationsExtrinsics (): Standard deviations estimated for extrinsic parameters
        perViewErrors (): Vector of the RMS re-projection error estimated for each pattern view
        objPoints (ndarray): Vector of vectors of calibration pattern points in the calibration pattern coordinate space
        imgPoints (ndarray): Vector of vectors of the projections of calibration pattern points
    """

    assert isinstance(gray_imgs, Iterable), "gray_imgs has to be a iterable there consists of the grayscale images"
    assert len(pattern_size) == 2 and isinstance(pattern_size, tuple), "pattern_size has to be a tuple of length 2"
    assert len(win_size) == 2 and isinstance(win_size, tuple), "win_size has to be a tuple of length 2"
    assert len(zero_zone) == 2 and isinstance(zero_zone, tuple), "zero_zone has to be a tuple of length 2"

    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    else:
        assert isinstance(criteria, tuple), "criteria has to be a tuple"

    # Object points for a chessboard
    objp = np.zeros((1, np.product(pattern_size[:2]), 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].transpose().reshape(-1, 2)

    # Arrays to hold the object points and coner point for all the chessboards
    obj_points = []
    img_points = []

    if verbose == 1:
        iter = tqdm(gray_imgs)
        print("Finding chessboard pattern in the images")
    else:
        iter = gray_imgs

    for i, gray_img in enumerate(iter):
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

    if verbose == 1:
        print("Doing camera calibrate")

    # Do the camera calibrtions from the object points and coners found in the imagese

    retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.calibrateCameraExtended(obj_points, img_points, (gray_imgs[0].shape[1], gray_imgs[0].shape[0]), cameraMatrix=None, distCoeffs=None, flags=calibrate_camera_flags)

    return retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, np.array(obj_points), np.array(img_points)
