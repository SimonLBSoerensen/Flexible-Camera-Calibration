import cv2
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


def calibrate_camera_chessboard(gray_imgs, pattern_size, win_size=(10, 10), zero_zone=(-1, -1), criteria=None,
                     flags=(cv2.CALIB_RATIONAL_MODEL +
                            cv2.CALIB_THIN_PRISM_MODEL +
                            cv2.CALIB_TILTED_MODEL), verbose=0, draw_chessboards=None):
    """
    Will do a normal camera calibration. This will be done by finding the chessboards in the provided grayscale
    images.

    :param gray_imgs: Array there contains the images with chessboards. The images has to be in grayscale colorspace and in a "int" datatype
    :type gray_imgs: iterable
    :param pattern_size: Number of inner corners per a chessboard row and column
    :type pattern_size: tuple
    :param win_size: Half of the side length of the search window
    :type win_size: tuple
    :param zero_zone: Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done. It is used sometimes to avoid possible singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size.
    :type zero_zone: tuple
    :param criteria: Criteria for termination of the iterative process of corner refinement. That is, the process of corner position refinement stops either after criteria.maxCount iterations or when the corner position moves by less than criteria.epsilon on some iteration. If None the criteria will be set to (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    :type criteria: tuple
    :param flags: The flags used in cv2.calibrateCameraExtended. Default flags CALIB_RATIONAL_MODEL, CALIB_THIN_PRISM_MODEL and CALIB_TILTED_MODEL. Flages tehre can be used:
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
    :type flags: int
    :param verbose:  0, or 1. Verbosity mode. 0 = silent, 1 = progress bar and print out
    :type verbose: int
    :param draw_chessboards: The file path to save draw chessboards images. If None it will not save any
    :type draw_chessboards: str

    :returns:
    - retval: The RMS re-projection error in pixels
    - cameraMatrix: Output 3x3 floating-point camera matrix
            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1]]
    - distCoeffs: Output vector of distortion coefficients of 4, 5, 8, 12 or 14 elements
            (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    - rvecs: Rotation vectors estimated for each pattern view
    - tvecs: Translation vectors estimated for each pattern view
    - stdDeviationsIntrinsics: Standard deviations estimated for intrinsic parameters
    - stdDeviationsExtrinsics: Standard deviations estimated for extrinsic parameters
    - perViewErrors: Vector of the RMS re-projection error estimated for each pattern view
    - objPoint: Vector of vectors of calibration pattern points in the calibration pattern coordinate space
    - imgPoints: Vector of vectors of the projections of calibration pattern points
    - not_used: Vector with the indexs of the images where no markers was found

    :rtype retval: float
    :rtype cameraMatrix: ndarray
    :rtype distCoeffs: ndarray
    :rtype rvecs: ndarray
    :rtype tvecs: ndarray
    :rtype stdDeviationsIntrinsics: ndarray
    :rtype stdDeviationsExtrinsics: ndarray
    :rtype perViewErrors: ndarray
    :rtype objPoints: ndarray
    :rtype imgPoints: ndarray
    :rtype not_used: ndarray
    """

    assert isinstance(gray_imgs, Iterable), "gray_imgs has to be a iterable there consists of the grayscale images"
    assert len(pattern_size) == 2 and isinstance(pattern_size, tuple), "pattern_size has to be a tuple of length 2"
    assert len(win_size) == 2 and isinstance(win_size, tuple), "win_size has to be a tuple of length 2"
    assert len(zero_zone) == 2 and isinstance(zero_zone, tuple), "zero_zone has to be a tuple of length 2"

    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    else:
        assert isinstance(criteria, tuple), "criteria has to be a tuple"

    if draw_chessboards is not None:
        os.makedirs(draw_chessboards, exist_ok=True)

    # Object points for a chessboard
    objp = np.zeros((1, np.product(pattern_size[:2]), 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].transpose().reshape(-1, 2)

    # Arrays to hold the object points and corner point for all the chessboards
    obj_points = []
    img_points = []
    not_used = []

    if verbose == 1:
        iter = tqdm(gray_imgs, unit="image")
        print("Finding chessboard pattern in the images")
    else:
        iter = gray_imgs

    for i, gray_img in enumerate(iter):
        # Find roof corners in the images
        pattern_was_found, corners = cv2.findChessboardCorners(gray_img, pattern_size)

        # If there was a chessboard in the image
        if pattern_was_found:
            # Add object points for the chessboard
            obj_points.append(objp)

            # Find better sub pix position for the corners in the roof corners neighbourhood
            new_better_corners = cv2.cornerSubPix(gray_img, corners, win_size, zero_zone, criteria)

            if draw_chessboards is not None:
                img_first = cv2.drawChessboardCorners(gray_img.copy(), pattern_size, corners,
                                                       pattern_was_found)
                img_better = cv2.drawChessboardCorners(gray_img.copy(), pattern_size, new_better_corners,
                                                       pattern_was_found)
                cv2.imwrite(os.path.join(draw_chessboards, "{}_first.png".format(i)), img_first)
                cv2.imwrite(os.path.join(draw_chessboards, "{}_better.png".format(i)), img_better)

            # Add the better corners
            img_points.append(new_better_corners)
        else:
            not_used.append(i)
    if verbose == 1:
        print("Doing camera calibrate")

    # Do the camera calibrations from the object points and corners found in the images

    retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.calibrateCameraExtended(
        obj_points, img_points, (gray_imgs[0].shape[1], gray_imgs[0].shape[0]), cameraMatrix=None, distCoeffs=None,
        flags=flags)

    obj_points, img_points, not_used = np.array(obj_points), np.array(img_points), np.array(not_used)

    obj_points = obj_points.reshape((obj_points.shape[0], obj_points.shape[2], obj_points.shape[3]))
    obj_points_temp = np.ndarray(obj_points.shape[0], dtype='object')
    img_points_temp = np.ndarray(img_points.shape[0], dtype='object')
    for i in range(len(obj_points)):
        obj_points_temp[i] = obj_points[i]
    for i in range(len(img_points)):
        img_points_temp[i] = img_points[i]

    return retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, \
           perViewErrors, obj_points_temp, img_points_temp, not_used


def find_Charuco(gray_imgs, dictionary, board, win_size=(10, 10), zero_zone=(-1, -1), criteria=None, detectorParameters=None, verbose=0, draw=None):
    """
    This function is used to find the charuco markeres and corneres in the function calibrate_camera_charuco. Can be used outside calibrate_camera_charuco for debug
    find_Charuco(gray_imgs, dictionary, board, win_size=(10, 10), zero_zone=(-1, -1), criteria=None, verbose=0, draw=None) -> markerCorners_all, markerIds_all, charucoCorners_all, charucoIds_all, obj_points_all

    :param gray_imgs: Array there contains the images with chessboards. The images has to be in grayscale colorspace and in a "int" datatype
    :type gray_imgs: iterable
    :param dictionary: dictionary of markers indicating the type of markers. Can normally be found with cv2.aruco.getPredefinedDictionary
    :type dictionary: opencv aruco_Dictionary
    :param board: The used charuco board
    :type board: aruco_CharucoBoard
    :param win_size: Half of the side length of the search window
    :type win_size: tuple
    :param zero_zone: Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done. It is used sometimes to avoid possible singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size.
    :type zero_zone: tuple
    :param criteria: Criteria for termination of the iterative process of corner refinement. That is, the process of corner position refinement stops either after criteria.maxCount iterations or when the corner position moves by less than criteria.epsilon on some iteration. If None the criteria will be set to (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    :type criteria: tuple
    :param detectorParameters: DetectorParameters used in cv2.aruco.detectMarkers. If None default parameters will be used. Can be one detectorParameters used on all images or a list with them there shoud be used on the diffrent images
    :type detectorParameters: cv2.aruco_DetectorParameters or list of cv2.aruco_DetectorParameter
    :param verbose:  0, or 1. Verbosity mode. 0 = silent, 1 = progress bar and print out
    :type verbose: int
    :param draw: The file path to save draw charuco images. If None it will not save any
    :type draw: str

    :returns:
    - markerCorners: Vector of vectors of the charuco corners found in the images
    - marmarkerIds: Vector of vectors of the charuco corners id's found in the images
    - charucoCorners: Vector of vectors of the charuco corners found in the images
    - charucoIds: Vector of vectors of the charuco corners id's found in the images
    - obj_points: Vector of vectors of the charuco corners as object points found in the images
    - not_used: Vector with the indexs of the images where no markers was found

    :rtype markerCorners: ndarray
    :rtype marmarkerIds: ndarrayv
    :rtype charucoCorners: ndarray
    :rtype charucoIds: ndarray
    :rtype obj_points: ndarray
    :rtype not_used: ndarray
    """
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    else:
        assert isinstance(criteria, tuple), "criteria has to be a tuple"

    if detectorParameters is None:
        detectorParameters = cv2.aruco.DetectorParameters_create()

    if not isinstance(detectorParameters, list):
        detectorParameters = [detectorParameters for _ in range(len(gray_imgs))]


    charucoCorners_all = []
    charucoIds_all = []

    markerCorners_all = []
    markerIds_all = []
    obj_points_all = []

    not_used = []

    itr = enumerate(gray_imgs)
    if verbose == 1:
        print("Finding charuco features")
        itr = tqdm(itr, total=len(gray_imgs), unit="image")

    for i, img_gray in itr:
        markerCorners, markerIds, rejectedImgPoints = cv2.aruco.detectMarkers(img_gray, dictionary, parameters=detectorParameters[i])

        markers_found = len(markerCorners) > 0

        if markers_found:

            #markerCorners, markerIds, rejectedCorners, recoveredIdxs = cv2.aruco.refineDetectedMarkers(img_gray, board, markerCorners, markerIds, rejectedImgPoints)

            markerCorners_all.append(markerCorners)
            markerIds_all.append(markerIds)

            retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(markerCorners, markerIds, img_gray,
                                                                                     board)

            charuco_corners_found = charucoCorners is not None and len(charucoCorners) > 3

            if charuco_corners_found:
                new_better_charucoCorners = cv2.cornerSubPix(img_gray, charucoCorners.copy(), win_size, zero_zone,
                                                             criteria)

                ids = charucoIds.flatten()
                obj_points = board.chessboardCorners[ids]

                charucoCorners_all.append(new_better_charucoCorners)
                charucoIds_all.append(charucoIds)
                obj_points_all.append(obj_points)
                if draw is not None:
                    img_gray_draw = img_gray.copy()
                    cv2.aruco.drawDetectedMarkers(img_gray_draw, markerCorners, markerIds)
                    cv2.aruco.drawDetectedCornersCharuco(img_gray_draw, new_better_charucoCorners, charucoIds)
                    cv2.imwrite(os.path.join(draw, "{}.png".format(i)), img_gray_draw)
            else:
                not_used.append(i)
        else:
            not_used.append(i)

    return markerCorners_all, markerIds_all, charucoCorners_all, charucoIds_all, obj_points_all, not_used


def calibrate_camera_charuco(gray_imgs, squaresX, squaresY, squareLength, markerLength, dictionary, win_size=(10, 10),
                     zero_zone=(-1, -1), criteria=None, detectorParameters=None,
                     flags=(cv2.CALIB_RATIONAL_MODEL +
                            cv2.CALIB_THIN_PRISM_MODEL +
                            cv2.CALIB_TILTED_MODEL), verbose=0, draw=None):
    """
    Will do a charuco camera calibration. This will be done by finding the charuco boards in the provided grayscale images.
    -> calibrate_retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, charucoCorners_all, charucoIds_all, markerCorners_all, armarkerIds_all, obj_points_all, board
    :param gray_imgs: Array there contains the images with chessboards. The images has to be in grayscale colorspace and in a "int" datatype
    :type gray_imgs: iterable
    :param squaresX: number of chessboard squares in X direction
    :type squaresX: int
    :param squaresY: number of chessboard squares in Y direction
    :type squaresY: int
    :param squareLength: chessboard square side length (normally in meters)
    :type squareLength: float
    :param markerLength: marker side length (same unit than squareLength)
    :type markerLength: float
    :param dictionary: dictionary of markers indicating the type of markers. Can normally be found with cv2.aruco.getPredefinedDictionary
    :type dictionary: opencv aruco_Dictionary
    :param win_size: Half of the side length of the search window
    :type win_size: tuple
    :param zero_zone: Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done. It is used sometimes to avoid possible singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size.
    :type zero_zone: tuple
    :param criteria: Criteria for termination of the iterative process of corner refinement. That is, the process of corner position refinement stops either after criteria.maxCount iterations or when the corner position moves by less than criteria.epsilon on some iteration. If None the criteria will be set to (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    :type criteria: tuple
    :param detectorParameters: DetectorParameters used in cv2.aruco.detectMarkers. If None default parameters will be used. Can be one detectorParameters used on all images or a list with them there shoud be used on the diffrent images
    :type detectorParameters: cv2.aruco_DetectorParameters or list of cv2.aruco_DetectorParameter
    :param flags: The flags used in cv2.calibrateCameraExtended. Default flags CALIB_RATIONAL_MODEL, CALIB_THIN_PRISM_MODEL and CALIB_TILTED_MODEL. Flages there can be used:
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
    :type flags: int
    :param verbose:  0, or 1. Verbosity mode. 0 = silent, 1 = progress bar and print out
    :type verbose: int
    :param draw: The file path to save draw charuco images. If None it will not save any
    :type draw: str

    :returns:
    - retval: The RMS re-projection error in pixels
    - cameraMatrix: Output 3x3 floating-point camera matrix
            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1]]
    - distCoeffs: Output vector of distortion coefficients of 4, 5, 8, 12 or 14 elements
            (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    - rvecs: Rotation vectors estimated for each pattern view
    - tvecs: Translation vectors estimated for each pattern view
    - stdDeviationsIntrinsics: Standard deviations estimated for intrinsic parameters
    - stdDeviationsExtrinsics: Standard deviations estimated for extrinsic parameters
    - perViewErrors: Vector of the RMS re-projection error estimated for each pattern view
    - charucoCorners: Vector of vectors of the charuco corners found in the images
    - charucoIds: Vector of vectors of the charuco corners id's found in the images
    - markerCorners: Vector of vectors of the charuco corners found in the images
    - marmarkerIds: Vector of vectors of the charuco corners id's found in the images
    - obj_points: Vector of vectors of the charuco corners as object points found in the images
    - board: The used charuco board
    - not_used: Vector with the indexs of the images where no markers was found

    :rtype retval: float
    :rtype cameraMatrix: ndarray
    :rtype distCoeffs: ndarray
    :rtype rvecs: ndarray
    :rtype tvecs: ndarray
    :rtype stdDeviationsIntrinsics: ndarray
    :rtype stdDeviationsExtrinsics: ndarray
    :rtype perViewErrors: ndarray
    :rtype charucoCorners: ndarray
    :rtype charucoIds: ndarray
    :rtype markerCorners: ndarray
    :rtype marmarkerIds: ndarrayv
    :rtype obj_points: ndarray
    :rtype board: aruco_CharucoBoard
    :rtype not_used: ndarray
    """
    if draw is not None:
        os.makedirs(draw, exist_ok=True)

    board = cv2.aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary)

    markerCorners_all, markerIds_all, charucoCorners_all, charucoIds_all, obj_points_all, not_used = find_Charuco(gray_imgs,
                                                                                                        dictionary,
                                                                                                        board,
                                                                                                        win_size,
                                                                                                        zero_zone,
                                                                                                        criteria,
                                                                                                        detectorParameters,
                                                                                                        verbose,draw)

    if verbose == 1:
        print("Doing camera calibrate")

    img_shape = gray_imgs[0].shape[:2]

    calibrate_retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(charucoCorners_all, charucoIds_all, board, img_shape, cameraMatrix=None, distCoeffs=None, flags=flags)

    if draw is not None:
        for i, (charucoCorners, charucoIds, rvec, tvec, error) in enumerate(
                zip(charucoCorners_all, charucoIds_all, rvecs, tvecs, perViewErrors)):
            img_f = os.path.join(draw, "{}.png".format(i))
            if os.path.exists(img_f):
                img = cv2.imread(img_f)
                try:
                    cv2.aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
                except:
                    pass
                f_out = os.path.join(draw, "{}_{:0.4f}.png".format(i, error[0]))

                cv2.imwrite(f_out, img)
                os.remove(img_f)

    charucoCorners_all, charucoIds_all = np.array(charucoCorners_all), np.array(charucoIds_all)
    markerCorners_all, armarkerIds_all = np.array(markerCorners_all), np.array(markerIds_all)
    obj_points_all = np.array(obj_points_all)

    if verbose == 1:
        print("Calibration done")

    return calibrate_retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, \
           perViewErrors, charucoCorners_all, charucoIds_all, markerCorners_all, armarkerIds_all, obj_points_all, board, not_used