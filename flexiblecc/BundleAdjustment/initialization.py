import numpy as np
import cv2


def grid_creation(image_shape, cameraMatrix, distCoeffs, border=0, step=400):
    image_height, image_width = image_shape

    x = np.arange(
        start=-border,
        stop=image_width + border + step,
        step=step,
        dtype=float)

    y = np.arange(
        start=-border,
        stop=image_height + border + step,
        step=step,
        dtype=float)

    src = np.transpose(np.meshgrid(x, y)).reshape((-1, 2))

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = np.array([0, 0, 0])

    Rt = np.ndarray((3, 4))
    Rt[:, :3] = R
    Rt[:, 3] = t

    P = np.dot(cameraMatrix, Rt)

    dst = cv2.undistortPoints(
        src=src,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
        P=P)[:, 0, :]

    cameraMatrixinv = np.linalg.inv(cameraMatrix)
    projected_pts = np.array([np.dot(cameraMatrixinv, np.array([[pt[0], ], [pt[1], ], [1, ]])) for pt in dst]).reshape(
        (-1, 3))

    return projected_pts
