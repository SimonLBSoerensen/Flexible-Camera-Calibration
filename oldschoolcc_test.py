import os
import flexiblecc as fcc


datasetpath = "CalImgs/cam_0*"



import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


color_cal_imgs_files = glob.glob(datasetpath)
color_cal_imgs = np.array([cv2.imread(f) for f in tqdm(color_cal_imgs_files)])

plt.figure()
plt.imshow(cv2.cvtColor(color_cal_imgs[0], cv2.COLOR_BGR2RGB))
plt.show()

gray_cal_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in tqdm(color_cal_imgs)]

plt.figure()
plt.imshow(gray_cal_imgs[0], cmap="gray")
plt.show()


pattern_size = (12, 12)

retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, obj_points, img_points = fcc.OldSchoolCC.calibrate_camera(gray_cal_imgs, pattern_size, verbose=1, draw_chessboards="Out")

print("RMS re-projection error in pixels", retval)
print("3x3 camera matrix:", cameraMatrix)
print("Distortion coefficients", distCoeffs)
print(f"Rotation vectors estimated for each pattern view. There are {len(rvecs)}. Example:", rvecs[0])
print(f"Translation vectors estimated for each pattern view. There are {len(tvecs)}. Example:", tvecs[0])
print(f"Standard deviations estimated for intrinsic parameters. There are {len(stdDeviationsIntrinsics)}. Example:", stdDeviationsIntrinsics[0])
print(f"Standard deviations estimated for extrinsic parameters. There are {len(stdDeviationsExtrinsics)}. Example:", stdDeviationsExtrinsics[0])
print(f"Vector of the RMS re-projection error estimated for each pattern view. There are {len(perViewErrors)}. Example:", perViewErrors[0])


def undistort(points):
    return cv2.undistortPoints(points, cameraMatrix, distCoeffs, P=cameraMatrix).reshape(-1, 2)

fcc.Metrics.plot_distort(undistort, gray_cal_imgs[0].shape, contour_n_levels=20)



