#!/usr/bin/env python
# coding: utf-8
#import sys
#sys.path.append("../")

import flexiblecc as fcc
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import flexiblecc.Parametric as parcc
import os
import uuid
import json
import shutil
from flexiblecc.CentralModel import BundleAdjustment

folder_out = "TestRes"
run_name = str(uuid.uuid4())
folder_out = os.path.join(folder_out, run_name)
os.makedirs(folder_out, exist_ok=True)

import sys
#sys.stdout = open(os.path.join(folder_out, "console.txt"), 'w')

datasetpath = "../CalImgs/ChArUco - Sorted/Samsung Galaxy S10 Plus/WideAngle/Fold_1/*.jpg"

paras = {
    "cm_stepsize": 252,
    "cm_order": 2,
    "ls_ftol": 1e-8,
    "ls_gtol": 1e-8,
    "datasetpath":datasetpath,
}

print("paras:", paras)

with open(os.path.join(folder_out, "para.json"), "w", encoding='utf-8') as f:
    json.dump(paras, f, ensure_ascii=False, indent=4)

shutil.copy2(os.path.realpath(__file__), os.path.join(folder_out, "run_script.txt"))

image_files = glob.glob(paras["datasetpath"])

color_images = [cv2.imread(f) for f in tqdm(image_files)]
gayscale_images = [cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY) for c_img in tqdm(color_images)]

image_shape = color_images[0].shape[:2]

squaresX = 28  # [#]
squaresY = 19  # [#]
squareLength = 0.01  # [m]
markerLength = 0.0075  # [m]
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

calibrate_retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, charucoCorners_all, charucoIds_all, markerCorners_all, armarkerIds_all, obj_points_all, board = parcc.calibrate_camera_charuco(
    gayscale_images, squaresX, squaresY,
    squareLength, markerLength, dictionary, verbose=1)

plt.figure()
fcc.Metrics.rtvecs_illustration.draw_rtvecs(rvecs, tvecs, obj_points_all)
plt.tight_layout()
plt.savefig(os.path.join(folder_out, "ParBoards.png"))
plt.close()

img_points_all, diff_all, angels_all, mag_all = fcc.Metrics.voronoi.projectPoints_and_cal_angles_and_mag(
    charucoCorners_all, obj_points_all, rvecs, tvecs, cameraMatrix, distCoeffs)

plt.figure()
fcc.Metrics.voronoi.plot_voronoi(img_points_all, angels_all)
plt.savefig(os.path.join(folder_out, "Par_Voronoi.png"))
plt.close()


print(f"RMS: {calibrate_retval:0.4f} pixels")



ba = BundleAdjustment(obj_points_all, rvecs, tvecs, charucoCorners_all, cameraMatrix, distCoeffs, image_shape,
                      cm_stepsize=paras["cm_stepsize"], cm_order=paras["cm_order"], ls_ftol=paras["ls_ftol"], ls_gtol=paras["ls_gtol"])

cm, res, rvecs_new, tvecs_new = ba.least_squares(folder_out)

fcc.CentralModel.cm_save(cm, os.path.join(folder_out, "cm"))


rmsCM, residuals_2D, estimated_points_2D, correct_points_2D = ba.calc_residuals_2D(np.array(res.x), return_points_2D=True, verbose=1)

np.save(os.path.join(folder_out, "calc_residuals_2D.npy"), [rmsCM, residuals_2D, estimated_points_2D, correct_points_2D])

rms_vs = f"{calibrate_retval:0.5f} VS {rmsCM:0.5f}"
with open(os.path.join(folder_out, rms_vs+".txt"), "w") as f:
    f.write(rms_vs)
print(rms_vs)

image_points = np.concatenate(correct_points_2D)
project_points = np.concatenate(estimated_points_2D)

imp, diff, angels, mag = fcc.Metrics.voronoi.cal_angles_and_mag(image_points, project_points)

plt.figure()
fcc.Metrics.voronoi.plot_voronoi(imp, angels)
plt.savefig(os.path.join(folder_out, "CM_Voronoi.png"))
plt.close()

#sys.stdout.close()