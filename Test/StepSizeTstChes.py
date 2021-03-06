#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("../")

import flexiblecc as fcc
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import flexiblecc.Parametric as parcc
import os
import json
import shutil
from flexiblecc.CentralModel import BundleAdjustment
import uuid

train_folder = r"S:\Programming\Studet\FCC\CalImgs\Checkerboard\DTU"

paras = {
    "cm_stepsize": 252,
    "cm_order": 2,
    "ls_ftol": 1e-8,
    "ls_gtol": 1e-8,
}

run_id = str(uuid.uuid4())

folder_out = os.path.join("TestStep", run_id)
if os.path.exists(folder_out):
    shutil.rmtree(folder_out)
os.makedirs(folder_out)

with open(os.path.join(folder_out, "para.json"), "w", encoding='utf-8') as f:
    json.dump(paras, f, ensure_ascii=False, indent=4)

shutil.copy2(os.path.realpath(__file__), os.path.join(folder_out, "run_script.txt"))

img_files = glob.glob(os.path.join(train_folder, "cam_0_*.png"))
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) for f in img_files]
#retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, obj_points_temp, img_points_temp, not_used = parcc.calibrate_camera_chessboard(imgs, (12, 12))

#np.save('parametriccal', [retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, obj_points_temp, img_points_temp, not_used])

retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, obj_points_temp, img_points_temp, not_used = np.load('parametriccal_DTU.npy', allow_pickle=True)

image_shape = imgs[0].shape[:2]

obj_points_temp = obj_points_temp.reshape((obj_points_temp.shape[0], obj_points_temp.shape[2], obj_points_temp.shape[3]))
obj_points = np.ndarray(obj_points_temp.shape[0], dtype='object')
img_points = np.ndarray(img_points_temp.shape[0], dtype='object')
for i in range(len(obj_points)):
    obj_points[i] = obj_points_temp[i]
for i in range(len(img_points)):
    img_points[i] = img_points_temp[i]

print('doing BundleAdjustment')
ba = BundleAdjustment(obj_points, rvecs, tvecs, img_points, cameraMatrix, distCoeffs, image_shape,
                      cm_stepsize=paras["cm_stepsize"], cm_order=paras["cm_order"], ls_ftol=paras["ls_ftol"], ls_gtol=paras["ls_gtol"])

cm, res, rvecs_new, tvecs_new = ba.least_squares(folder_out)

fcc.CentralModel.cm_save(cm, os.path.join(folder_out, "cm"))

rmsCM, residuals_2D, estimated_points_2D, correct_points_2D = ba.calc_residuals_2D(np.array(res.x), return_points_2D=True, verbose=1)

np.save(os.path.join(folder_out, "calc_residuals_2D.npy"), [rmsCM, residuals_2D, estimated_points_2D, correct_points_2D])

rms_vs = f"{retval:0.5f} VS {rmsCM:0.5f}"
with open(os.path.join(folder_out, rms_vs+".txt"), "w") as f:
    f.write(rms_vs)
print(rms_vs)

image_points = np.concatenate(correct_points_2D)
project_points = np.concatenate(estimated_points_2D)

imp, diff, angels, mag = fcc.Metrics.voronoi.cal_angles_and_mag(image_points, project_points)

plt.figure()
fcc.Metrics.voronoi.plot_voronoi(imp, angels)
plt.savefig(os.path.join(folder_out, "CM_Voronoi.svg"))
plt.close()

#_, _, _, rvecs_test, tvecs_test, _, _, _, charucoCorners_all_test, charucoIds_all_test, _, _, obj_points_all_test = np.load(os.path.join(test_folder,"out", "cali.npy"), allow_pickle=True)
#ba_test = BundleAdjustment(obj_points_all_test, rvecs_test, tvecs_test, charucoCorners_all, cameraMatrix, distCoeffs, image_shape, cm_fit_control_points=False,
#                           control_points=ba.cm_control_points,
#                           cm_stepsize=paras["cm_stepsize"], cm_order=paras["cm_order"], ls_ftol=paras["ls_ftol"], ls_gtol=paras["ls_gtol"])
#rmsCM_test, residuals_2D_test, estimated_points_2D_test, correct_points_2D_test = ba_test.calc_residuals_2D(np.array(res.x), return_points_2D=True, verbose=1)
