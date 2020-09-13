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
from tqdm import tqdm

train_folder = r"D:\WindowsFolders\Documents\300mm_overfit\*"

paras = {
    "cm_stepsize": 300,
    "cm_order": 2,
    "ls_ftol": 1e-8,
    "ls_gtol": 1e-8,
}

run_id = str(uuid.uuid4())
run_id = '77e16ccb-bff0-4f9b-8936-982bc5d09018'

folder_out = os.path.join("TestL", run_id)
#if os.path.exists(folder_out):
    #shutil.rmtree(folder_out)
#os.makedirs(folder_out)

with open(os.path.join(folder_out, "para.json"), "w", encoding='utf-8') as f:
    json.dump(paras, f, ensure_ascii=False, indent=4)

shutil.copy2(os.path.realpath(__file__), os.path.join(folder_out, "run_script.txt"))

imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) for f in tqdm(glob.glob(os.path.join(train_folder, "*.jpg"))[:1])]
image_shape = imgs[0].shape[:2]

[calibrate_retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, charucoCorners_all, charucoIds_all, markerCorners_all, armarkerIds_all, obj_points_all, _] = np.load(os.path.join(folder_out, "cali.npy"), allow_pickle=True)

train_split = 0.50

indeces_train = np.array([np.random.choice(list(range(len(row))), int(len(row)*train_split), replace=False) for row in obj_points_all])

obj_points_train = np.array([np.array([row[j] for j in range(len(row)) if j in indeces_train[i]]) for i, row in enumerate(obj_points_all)])
obj_points_test = np.array([np.array([row[j] for j in range(len(row)) if not j in indeces_train[i]]) for i, row in enumerate(obj_points_all)])

charucoCorners_train = np.array([np.array([row[j] for j in range(len(row)) if j in indeces_train[i]]) for i, row in enumerate(charucoCorners_all)])
charucoCorners_test = np.array([np.array([row[j] for j in range(len(row)) if not j in indeces_train[i]]) for i, row in enumerate(charucoCorners_all)])

np.save(os.path.join(folder_out, "points_test_train.npy"), [indeces_train, obj_points_train, obj_points_test, obj_points_all, charucoCorners_train, charucoCorners_test, charucoCorners_all])

ba = BundleAdjustment(obj_points_train, rvecs, tvecs, charucoCorners_train, cameraMatrix, distCoeffs, image_shape,
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
plt.savefig(os.path.join(folder_out, "CM_Voronoi.svg"))
plt.close()
