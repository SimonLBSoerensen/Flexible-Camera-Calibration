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

folders = []
#folders += glob.glob(r"../CalImgs\ChArUco - Threshold\InitTest/*")
folders += [r"../CalImgs\ChArUco - Threshold\NikonD3100\AF-S NIKKOR 18-55mm\18mm"]
folders += glob.glob(r"../CalImgs\ChArUco - Threshold\NikonD3100\AF-S NIKKOR 18-55mm\22mm/*")
#folders += glob.glob(r"../CalImgs\ChArUco - Threshold\NikonD3100\AF-S NIKKOR 55-300mm/*/*")
#folders += glob.glob(r"../CalImgs\ChArUco - Threshold\Samsung Galaxy S10 Plus/*/*")
print(folders)

paras = {
    "cm_stepsize": 252,
    "cm_order": 2,
    "ls_ftol": 1e-8,
    "ls_gtol": 1e-8,
}

for folder in tqdm(folders):
    folder_out = os.path.join(folder, "out/CM")
    if os.path.exists(folder_out):
        shutil.rmtree(folder_out)
    os.makedirs(folder_out)

    with open(os.path.join(folder_out, "para.json"), "w", encoding='utf-8') as f:
        json.dump(paras, f, ensure_ascii=False, indent=4)

    shutil.copy2(os.path.realpath(__file__), os.path.join(folder_out, "run_script.txt"))

    calibrate_retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, \
        perViewErrors, charucoCorners_all, charucoIds_all, markerCorners_all, \
        armarkerIds_all, obj_points_all = np.load(os.path.join(folder,"out", "cali.npy"), allow_pickle=True)

    image_shape = cv2.imread(glob.glob(os.path.join(folder, "*.jpg"))[0]).shape[:2]

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
    plt.savefig(os.path.join(folder_out, "CM_Voronoi.svg"))
    plt.close()
