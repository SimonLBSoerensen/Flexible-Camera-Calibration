#!/usr/bin/env python
# coding: utf-8
import timeit

rep = 10

s = timeit.default_timer()
time = timeit.timeit(r"""
calibrate_retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, charucoCorners_all, charucoIds_all, markerCorners_all, \
    armarkerIds_all, obj_points_all, board, not_used = parcc.calibrate_camera_charuco(imgs, squaresX, squaresY,
                                                                                      squareLength, markerLength,
                                                                                      dictionary,
                                                                                      detectorParameters=detectorParameters)
""", r"""
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
import timeit

train_folder = r"S:\Programming\Studet\FCC\CalImgs\ChArUco - Threshold\NikonD3100\AF-S NIKKOR 55-300mm\55mm\Fold_0"

imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) for f in tqdm(glob.glob(os.path.join(train_folder, "*.jpg")))]

rep = 10


squaresX = 28 # [#]
squaresY = 19 # [#]
squareLength = 0.01 # [m]
markerLength = 0.0075 # [m]
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

detectorParameters = cv2.aruco.DetectorParameters_create()
detectorParameters.minOtsuStdDev = 12
""", number=rep)
d = timeit.default_timer()

print(time/rep)
print(d-s)

