import numpy as np 
import matplotlib.pyplot as plt
import flexiblecc.Metrics.stats as stats
import json
import cv2

calibrate_retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, \
           perViewErrors, charucoCorners_all, charucoIds_all, markerCorners_all, armarkerIds_all, obj_points_all = np.load('cali.npy', allow_pickle=True)


for i,r in enumerate(rvecs):
    rotation_matrix = cv2.Rodrigues(r)
    


with open(r'C:\Users\Jakob\Documents\GitHub\Flexible-Camera-Calibration\flexiblecc\BundleAdjustment\residuals_2D_2020-05-07_13-44-31.json') as f:
    data = json.load(f)

r = 30

data = np.array(data).reshape((-1,2))

data_without_outliers = np.array([[x,y] for x,y in data if -30 < x < 30 and -30 < y < 30])

outlier_indeces = np.array([i for i,[x,y] in enumerate(data) if np.sqrt(x**2+y**2) > r])

#plt.scatter(data[:,0], data[:,1])
#plt.scatter(data_without_outliers[:,0], data_without_outliers[:,1])

stats.plot_model_check(data_without_outliers)

plt.show()